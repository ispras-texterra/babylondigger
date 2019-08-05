from typing import Iterator, Iterable, Tuple, Dict
from os import path

import tensorflow as tf
import numpy as np

from babylondigger.datamodel import Document
from babylondigger.tagger import TaggerInterface
from babylondigger.parser import ParserInterface
from babylondigger.processor import DocumentProcessorInteface
from babylondigger.evaluation.conllu_io import to_conll, default_column_extractors, default_line_parser

import tdozat_parser.parser as parser
from tdozat_parser.parser.config import Config
from tdozat_parser.parser.structs.conllu_dataset import CoNLLUDataset
from tdozat_parser.parser.graph_outputs import GraphOutputs


class TDozatDocumentProcessor(DocumentProcessorInteface):
    def __init__(self, save_metadir: str, config_file_name: str, extra_fields: Dict[str, Tuple[str, ...]] = None):
        # following code is copy-pasted from run method of tdozat_parser.main#run
        kwargs = {'DEFAULT': {}}
        config_file = path.join(save_metadir, config_file_name)
        kwargs['DEFAULT']['save_metadir'] = save_metadir
        self.__config = Config(defaults_file='', config_file=config_file, **kwargs)

        network_class_name = self.__config.get('DEFAULT', 'network_class')
        self.__network_class = getattr(parser, network_class_name)

        self.__g = tf.Graph()
        with self.__g.as_default():
            self.__network = self.__network_class(config=self.__config)

        # following code is copy-pasted from tdozat_parser.parser.base_network.BaseNetwork#parse
        factored_deptree = None
        factored_semgraph = None
        for vocab in self.__network.output_vocabs:
            if vocab.field == 'deprel':
                factored_deptree = vocab.factorized
            elif vocab.field == 'semrel':
                factored_semgraph = vocab.factorized

        with self.__g.as_default():
            with tf.variable_scope(self.__network.classname, reuse=False):
                outputs, tokens = self.__network.build_graph(reuse=True)
                self.parse_outputs = GraphOutputs(outputs, tokens, load=False, factored_deptree=factored_deptree, factored_semgraph=factored_semgraph, config=self.__config)
        self.__extra_fields = {}
        if extra_fields is not None:
            for field_name, field_path in extra_fields.items():
                output = outputs
                for key in field_path:
                    if isinstance(output, dict) and key in output:
                        output = output[key]
                    else:
                        raise ValueError(f"there is no {field_path} tensor in the model (specify one of {outputs})")
                self.__extra_fields[field_name] = output
        self._sess = None

    def process(self, documents: Iterator[Document]) -> Iterator[Document]:
        if self._sess is None:
            raise RuntimeError(f'Use {self.__name__} as context manager')
        documents_list = list(documents)
        return iter(self.__parse_documents(documents_list))

    def _validate_network_class(self, network_class):
        return self.__network_class is network_class

    def __enter__(self):
        with self.__g.as_default():
            # following code is copy-pasted from tdozat_parser.parser.base_network.BaseNetwork#parse
            all_variables = set(tf.global_variables())
            non_save_variables = set(tf.get_collection('non_save_variables'))
            save_variables = all_variables - non_save_variables
            saver = tf.train.Saver(list(save_variables), max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self._sess = tf.Session(graph=self.__g, config=config)
        self._sess.run(tf.variables_initializer(list(non_save_variables)))
        saver.restore(self._sess, tf.train.latest_checkpoint(self.__network.save_dir))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sess.close()
        self._sess = None

    def __parse_documents(self, documents: Iterable[Document]):
        for vocab in self.__network.vocabs:
            vocab.reset()

        dataset = _DocumentDataset(documents, self.__network.vocabs, config=self.__config)
        # following code is copy-pasted from tdozat_parser.parser.base_network.BaseNetwork#parse_file
        probability_tensors = self.parse_outputs.probabilities
        for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
            tokens, lengths = dataset.get_tokens(indices)
            feed_dict = dataset.set_placeholders(indices)
            probabilities, extras = self._sess.run((probability_tensors, self.__extra_fields), feed_dict=feed_dict)
            predictions = self.parse_outputs.probs_to_preds(probabilities, lengths)
            tokens.update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.__network.output_vocabs})
            if extras:
                tokens.update({extra_field: extra_value for extra_field, extra_value in extras.items()})
            self.parse_outputs.cache_predictions(tokens, indices)

        # following code is similar to tdozat_parser.parser.graph_outputs.GraphOutputs#print_current_predictions
        order = np.argsort(self.parse_outputs.predictions['indices'])
        fields = ['form', 'lemma', 'upos', 'xpos', 'ufeats', 'dephead', 'deprel', 'semrel', 'misc']
        result = []
        _i = 0
        for document in documents:
            result_tokens = []
            for sentence in document.sentences:
                i = order[_i]
                for j, token in enumerate(sentence, 1):
                    token_line = [self.parse_outputs.predictions['id'][i][j]]
                    for field in fields:
                        if field in self.parse_outputs.predictions:
                            token_line.append(self.parse_outputs.predictions[field][i][j])
                        else:
                            token_line.append('_')
                    extras = {}
                    for field in self.__extra_fields:
                        extras[field] = self.parse_outputs.predictions[field][i][j]
                    int_token = default_line_parser(token_line)
                    token = token.with_morphology(token, pos=int_token.pos, lemma=int_token.lemma)
                    token = token.with_syntax(token, deprel=int_token.deprel, head_sentence_index=self.__compute_head(int_token.head))
                    token = token.update_extras(token, updates=extras)
                    result_tokens.append(token)
                _i += 1
            result.append(document.from_document(document, result_tokens))
        self.parse_outputs.predictions = {'indices': []}
        return result

    @classmethod
    def __compute_head(cls, parsed_head):
        if parsed_head is None:
            return None
        if parsed_head == 0:
            return -1
        return parsed_head - 1


class TDozatParser(TDozatDocumentProcessor, ParserInterface):
    def __init__(self, model_path: str, *, parser_recur_layer: bool = False, extra_fields: Dict[str, Tuple[str, ...]] = None):
        merged_extra_fields = {}
        if parser_recur_layer:
            merged_extra_fields['parser_recur_layer'] = ('deptree', 'recur_layer')
        if extra_fields:
            merged_extra_fields.update(extra_fields)
        TDozatDocumentProcessor.__init__(self, model_path, 'ParserNetwork.cfg', merged_extra_fields)
        if not self._validate_network_class(parser.ParserNetwork):
            raise ValueError('Incorrect model configuration')
        self.parse = self.process


class TDozatTagger(TDozatDocumentProcessor, TaggerInterface):
    def __init__(self, model_path: str, *, tagger_recur_layer: bool = False, upos_hidden: bool = False,
                 xpos_hidden: bool = False, extra_fields: Dict[str, Tuple[str, ...]] = None):
        merged_extra_fields = {}
        if tagger_recur_layer:
            merged_extra_fields['tagger_recur_layer'] = ('upos', 'recur_layer')
        if upos_hidden:
            merged_extra_fields['upos_hidden'] = ('upos', 'hidden_layer')
        if xpos_hidden:
            merged_extra_fields['xpos_hidden'] = ('xpos', 'hidden_layer')
        if extra_fields:
            merged_extra_fields.update(extra_fields)
        TDozatDocumentProcessor.__init__(self, model_path, 'TaggerNetwork.cfg', merged_extra_fields)
        if not self._validate_network_class(parser.TaggerNetwork):
            raise ValueError('Incorrect model configuration')
        self.tag = self.process


class _DocumentDataset(CoNLLUDataset):

    def __init__(self, documents: Iterable[Document], vocabs, config=None):
        CoNLLUDataset.__init__(self, documents, vocabs, config)

    @staticmethod
    def itersents(conllu_file: Document):
        result = []
        for sentence in conllu_file.sentences:
            for index, token in enumerate(sentence):
                result.append(to_conll(default_column_extractors, index, token, fake_heads=False))
            yield result
            result = []
        yield result
