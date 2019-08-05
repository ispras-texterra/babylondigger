import os
import pickle
import sys
from itertools import chain
from typing import Iterator, Iterable, List

from babylondigger.datamodel import Document
from babylondigger.neural.model import Model
from babylondigger.neural.tagger.network_provider import get_nn_builder
from babylondigger.tagger import TaggerInterface, TaggerTrainerInterface
from babylondigger.features.provider import FeatureExtractorBuilderProvider


class NeuralTagger(TaggerInterface):
    """
        Tagger that uses neural network models for prediction.
    """

    def __init__(self, feature_extractor, model, interpreter):
        self.__feature_extractor = feature_extractor
        self.__model = model
        self.__interpreter = interpreter

    def tag(self, documents: Iterator[Document]) -> Iterator[Document]:
        for document in documents:
            annotated_tokens = []
            for sentence in document.sentences:
                input_features = self.__feature_extractor.get_features([sentence])
                prediction = self.__model.predict(input_features)
                annotated_tokens += chain.from_iterable(self.__interpreter.interpret([sentence], prediction))

            yield Document.from_document(document, annotated_tokens)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(_feature_extractor_path(path), 'wb') as f:
            pickle.dump((self.__feature_extractor, self.__interpreter), f)
        self.__model.save(path)
        return LazyNeuralTagger(path)

    @classmethod
    def load(cls, path):
        with open(_feature_extractor_path(path), 'rb') as f:
            feature_extractor, interpreter = pickle.load(f)
        model = Model.load(path)
        return NeuralTagger(feature_extractor, model, interpreter)

    def close(self):
        self.__model.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _feature_extractor_path(path):
    return os.path.join(path, 'extractor-interpreter')


class LazyNeuralTagger(TaggerInterface):

    def __init__(self, path):
        self.__path = path

    def tag(self, documents: Iterator[Document]):
        with NeuralTagger.load(self.__path) as tagger:
            for result in tagger.tag(documents):
                yield result

    def __str__(self):
        return "NeuralTagger[{}]".format(self.__path)


class PretrainedTaggerTrainer(TaggerTrainerInterface):
    def train(self, documents: Iterable[Document], dev_tester=None, **kwargs) -> List[TaggerInterface]:
        path = kwargs['model_dir']
        models = sorted(os.path.join(root, dir) for root, dirs, _ in os.walk(path) for dir in dirs)
        taggers = []
        if dev_tester is not None:
            for model_path in models:
                tagger = LazyNeuralTagger(model_path)
                yield dev_tester(tagger), tagger
                taggers.append(tagger)
        return taggers


class NeuralTaggerTrainer(TaggerTrainerInterface):

    def train(self, documents: Iterable[Document], dev_tester=None, model_name_format='epoch-{}', **kwargs) -> List[TaggerInterface]:
        seed = kwargs['seed']

        nn_config = kwargs['nn_config']
        task_nn_builders = [nn_config[task] for task in kwargs['tasks']]
        nn_builder = get_nn_builder(nn_config['shared_part'], task_nn_builders, nn_config['general'], seed)
        sparse_input_prob = nn_config['shared_part']['config'].get('sparse_input_prob', 0)

        features_provider = FeatureExtractorBuilderProvider(nn_builder.feature_providers())
        feature_extractor, label_extractor, interpreter, initializer = features_provider.build(documents)
        network, tf_graph = nn_builder.build(initializer.shapes())

        taggers = []
        with Model.new(network, tf_graph) as model:

            model.initialize(**initializer.initial_values())
            optimizer = nn_config['general']['optimizer']

            lr = optimizer['initial_lr']
            _previous_accs = [0] * 5
            for epoch in range(kwargs['epochs']):
                print("epoch {:} out of {:}".format(epoch + 1, kwargs['epochs']))

                lr = _get_lr(epoch, lr, optimizer['lr_decay'])

                _avg_loss = _train_epoch(documents, model, feature_extractor, label_extractor,
                                              lr, kwargs['dropout'], kwargs['batch_size'], sparse_input_prob)  # train one epoch

                print('-- loss after epoch {}:\t{:.5f}'.format(epoch + 1, _avg_loss))

                path = os.path.join(kwargs['model_dir'], model_name_format.format(epoch))
                tagger = NeuralTagger(feature_extractor, model, interpreter).save(path)  # save tagger after epoch

                if dev_tester is not None:
                    scores = dev_tester(tagger)
                    acc = sum(score.aligned_accuracy for score in scores.values() if score.aligned_accuracy is not None)
                    if acc <= _previous_accs[-1] and acc <= sum(_previous_accs) / len(_previous_accs):
                        lr /= optimizer['lr_decrease_factor']

                    _previous_accs.append(acc)
                    _previous_accs.pop(0)

                    yield scores, tagger

                taggers.append(tagger)

        return taggers


def _train_epoch(documents: Iterable[Document], model, feature_extractor, label_extractor, lr, dropout, batch_size, sparse_rate):
    loss = 0
    batch_count = 0
    for features, labels in _get_minibatches(documents, feature_extractor, label_extractor, batch_size, sparse_rate):
        labels['learning_rate'] = lr
        labels['dropout_probability'] = dropout
        batch_loss = model.train(features, labels)
        loss += batch_loss
        batch_count += 1
        sys.stdout.write('\r-- batches processed: {}'.format(batch_count))
        sys.stdout.flush()
    print()

    return loss / batch_count


def _get_lr(epoch, inital_value=0.01, lr_decay=0.05):
    return inital_value / (1 + lr_decay * epoch)


def _get_minibatches(documents: Iterable[Document], feature_extractor, label_extractor, batch_size, sparse_rate):

    def extract():
        return feature_extractor.get_features(sents, sparse_rate=sparse_rate), \
               label_extractor.get_features(sents)

    sents = []
    for document in documents:
        for sent in document.sentences:
            sents.append(sent)
            if len(sents) >= batch_size:
                yield extract()
                sents = []

    if len(sents) > 0:
        yield extract()