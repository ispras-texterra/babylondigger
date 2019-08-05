from collections import Counter, defaultdict, OrderedDict
from itertools import chain
from typing import Dict, List, Tuple, Iterator, Iterable, Callable, TypeVar, Any, Generic

from bpemb import BPEmb
import morfessor as mf

from babylondigger.datamodel import Document, NavigableSentence, NavigableToken, Pos, Token, DocumentToken
import babylondigger.features.builder as builder
from babylondigger.features.builder import OOV, PADDING, START, END

from babylondigger.lemmas.transformation_rules import LemmaSuffixTransformation, LemmaTransformationTrie


########################### Sentence Length ##########################


class SentenceLengthFeatureExtractorBuilder(builder.FeatureExtractorBuilder[NavigableSentence, int]):
    """
    Sentence-level feature extractor builder to obtain length of sentence feature extractor.
    """

    def __init__(self, feature_name: str):
        self.__feature_name = feature_name

    def build(self, dataset: Iterable[Document]):
        return SentenceLengthFeatureExtractor(
            self.__feature_name), builder.DoNothingInterpreter(), builder.EmptyFeatureInitializer()


class SentenceLengthFeatureExtractor(builder.FeatureExtractorInterface[NavigableSentence, int]):
    def __init__(self, feature_name):
        self.__feature_name = feature_name

    def get_features(self, obj: NavigableSentence, **kwargs) -> Dict[str, int]:
        return {self.__feature_name: obj.tokens_count}


########################### Word Embedding ###########################


class WordEmbeddingFeatureExtractorBuilder(builder.FeatureExtractorBuilder[NavigableToken, int]):
    """
    Token-level feature extractor builder for word embeddings features.

    It uses word2vec like file for initialization (no dataset).

    It constructs vocabulary (enumerates words in specified file) and creates initializer for embedding layer variable.

    Feature is index of word in vocabulary
    """

    def __init__(self, feature_name: str, initializer_name: str, path: str, lowercase: bool):
        self.__feature_name = feature_name
        self.__initializer_name = initializer_name
        self.__path = path
        self.__lowercase = lowercase

    def build(self, dataset: Iterable[Document]):
        word2vec, embed_size = _read_embeddings(self.__path)
        word2id = {word: index for index, word in enumerate(word2vec, 2)}

        return WordEmbeddingFeatureExtractor(self.__feature_name, word2id, self.__lowercase), \
               builder.DoNothingInterpreter(), \
               EmbeddingInitializer(self.__initializer_name, word2id, word2vec, embed_size, len(word2id) + 2)


def _read_embeddings(path: str) -> Tuple[Dict[str, List[float]], int]:
    embeddings = {}
    embed_size = None
    with open(path, 'r', encoding='utf8') as src:
        line_nr = 0
        for line in src:
            values = line.split()
            if line_nr == 0 and len(values) == 2:
                embed_size = int(values[1])
            else:
                key = values[0]
                embeddings[key] = list(map(float, values[1:]))
                if embed_size is not None:
                    if len(embeddings[key]) != embed_size:
                        print(path, line_nr)
                        raise ValueError('Inconsistent vector sizes!')
                else:
                    embed_size = len(embeddings[key])
            line_nr += 1

    return embeddings, embed_size


class WordEmbeddingFeatureExtractor(builder.FeatureExtractorInterface[NavigableToken, int]):
    def __init__(self, feature_name, word2id: Dict[str, int], lowercase):
        self.__feature_name = feature_name
        self.__word2id = word2id
        self.__lowercase = lowercase

    def get_features(self, obj: NavigableToken, **kwargs) -> Dict[str, int]:
        obj_text = obj.text.lower() if self.__lowercase else obj.text
        return {self.__feature_name: self.__word2id.get(obj_text, OOV)}


class EmbeddingInitializer(builder.ConstantFeatureInitializer):
    def __init__(self, initializer_name: str, word2id: Dict[str, int], word2vec: Dict[str, List[float]],
                 embed_size: int, vocab_size: int):
        builder.ConstantFeatureInitializer.__init__(self, initializer_name, [vocab_size, embed_size])
        self.__initializer_name = initializer_name
        self.__word2id = word2id
        self.__word2vec = word2vec
        self.__embed_size = embed_size

    def initial_values(self) -> Dict[str, Iterator[Tuple[int, List[float]]]]:
        values = chain(
            [(builder.PADDING, [0] * self.__embed_size)],
            ((self.__word2id.get(word), vec) for word, vec in self.__word2vec.items()))
        return {self.__initializer_name: values}


class TrainableEmbeddingsExtractorBuilder(builder.FeatureExtractorBuilder[NavigableToken, List[int]]):
    def __init__(self, feature_name: str, threshold: int, lowercase: bool):
        self.__feature_name = feature_name
        self.__threshold = threshold
        self.__lowercase = lowercase

    def build(self, dataset: Iterable[Document]):
        tokens = chain.from_iterable(_tokens(dataset, lambda token: token.text))
        counter = Counter(tokens)
        filtered = map(lambda item: item[0], filter(lambda item: item[1] >= self.__threshold, counter.most_common()))
        word2id = {wr: index for index, wr in enumerate(sorted(filtered), 2)}

        return WordEmbeddingFeatureExtractor(self.__feature_name, word2id, self.__lowercase), \
               builder.DoNothingInterpreter(), \
               builder.ConstantFeatureInitializer(self.__feature_name, [len(word2id) + 2])


############################# Subword Embedding #############################


class SubwordEmbeddingFeatureExtractorBuilder(builder.FeatureExtractorBuilder[NavigableToken, int]):

    def __init__(self, feature_name: str, initializer_name: str, config):
        self.__feature_name = feature_name
        self.__initializer_name = initializer_name
        self.__path = config["embedding_path"]
        self.__segmenter = config['segmenter']
        self.__segmenter_config = config.get('segmenter_config', {})

    def build(self, dataset: Iterable[Document]):
        if self.__segmenter not in _segmenters:
            raise ValueError("Invalid segmentator name specified {}. (Available splitters: {})"
                             .format(self.__segmenter, _segmenters.keys()))
        segmenter = _segmenters[self.__segmenter](**self.__segmenter_config)
        subword2vec, embed_size = _read_embeddings(self.__path)
        subword2id = {subword: index for index, subword in enumerate(subword2vec, 4)}

        return SubwordFeatureExtractor(self.__feature_name, subword2id, segmenter), \
               builder.DoNothingInterpreter(), \
               EmbeddingInitializer(self.__initializer_name, subword2id, subword2vec, embed_size, len(subword2id) + 4)


############################# Characters #############################


class TokenCharactersFeatureExtractorBuilder(builder.FeatureExtractorBuilder[NavigableToken, List[int]]):
    """
    Token-level feature extractor builder for character-based word embeddings.

    It constructs character vocabulary (enumeration) for specified dataset.

    Feature is list of character indices with constant padding
    """

    def __init__(self, feature_name: str, config):
        self.__feature_name = feature_name
        self.__threshold = config['min_frequency']
        self.__segmenter = config['segmenter']
        self.__segmenter_config = config.get('segmenter_config', {})

    def build(self, dataset: Iterable[Document]):
        if self.__segmenter not in _segmenters:
            raise ValueError("Invalid segmentator name specified {}. (Available splitters: {})"
                             .format(self.__segmenter, _segmenters.keys()))
        segmenter = _segmenters[self.__segmenter](**self.__segmenter_config)
        characters = chain.from_iterable(_tokens(dataset, lambda token: segmenter.segment(token.text)))
        counter = Counter(characters)
        filtered = map(lambda item: item[0], filter(lambda item: item[1] >= self.__threshold, counter.most_common()))
        char2id = {ch: index for index, ch in enumerate(sorted(filtered), 4)}
        return SubwordFeatureExtractor(self.__feature_name, char2id, segmenter), builder.DoNothingInterpreter(), \
               builder.ConstantFeatureInitializer(self.__feature_name, [len(char2id) + 4])


class SubwordFeatureExtractor(builder.FeatureExtractorInterface[NavigableToken, List[int]]):

    def __init__(self, feature_name: str, subword2id: Dict[str, int], segmenter):
        self.__feature_name = feature_name
        self.__subword2id = subword2id
        self.__segmenter = segmenter

    def get_features(self, obj: NavigableToken, **kwargs) -> Dict[str, List[int]]:
        ids = [START]
        ids.extend([self.__subword2id.get(ch, OOV) for ch in self.__segmenter.segment(obj.text)])
        ids.append(END)
        return {self.__feature_name: ids}


class TokenSubwordSegmenter:

    def segment(self, token: str):
        pass


class TokenCharSegmenter(TokenSubwordSegmenter):

    def __init__(self, *args, **kwargs):
        pass

    def segment(self, token: str):
        return [ch for ch in token]


class TokenBPESegmenter(TokenSubwordSegmenter):

    def __init__(self, language, vocab_size, *args, **kwargs):
        self.__bpe = BPEmb(lang=language, vs=vocab_size)

    def segment(self, token: str):
        subwords = [ch for ch in self.__bpe.encode(token)]
        if subwords[0].startswith('\u2581'):
            if len(subwords[0]) > 1:
                subwords[0] = subwords[0][1:]
            else:
                subwords.pop(0)
        return subwords


class TokenMorfessorSegmenter(TokenSubwordSegmenter):

    def __init__(self, path):
        self.io = mf.MorfessorIO()
        self.model = self.io.read_any_model(path)

    def segment(self, token: str):
        return self.model.viterbi_segment(token)[0]


_segmenters = {
    'char': TokenCharSegmenter,
    'bpe': TokenBPESegmenter,
    'morfessor': TokenMorfessorSegmenter
}

########################### Part of Speech ###########################

_L = TypeVar('L')


class Splitter(Generic[_L]):
    def __init__(self, counts: Dict[_L, int]):
        pass

    def split(self, pos: _L) -> Tuple[Any, ...]:
        pass

    def combine(self, parts: Tuple[Any, ...]) -> _L:
        pass


class PosNoSplitter(Splitter[Pos]):
    """
    Pos splitter that doesn't split Pos
    """

    def split(self, pos: Pos):
        return OrderedDict([("POS", pos)])

    def combine(self, parts: Tuple[Pos]) -> Pos:
        return parts[0]


class PosUXFSplitter(Splitter[Pos]):
    """
    Pos splitter that splits Pos to three parts: (UPOS, XPOS, FEATS)
    """

    def split(self, pos: Pos):
        return OrderedDict([("UPOS", pos.upos), ("XPOS", pos.xpos), ("FEATS", _dict2repr(pos.feats))])

    def combine(self, parts: Tuple[Any, ...]) -> Pos:
        return Pos(parts[0], parts[1], dict(parts[2]))


class PosXFSplitter(Splitter[Pos]):
    """
    Pos splitter that splits Pos to two parts: (UPOS, XPOS+FEATS)
    """

    def split(self, pos: Pos):
        return OrderedDict([("UPOS", pos.upos), ("XPOS+FEATS", (pos.xpos, _dict2repr(pos.feats)))])

    def combine(self, parts: Tuple[Any, ...]) -> Pos:
        return Pos(parts[0], parts[1][0], dict(parts[1][1]))


class PosGranularSplitter(Splitter):
    """
    Pos splitter that splits Pos to many (more than two) parts: (UPOS, XPOS, feat1, feat2, ...).
    The number of Pos parts depends on train set (`poses` dict). All feats parts are sorted by feat class name.
    """
    def __init__(self, poses: Dict[Pos, int]):
        Splitter.__init__(self, poses)
        self.__feats_keys = sorted({feat_type for pos in poses.keys() for feat_type in pos.feats.keys()})  # type: List[str]

    def split(self, pos: Pos):
        feats = pos.feats
        splitted = OrderedDict([(key, feats.get(key, None)) for key in self.__feats_keys])
        result = OrderedDict([("UPOS", pos.upos), ("XPOS", pos.xpos)])
        result.update(splitted)
        return result

    def combine(self, parts: Tuple[Any, ...]) -> Pos:
        feats = {self.__feats_keys[i-2]: parts[i] for i in range(2, len(parts)) if parts[i] is not None}
        return Pos(parts[0], parts[1], feats)


def _dict2repr(d: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    items = sorted(d.items())  # type: List[Tuple[str, str]]
    return tuple(items)


_splitters = {
    'pos_joint': PosNoSplitter,
    'pos_XF': PosXFSplitter,
    'pos_UXF': PosUXFSplitter,
    'pos_gran': PosGranularSplitter
}


class PosDictFeatureExtractorBuilder(builder.FeatureExtractorBuilder[Token, int]):
    """
    Token-level feature (label) extractor.

    Label extractor for Pos.
    It splits Pos according to specified strategy, enumerates all Pos' parts in dataset and returns dict feature extractor with dict label interpreter.
    """
    def __init__(self, label_name: str, splitting_name: str):
        self.__label_name = label_name
        if splitting_name not in _splitters:
            raise ValueError("Invalid splitter name specified {}. (Available splitters: {})".format(splitting_name,
                                                                                                    _splitters.keys()))
        self.__splitting_name = splitting_name

    def build(self, dataset: Iterable[Document]):
        label_counts = _count_labels(dataset, lambda token: token.pos)
        label_splitter = _splitters[self.__splitting_name](label_counts)
        parts_counts = _count_parts(label_counts, label_splitter)
        id2parts = [_id2obj(parts_counts[part_counts]) for part_counts in parts_counts]
        parts2id = [_obj2id(id2part) for id2part in id2parts]
        lens = OrderedDict([(part_counts, len(id2part)) for part_counts, id2part in zip(parts_counts, id2parts)])

        return PosDictFeatureExtractor(self.__label_name, label_splitter, parts2id), \
               PosDictInterpreter(self.__label_name, label_splitter, id2parts), \
               builder.ConstantFeatureInitializer(self.__label_name, lens)


class PosDictFeatureExtractor(builder.FeatureExtractorInterface[NavigableToken, List[int]]):
    def __init__(self, label_name: str, splitter: Splitter[Pos], parts2id):
        self.__label_name = label_name
        self.__splitter = splitter
        self.__parts2id = parts2id

    def get_features(self, obj: Token, **kwargs) -> Dict[str, List[int]]:
        split = self.__splitter.split(obj.pos)
        ids = [self.__parts2id[i].get(split[part], 0) for i, part in enumerate(split)]
        return {self.__label_name: ids}


class PosDictInterpreter(builder.InterpreterInterface[DocumentToken, List[int]]):
    def __init__(self, label_name: str, splitter: Splitter[Pos], id2parts):
        self.__label_name = label_name
        self.__splitter = splitter
        self.__id2parts = id2parts

    def interpret(self, obj: DocumentToken, value: Dict[str, List[int]]) -> DocumentToken:
        ids = value[self.__label_name]
        parts = tuple(self.__id2parts[i][id] for i, id in enumerate(ids))
        return obj.with_morphology(obj, pos=self.__splitter.combine(parts), lemma=obj.lemma)


############################# Lemma Dict #############################


class LemmaDictFeatureExtractorBuilder(builder.FeatureExtractorBuilder[Token, int]):
    """
    Token-level feature (label) extractor.

    Label extractor for Lemma.
    It enumerates all lemmas in dataset and returns dict feature extractor with dict label interpreter.
    """

    def __init__(self, label_name: str):
        self.__label_name = label_name

    def build(self, dataset: Iterable[Document]):
        lemma_count = _count_labels(dataset, lambda token: token.lemma)
        id2lemma = _id2obj(lemma_count)
        lemma2id = _obj2id(id2lemma)
        return LemmaDictFeatureExtractor(self.__label_name, lemma2id), \
               LemmaDictInterpreter(self.__label_name, id2lemma), \
               builder.ConstantFeatureInitializer(self.__label_name, {"LEMMA": len(id2lemma)})


class LemmaDictFeatureExtractor(builder.FeatureExtractorInterface[Token, List[int]]):
    def __init__(self, label_name: str, lemma2id: Dict[str, int]):
        self.__label_name = label_name
        self.__lemma2id = lemma2id

    def get_features(self, obj: Token, **kwargs) -> Dict[str, List[int]]:
        return {self.__label_name: [self.__lemma2id.get(obj.lemma, OOV)]}


class LemmaDictInterpreter(builder.InterpreterInterface[DocumentToken, List[int]]):
    def __init__(self, label_name: str, id2lemma: List[str]):
        self.__label_name = label_name
        self.__id2lemma = id2lemma

    def interpret(self, obj: DocumentToken, value: Dict[str, List[int]]) -> DocumentToken:
        return obj.with_morphology(obj, pos=obj.pos, lemma=self.__id2lemma[value[self.__label_name][0]])


##################### Lemma Suffix Transformation ####################


class CachedSuffixRulesBuilder(object):
    """
    lemma suffix transformation trie builder with cache (to avoid double computation)
    """

    def __init__(self, threshold):
        self.__threshold = threshold
        self.__cache = {}

    def build_rules(self, dataset: Iterable[Document]):
        if dataset not in self.__cache:
            suffix_count = _count_labels(dataset, _suffix_transformation)
            id2suffix = _id2obj(suffix_count, self.__threshold)
            suffix2id = _obj2id(id2suffix)
            self.__cache[dataset] = (suffix2id, id2suffix)
        return self.__cache[dataset]


class LemmaSuffixTransformationFeatureExtractorBuilder(builder.FeatureExtractorBuilder[NavigableToken, int]):
    """
    Token-level feature (label) extractor.

    Label extractor for Lemma.
    It builds lemma transformation rules trie with specified builder and builds feature extractor that
    compute transformation rule for current token (with specified lemma) and returns its index.

    Interpreter could apply transformation rule by index to Token
    """

    def __init__(self, label_name: str, rules_builder: CachedSuffixRulesBuilder):
        self.__label_name = label_name
        self.__rules_builder = rules_builder

    def build(self, dataset: Iterable[Document]):
        rule2id, id2rule = self.__rules_builder.build_rules(dataset)
        return SuffixTransformationFeatureExtractor(self.__label_name, rule2id), \
               SuffixTransformationInterpreter(self.__label_name, id2rule), \
               builder.ConstantFeatureInitializer(self.__label_name, {"LEMMA": len(id2rule)})


class SuffixTransformationFeatureExtractor(builder.FeatureExtractorInterface[NavigableToken, List[int]]):
    def __init__(self, label_name: str, rule2id: Dict[LemmaSuffixTransformation, int]):
        self.__label_name = label_name
        self.__rule2id = rule2id

    def get_features(self, obj: NavigableToken, **kwargs) -> Dict[str, List[int]]:
        rule = _suffix_transformation(obj)
        return {self.__label_name: [self.__rule2id.get(rule, 0)]}


class SuffixTransformationInterpreter(builder.InterpreterInterface[DocumentToken, List[int]]):
    def __init__(self, label_name: str, id2rule: List[LemmaSuffixTransformation]):
        self.__label_name = label_name
        self.__id2rule = id2rule

    def interpret(self, obj: DocumentToken, value: Dict[str, List[int]]) -> DocumentToken:
        rule = self.__id2rule[value[self.__label_name][0]]
        if rule.is_applicable(obj.text.lower()):
            return obj.with_morphology(obj, pos=obj.pos, lemma=rule.transform(obj.text.lower()))
        print("warn: inapplicable rule '{}' for token '{}'".format(rule, obj.text))
        return obj.with_morphology(obj, pos=obj.pos, lemma=obj.text.lower())  # do nothing if rule is not applicable


class ApplicableSuffixTransformationsBuilder(builder.FeatureExtractorBuilder[NavigableToken, List[int]]):
    """
    Token-level feature extractor.

    Builds feature extractor that provide some-hot binary vector for current token.
    This vector represents which lemma transformation rules are applicable for given token.
    """

    def __init__(self, feature_name: str, rules_builder: CachedSuffixRulesBuilder):
        self.__feature_name = feature_name
        self.__rules_builder = rules_builder

    def build(self, dataset: Iterable[Document]):
        rule2id, id2rule = self.__rules_builder.build_rules(dataset)
        suffix_trie = LemmaTransformationTrie.build(filter(lambda t: t[0] is not None, rule2id.items()))
        return ApplicableSuffixTransformations(self.__feature_name, suffix_trie, len(id2rule)), \
               builder.DoNothingInterpreter(), \
               builder.ConstantFeatureInitializer(self.__feature_name, [len(id2rule)])


class ApplicableSuffixTransformations(builder.FeatureExtractorInterface[NavigableToken, List[int]]):
    def __init__(self, feature_name: str, suffix_trie: LemmaTransformationTrie, size: int):
        self.__feature_name = feature_name
        self.__suffix_trie = suffix_trie
        self.__size = size

    def get_features(self, obj: NavigableToken, **kwargs) -> Dict[str, List[int]]:
        applicable = self.__suffix_trie.get_applicable_rules(obj.text.lower())
        result = [False] * self.__size
        for _, index in applicable:
            result[index] = True
        return {self.__feature_name: result}


class LemmaCharFeatureExtractorBuilder(builder.FeatureExtractorBuilder[NavigableToken, List[int]]):
    def __init__(self, feature_name: str, config):
        self.__feature_name = feature_name
        self.__threshold = config['min_frequency']
        self.__max_word_length = config['max_word_length']
        self.__segmenter = config['segmenter']
        self.__segmenter_config = config.get('segmenter_config', {})

    def build(self, dataset: Iterable[Document]):
        if self.__segmenter not in _segmenters:
            raise ValueError("Invalid segmentator name specified {}. (Available splitters: {})"
                             .format(self.__segmenter, _segmenters.keys()))
        segmenter = _segmenters[self.__segmenter](**self.__segmenter_config)
        lemma_chars = chain.from_iterable(_tokens(dataset, lambda token: segmenter.segment(token.lemma)))
        counter = Counter(lemma_chars)
        filtered = map(lambda item: item[0], filter(lambda item: item[1] >= self.__threshold, counter.most_common()))
        char2id = {ch: index for index, ch in enumerate(sorted(filtered), 4)}
        id2char = {index: ch for ch, index in char2id.items()}
        return LemmaCharExtractor(self.__feature_name, char2id, self.__max_word_length, segmenter),\
               LemmaCharInterpreter(self.__feature_name, id2char), \
               builder.ConstantFeatureInitializer(self.__feature_name, [len(char2id) + 4])


class LemmaCharExtractor(builder.FeatureExtractorInterface[NavigableToken, List[int]]):
    def __init__(self, label_name: str, char2id: Dict[str, int], max_word_length: int, segmenter):
        self.__label_name = label_name
        self.__char2id = char2id
        self.__max_word_length = max_word_length
        self.__segmenter = segmenter

    def get_features(self, obj: NavigableToken, **kwargs) -> Dict[str, List[int]]:
        ids = [START]
        ids.extend([self.__char2id.get(c, OOV) for c in self.__segmenter.segment(obj.lemma)])
        ids.append(END)
        if len(ids) > self.__max_word_length:
            ids = ids[:self.__max_word_length]
        else:
            ids.extend([PADDING] * (self.__max_word_length - len(ids)))
        return {self.__label_name: ids}


class LemmaCharInterpreter(builder.InterpreterInterface[DocumentToken, List[int]]):
    def __init__(self, label_name: str, id2char: Dict[int, str]):
        self.__label_name = label_name
        self.__id2char = id2char

    def interpret(self, obj: DocumentToken, value: Dict[str, List[int]]) -> DocumentToken:
        lemma = ''
        for id_ in value[self.__label_name]:
            if id_ == END:
                break
            elif id_ not in [START, OOV, PADDING]:
                lemma += self.__id2char[id_]
        return obj.with_morphology(obj, pos=obj.pos, lemma=lemma)


################################ Utils ###############################


__T = TypeVar('T')


def _tokens(dataset: Iterable[Document], mapper: Callable[[NavigableToken], __T]) -> Iterator[__T]:
    return map(mapper, chain.from_iterable(dataset))


def _count_labels(dataset: Iterable[Document], label_extractor: Callable[[NavigableToken], __T]) -> Dict[__T, int]:
    return Counter(_tokens(dataset, label_extractor))


def _count_parts(labels_counts: Dict[__T, int], label_splitter: Splitter[__T]):
    parts = label_splitter.split(next(iter(labels_counts.keys())))
    result = OrderedDict([(part, defaultdict(int)) for part in parts])
    for label, count in labels_counts.items():
        split = label_splitter.split(label)
        for part in split:
            result[part][split[part]] += count
    return result


def _id2obj(labels_counts: Dict[__T, int], threshold=0) -> List[__T]:
    if not isinstance(labels_counts, Counter):
        labels_counts = Counter(labels_counts)
    most_common = labels_counts.most_common(1)[0][0]
    objects = labels_counts.items()
    if threshold > 0:
        objects = filter(lambda item: item[1] > threshold, objects)
    objects = map(lambda item: item[0], objects)
    objects = filter(lambda obj: obj != most_common, objects)
    id2obj = sorted(objects, key=lambda x: (x is not None, x))
    id2obj.insert(0, most_common)
    return id2obj


def _obj2id(id2obj: List[__T]) -> Dict[__T, int]:
    return {obj: i for i, obj in enumerate(id2obj)}


def _suffix_transformation(token: NavigableToken) -> LemmaSuffixTransformation:
    return LemmaSuffixTransformation.get_rule(token.text.lower(), token.lemma.lower())
