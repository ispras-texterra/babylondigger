from typing import TypeVar, Generic, Dict, List, Iterable, Iterator, Any, Tuple

from babylondigger.datamodel import Document

from itertools import chain
from collections import defaultdict

import random

_T = TypeVar('T')
_V = TypeVar('V')


############################## Constants #############################


PADDING = 0
OOV = 1
START = 2
END = 3

############################# Interfaces #############################


class FeatureExtractorInterface(Generic[_T, _V]):
    """
    Interface for features (labels) extractor.

    Each feature extractor could extract more then one feature
    """

    def get_features(self, obj: _T, **kwargs) -> Dict[str, _V]:
        """
        extract features from specified object

        :param obj: object to be processed
        :return: dict from feature name to feature value
        """
        pass


class InterpreterInterface(Generic[_T, _V]):
    """
    Interface for classified labels interpretation
    """

    def interpret(self, obj: _T, value: Dict[str, _V]) -> _T:
        """
        Interpret classification result for given object (Document, Sentence, Token).
        :param obj: classified object
        :param value: classification result
        :return: object with classification result interpretation (e.g. Token with Pos)
        """
        pass


class FeatureInitializerInterface(object):
    """
    Interface for neural network specific initialization.

    Most of features have predefined shape (Tensor dimensionality).
    Moreover some features have predefined initial values for Variables in neural network (e.g. word vectors).
    """

    def shapes(self) -> Dict[str, List[int]]:
        """
        get features shapes (Tensor dimensionality)
        :return: feature shape for features (dict feature name to shape)
        """
        pass

    def initial_values(self) -> Dict[str, Any]:
        """
        get initial values for features
        :return: dict feature name to its initial values (implementation-dependent format)
        """
        pass


class FeatureExtractorBuilder(Generic[_T, _V]):
    """
    Interface for feature extractor builder.
    """

    def build(self, dataset: Iterable[Document]) -> \
            Tuple[FeatureExtractorInterface[_T, _V], InterpreterInterface[_T, _V], FeatureInitializerInterface]:
        """
        build feature extractor for specified dataset

        :param dataset: train set (labeled documents)
        :return: tuple of feature extractor, label interpreter and feature initializer
        """
        pass

    @classmethod
    def merge(cls, builders: List['FeatureExtractorBuilder[_T, _V]']) -> 'FeatureExtractorBuilder[_T, _V]':
        """
        merge uniform feature extractor builders to single feature extractor builder

        This method should be implemented only for wrappers

        :param builders: uniform feature extractor builders
        :return: single builder of the same type
        """
        raise NotImplementedError()


############################### Padding ##############################


class PaddingFeatureExtractorBuilder(FeatureExtractorBuilder[_T, _V]):
    """
    Feature extractor builder wrapper that pad all sequential features
    """

    def __init__(self, builder: FeatureExtractorBuilder[_T, _V]):
        self.__builder = builder

    def build(self, dataset: Iterable[Document]):
        extractor, interpreter, initializer = self.__builder.build(dataset)
        return PaddingFeatureExtractor(extractor), interpreter, initializer

    @classmethod
    def merge(cls, builders: List['PaddingFeatureExtractorBuilder']) -> 'PaddingFeatureExtractorBuilder':
        internal_builders = [builder.__builder for builder in builders]
        builder = merge(internal_builders)
        return PaddingFeatureExtractorBuilder(builder)


class PaddingFeatureExtractor(FeatureExtractorInterface[_T, _V]):
    def __init__(self, extractor: FeatureExtractorInterface[_T, _V]):
        self.__extractor = extractor

    def get_features(self, obj: _T, **kwargs) -> Dict[str, _V]:
        features = self.__extractor.get_features(obj, **kwargs)
        for value in features.values():
            _pad_sequence(value)
        return features


def _pad_sequence(sequence: Iterable[Any]):

    class FlatMapIterable(object):
        def __iter__(self):
            return chain.from_iterable(sequence)

    if all(map(lambda x: not isinstance(x, list), sequence)):
        return PADDING
    if not all(map(lambda x: isinstance(x, list), sequence)):
        raise ValueError("Inhomogeneous feature")
    max_len = max(map(lambda x: len(x), sequence))
    padding = _pad_sequence(FlatMapIterable())
    for element in sequence:
        element.extend([padding] * (max_len - len(element)))
    return [padding] * max_len


class SparseFeatureExtractorBuilder(FeatureExtractorBuilder[_T, _V]):

    def __init__(self, builder: FeatureExtractorBuilder[_T, _V]):
        self.__builder = builder

    def build(self, dataset: Iterable[Document]):
        extractor, interpreter, initializer = self.__builder.build(dataset)
        return SparseFeatureExtractor(extractor), interpreter, initializer

    @classmethod
    def merge(cls, builders: List['SparseFeatureExtractorBuilder']) -> 'SparseFeatureExtractorBuilder':
        internal_builders = [builder.__builder for builder in builders]
        builder = merge(internal_builders)
        return SparseFeatureExtractorBuilder(builder)


class SparseFeatureExtractor(FeatureExtractorInterface[_T, _V]):
    def __init__(self, extractor: FeatureExtractorInterface[_T, _V]):
        self.__extractor = extractor

    def get_features(self, obj: _T, sparse_rate=0) -> Dict[str, _V]:
        features = self.__extractor.get_features(obj)

        for feature in features.keys():
            features[feature] = _sparse_rate_sequence(features[feature], sparse_rate)
        return features


def _sparse_rate_sequence(input_, sparse_rate):
    if isinstance(input_, int):
        return input_ if random.random() > sparse_rate else OOV
    elif isinstance(input_, list):
        return [_sparse_rate_sequence(el, sparse_rate) for el in input_]
    else:
        raise NotImplementedError('Sparsification of feature of this type is not supported: {}'.format(type(input_)))


#################### Composite Feature Extractors ####################


class CompositeFeatureExtractorBuilder(FeatureExtractorBuilder[_T, _V]):
    """
    Feature extractor builder wrapper that builds feature extractor for composition of specified builders
    """

    def __init__(self, builders: List[FeatureExtractorBuilder[_T, _V]]):
        self.__builders = builders

    def build(self, dataset: Iterable[Document]):
        if len(self.__builders) == 1:
            return self.__builders[0].build(dataset)
        extractors, interpreters, initializers = zip(*map(lambda builder: builder.build(dataset), self.__builders))
        return CompositeFeatureExtractor(extractors), _composite_interpreter(interpreters), CompositeInitializer(initializers)

    @classmethod
    def merge(cls, builders: List['CompositeFeatureExtractorBuilder']) -> FeatureExtractorBuilder:
        internal_builders = [internal for builder in builders for internal in builder.__builders]
        return merge(internal_builders)


def _composite_interpreter(interpreters: Iterable[InterpreterInterface[_T, _V]]) -> InterpreterInterface[_T, _V]:
    interpreters = list(filter(lambda interpreter: not isinstance(interpreter, DoNothingInterpreter), interpreters))
    if len(interpreters) == 0:
        return DoNothingInterpreter()
    if len(interpreters) == 1:
        return interpreters[0]
    return CompositeInterpreter(interpreters)


class CompositeFeatureExtractor(FeatureExtractorInterface[_T, _V]):
    def __init__(self, extractors: Iterable[FeatureExtractorInterface[_T, _V]]):
        self.__extractors = extractors

    def get_features(self, obj: _T, **kwargs) -> Dict[str, _V]:
        return _merge_dicts(map(lambda x: x.get_features(obj, **kwargs), self.__extractors))


class CompositeInterpreter(InterpreterInterface[_T, _V]):
    def __init__(self, interpreters: Iterable[InterpreterInterface[_T, _V]]):
        self.__interpreters = interpreters

    def interpret(self, obj: _T, value: Dict[str, _V]) -> _T:
        result = obj
        for extractor in self.__interpreters:
            result = extractor.interpret(result, value)
        return result


class CompositeInitializer(FeatureInitializerInterface):
    def __init__(self, initializers: Iterable[FeatureInitializerInterface]):
        self.__initializers = initializers

    def shapes(self) -> Dict[str, List[int]]:
        return _merge_dicts(map(lambda x: x.shapes(), self.__initializers))

    def initial_values(self) -> Dict[str, Any]:
        return _merge_dicts(map(lambda x: x.initial_values(), self.__initializers))


def _merge_dicts(dicts: Iterator[Dict[str, _V]]) -> Dict[str, _V]:
    result = {}
    for d in dicts:
        if any(key in result for key in d):
            raise ValueError('Duplicate keys')
        result.update(d)
    return result


##################### Iterable Feature Extractor #####################


class IterableFeatureExtractorBuilder(Generic[_T, _V], FeatureExtractorBuilder[Iterable[_T], Iterable[_V]]):
    """
    Feature extractor builder wrapper to construct feature extractor for sequence of objects

    It applies feature extractor (which is constructed from specified builder) for each object in sequence and
    merges feature values.
    """

    def __init__(self, builder: FeatureExtractorBuilder[_T, _V]):
        self.__builder = builder

    def build(self, dataset: Iterable[Document]):
        extractor, interpreter, initializer = self.__builder.build(dataset)
        return IterableFeatureExtractor(extractor), IterableInterpreter(interpreter), initializer

    @classmethod
    def merge(cls, builders: List['IterableFeatureExtractorBuilder']) -> 'IterableFeatureExtractorBuilder':
        internal_builders = [builder.__builder for builder in builders]
        builder = merge(internal_builders)
        return IterableFeatureExtractorBuilder(builder)


class IterableFeatureExtractor(Generic[_T, _V], FeatureExtractorInterface[Iterable[_T], Iterable[_V]]):
    def __init__(self, extractor: FeatureExtractorInterface[_T, _V]):
        self.__extractor = extractor

    def get_features(self, obj: Iterable[_T], **kwargs) -> Dict[str, Iterable[_V]]:
        def __append(value: Dict[str, _V]):
            for key, value in value.items():
                if key not in result:
                    result[key] = []
                result[key].append(value)
        result = {}
        for element in obj:
            features = self.__extractor.get_features(element, **kwargs)
            __append(features)
        return result


class IterableInterpreter(Generic[_T, _V], InterpreterInterface[Iterable[_T], Iterable[_V]]):
    def __init__(self, interpreter: InterpreterInterface[_T, _V]):
        self.__interpreter = interpreter

    def interpret(self, obj: Iterable[_T], value: Dict[str, Iterable[_V]]) -> Iterable[_T]:
        # Dict[str, Iterable[_V]] -> Iterable[Dict[str, _V]]
        converted_value = (dict(zip(value, t)) for t in zip(*value.values()))
        return [self.__interpreter.interpret(el, el_val) for el, el_val in zip(obj, converted_value)]


########################### Implementations ##########################


class EmptyFeatureExtractor(FeatureExtractorInterface):
    """
    Feature extractor that returns no features
    """

    def get_features(self, obj: _T, **kwargs) -> Dict[str, _V]:
        return {}


class EmptyFeatureInitializer(FeatureInitializerInterface):
    """
    Feature initializer for features which don't need any initialization
    """

    def shapes(self) -> Dict[str, List[int]]:
        return {}

    def initial_values(self) -> Dict[str, Any]:
        return {}


class ConstantFeatureInitializer(FeatureInitializerInterface):
    """
    Feature initializer that returns constant predefined value for specified feature name
    """

    def __init__(self, name: str, shape):
        self.__shape = {name: shape}

    def shapes(self) -> Dict[str, List[int]]:
        return self.__shape

    def initial_values(self) -> Dict[str, Any]:
        return {}


class DoNothingInterpreter(InterpreterInterface):
    """
    Labels interpreter that returns object as is (without any interpretation of labels).
    """

    def interpret(self, obj, value):
        return obj


class EmptyFeatureExtractorBuilder(FeatureExtractorBuilder):
    """
    Builder for empty feature extractor.

    This extractor should be used when no features are needed for method.
    """

    def build(self, dataset: Iterable[Document]):
        return EmptyFeatureExtractor(), DoNothingInterpreter(), EmptyFeatureInitializer()

    @classmethod
    def merge(cls, builders: List['EmptyFeatureExtractorBuilder']) -> 'EmptyFeatureExtractorBuilder':
        return builders[0]


################################ Utils ###############################

def merge(builders: Iterator[FeatureExtractorBuilder]) -> FeatureExtractorBuilder:
    """
    merge feature extractor builders to obtain single builder (all wrappers are merged if possible)

    :param builders:
    :return:
    """
    type2builder = defaultdict(list)
    for builder in builders:
        type2builder[type(builder)].append(builder)
    if len(type2builder) == 0:
        raise ValueError()
    skip_empty_builder = len(type2builder) > 1
    result = []
    for t, values in type2builder.items():
        if skip_empty_builder and t == EmptyFeatureExtractorBuilder:
            continue
        if len(values) == 1:
            result.append(values[0])
            continue
        try:
            result.append(t.merge(values))
        except NotImplementedError:
            result.extend(values)
    if len(result) == 1:
        return result[0]
    return CompositeFeatureExtractorBuilder(result)


def wrap_sentence_level(sentence_level_extractor_builder):
    """
    helper method to create batch-level feature extractor builder from sentence-level feature extractor

    :param sentence_level_extractor_builder:
    :return:
    """
    return PaddingFeatureExtractorBuilder(
        IterableFeatureExtractorBuilder(
            sentence_level_extractor_builder))


def wrap_word_level(word_level_extractor_builder, sparsify=False):
    """
    helper method to create batch-level feature extractor builder from token-level feature extractor
    :param word_level_extractor_builder:
    :return:
    """
    if sparsify:
        return wrap_sentence_level(
            IterableFeatureExtractorBuilder(SparseFeatureExtractorBuilder(word_level_extractor_builder)))
    return wrap_sentence_level(IterableFeatureExtractorBuilder(word_level_extractor_builder))
