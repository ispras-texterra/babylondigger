from typing import Tuple, Iterable

from babylondigger.datamodel import Document
import babylondigger.features.builder as builder
import babylondigger.features.extractors as extractors


class FeatureExtractorBuilderProvider(object):
    """
    Provider for feature extractor and label extractor assembly.

    It merges provided builders for feature extraction and label extraction and builds extractors
    """

    def __init__(self, builder_providers: Iterable['AbstractExtractorBuilderProvider']):
        self._feature_extractor_builder = builder.merge(map(lambda x: x.feature_extractor_builder(), builder_providers))
        self._labels_extractor_builder = builder.merge(map(lambda x: x.labels_extractor_builder(), builder_providers))

    def build(self, dataset: Iterable[Document]):
        """
        build feature and label extractors provided by specified providers
        :param dataset: train set for feature and label extractor building
        :return: tuple of feature extractor (used for both training and testing),
        label extractor (used for training), label interpreter (used for testing) and
        feature initializer (used for neural network building)
        """

        feature_extractor, _, feature_initializer = self._feature_extractor_builder.build(dataset)
        label_extractor, label_interpreter, label_initializer = self._labels_extractor_builder.build(dataset)
        return feature_extractor, label_extractor, label_interpreter, \
               builder.CompositeInitializer([feature_initializer, label_initializer])


class AbstractExtractorBuilderProvider(object):
    """
    Abstract feature and label extractor builder provider

    This class is created for consistent feature and label extractor builders configuration.
    """

    def feature_extractor_builder(self):
        return builder.EmptyFeatureExtractorBuilder()

    def labels_extractor_builder(self):
        return builder.EmptyFeatureExtractorBuilder()

############################## Features ##############################


class SentenceLengthExtractorProvider(AbstractExtractorBuilderProvider):
    """
    Provider for sentence length feature
    """

    def __init__(self, feature_name: str):
        self.__feature_name = feature_name

    def feature_extractor_builder(self):
        return builder.wrap_sentence_level(extractors.SentenceLengthFeatureExtractorBuilder(self.__feature_name))


class WordEmbeddingExtractorProvider(AbstractExtractorBuilderProvider):
    """
    Provider for word embedding feature
    """

    def __init__(self, feature_name: str, initializer_name: str, config):
        self.__feature_name = feature_name
        self.__initializer_name = initializer_name

        self.__path = config['path']
        self.__lowercase = config['lowercase']

    def feature_extractor_builder(self):
        return builder.wrap_word_level(extractors.WordEmbeddingFeatureExtractorBuilder(self.__feature_name,
                                                                                       self.__initializer_name,
                                                                                       self.__path, self.__lowercase), True)


class TrainableWordEmbeddingExtractorProvider(AbstractExtractorBuilderProvider):
    """
    Provider for word embedding feature
    """
    def __init__(self, feature_name: str, config):
        self.__feature_name = feature_name
        self.__threshold = config['min_frequency']
        self.__lowercase = config["lowercase"]

    def feature_extractor_builder(self):
        return builder.wrap_word_level(extractors.TrainableEmbeddingsExtractorBuilder(self.__feature_name,
                                                                                      self.__threshold,
                                                                                      self.__lowercase), True)


class SubwordEmbeddingExtractorProvider(AbstractExtractorBuilderProvider):
    """
    Provider for subword embedding feature
    """

    def __init__(self, feature_name: str, initializer_name: str, config):
        self.__feature_name = feature_name
        self.__initializer_name = initializer_name

        self.__config = config

    def feature_extractor_builder(self):
        return builder.wrap_word_level(extractors.SubwordEmbeddingFeatureExtractorBuilder(self.__feature_name,
                                                                                       self.__initializer_name,
                                                                                       self.__config), True)


class SubwordsExtractorProvider(AbstractExtractorBuilderProvider):
    """
    Provider for word characters feature
    """
    def __init__(self, feature_name: str, config):
        self.__feature_name = feature_name
        self.__config = config

    def feature_extractor_builder(self):
        return builder.wrap_word_level(extractors.TokenCharactersFeatureExtractorBuilder(self.__feature_name, self.__config), True)


############################### Labels ###############################


class LemmaDictExtractorProvider(AbstractExtractorBuilderProvider):
    """
    Provider for lemma id (vocabulary) label representation
    """

    def __init__(self, label_name: str):
        self.__label_name = label_name

    def labels_extractor_builder(self):
        return builder.wrap_word_level(extractors.LemmaDictFeatureExtractorBuilder(self.__label_name))


class PosDictExtractorProvider(AbstractExtractorBuilderProvider):
    """
    Provider for Pos id (vocabulary) label representation
    """

    def __init__(self, label_name: str, splitting: str):
        self.__label_name = label_name
        self.__splitting = splitting

    def labels_extractor_builder(self):
        return builder.wrap_word_level(extractors.PosDictFeatureExtractorBuilder(self.__label_name, self.__splitting))


class LemmaTransformationExtractorProvider(AbstractExtractorBuilderProvider):
    """
    Provider for lemma transformation rule id (vocabulary) label representation.

    This provider additionaly provides rules applicability feature.
    """

    def __init__(self, feature_name: str, label_name: str, config):
        self.__feature_name = feature_name
        self.__label_name = label_name

        self.__rules_builder = extractors.CachedSuffixRulesBuilder(config['rules_threshold'])

    def labels_extractor_builder(self):
        return builder.wrap_word_level(extractors.LemmaSuffixTransformationFeatureExtractorBuilder(self.__label_name, self.__rules_builder))

    def feature_extractor_builder(self):
        return builder.wrap_word_level(extractors.ApplicableSuffixTransformationsBuilder(self.__feature_name, self.__rules_builder))


class LemmaCharExtractorProvider(AbstractExtractorBuilderProvider):
    def __init__(self, label_name: str, lemma_characters_config):
        self.__label_name = label_name
        self.__lemma_characters_config = lemma_characters_config

    def labels_extractor_builder(self):
        return builder.wrap_word_level(
            extractors.LemmaCharFeatureExtractorBuilder(self.__label_name, self.__lemma_characters_config))
