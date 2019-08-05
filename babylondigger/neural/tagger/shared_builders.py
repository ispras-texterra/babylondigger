import tensorflow as tf

import babylondigger.neural.utils as utils
from babylondigger.neural.network import AbstractSharedNetworkBuilder
from babylondigger.features.provider import SentenceLengthExtractorProvider, WordEmbeddingExtractorProvider, \
    SubwordsExtractorProvider, TrainableWordEmbeddingExtractorProvider, SubwordEmbeddingExtractorProvider


class WECharCNNSharedNetworkBuilder(AbstractSharedNetworkBuilder):

    def __init__(self, config, features_config):
        self._features_config = features_config
        self._inputs_dict = None
        self._build_features()

    def _build_features(self):
        inputs_builders = [lambda: ('sentences_lengths', tf.placeholder(tf.int32, [None], name='sentences_lengths'))]
        providers = [SentenceLengthExtractorProvider('sentences_lengths')]

        feature_builders = []
        if 'word_embedding' in self._features_config:
            word_embedding_config = self._features_config['word_embedding']
            we_initializer_name = 'word_embeddings'
            providers.append(WordEmbeddingExtractorProvider('words', we_initializer_name, word_embedding_config))
            inputs_builders.append(lambda: ('words', tf.placeholder(tf.int32, [None, None], name='words')))

            def we_features_builder(shapes, inputs, dropout):
                # pre-trained word embedding
                we, we_matrix = utils.embedding_layer(we_initializer_name,
                                                      inputs['words'],
                                                      shapes[we_initializer_name],
                                                      word_embedding_config['trainable'])
                we_initializer = utils.matrix_initializer(we_initializer_name, we_matrix)
                if "we_dense_size" not in word_embedding_config:
                    return ("word_embedding", we), [we_initializer]
                else:
                    # transform pre-trained embeddings using a dense layer
                    we_dense_layer = tf.nn.relu(
                        tf.layers.dense(inputs=we, name="we_dense_layer", units=word_embedding_config["we_dense_size"]))
                    return ("word_embedding", we_dense_layer), [we_initializer]

            feature_builders.append(we_features_builder)

        if 'word_trained_embedding' in self._features_config:
            trained_embedding_config = self._features_config['word_trained_embedding']
            providers.append(TrainableWordEmbeddingExtractorProvider('trained_words', trained_embedding_config))
            inputs_builders.append(lambda: ('trained_words', tf.placeholder(tf.int32, [None, None], name='trained_words')))

            def te_features_builder(shapes, inputs, dropout):
                # trained word embeddings
                te_shape = [shapes['trained_words'][0], trained_embedding_config['size']]
                te, _ = utils.embedding_layer('trained_words', inputs['trained_words'], te_shape,
                                              trained_embedding_config['trainable'])
                return ("trained_embedding", te), []

            feature_builders.append(te_features_builder)

        if 'word_characters' in self._features_config:
            results = utils.subwords_features_builder('characters', self._features_config['word_characters'])
            providers.append(results["extractor_provider"])
            inputs_builders.append(results["input_builder"])
            feature_builders.append(results["features_builder"])

        if 'bpe_subwords' in self._features_config:
            results = utils.subwords_features_builder('bpe_subwords', self._features_config['bpe_subwords'])
            providers.append(results["extractor_provider"])
            inputs_builders.append(results["input_builder"])
            feature_builders.append(results["features_builder"])

        if 'morfessor_subwords' in self._features_config:
            results = utils.subwords_features_builder("morfessor_subwords", self._features_config["morfessor_subwords"])
            providers.append(results["extractor_provider"])
            inputs_builders.append(results["input_builder"])
            feature_builders.append(results["features_builder"])

        self.__feature_providers = providers
        self.__input_builders = inputs_builders
        self.__feature_builders = feature_builders

    def feature_providers(self):
        return self.__feature_providers

    def inputs(self, shapes):
        if self._inputs_dict is None:
            self._inputs_dict = dict(input_builder() for input_builder in self.__input_builders)
        return self._inputs_dict

    def build(self, shapes, inputs, dropout):

        result = {}
        word_features = []
        initializers = []

        for feature_builder in self.__feature_builders:
            (name, features), feature_initializers = feature_builder(shapes, inputs, dropout)
            result.update([(name, features)])
            word_features.append(features)
            initializers.extend(feature_initializers)

        # sentence mask
        mask = tf.sequence_mask(inputs['sentences_lengths'], dtype=tf.float32)
        norm = tf.reduce_sum(tf.to_float(inputs['sentences_lengths']))

        result.update({
            'word_features': tf.concat(word_features, axis=-1),
            'sentence_mask': mask,
            'sentence_norm': norm
        })

        return result, initializers


class CharCNNBiLSTMSharedNetworkBuilder(WECharCNNSharedNetworkBuilder):
    def __init__(self, config, features_config):
        WECharCNNSharedNetworkBuilder.__init__(self, config, features_config)
        self.__lstm_cell_size = config['lstm_cell_size']
        self.__skip_connection = config.get('skip_connection', None)
        self.__bilstm_layer_count = config.get('bilstm_layer_count', 1)

    def build(self, shapes, inputs, dropout):
        result, initializers = super().build(shapes, inputs, dropout)
        if self.__bilstm_layer_count <= 0:
            raise ValueError("Incorrect count of BiLSTM layers: {}".format(self.__bilstm_layer_count))

        bilstm_input = tf.nn.dropout(result['word_features'], dropout)
        bilstm = utils.bilstm_layer(bilstm_input, self.__lstm_cell_size, inputs['sentences_lengths'], self.__bilstm_layer_count, dropout)

        result['bilstm'] = bilstm

        if self.__skip_connection is None:
            return result, initializers

        word_features_dropout = tf.nn.dropout(result['word_features'], dropout)

        if self.__skip_connection == 'sum':
            if result["bilstm"].shape.as_list() != word_features_dropout.shape.as_list():
                raise ValueError("Cant sum because bilstm result's shape and word's shape don't match: {} and {} ".
                                 format(result["bilstm"].shape, word_features_dropout.shape))
            skip_connection = tf.add(result['bilstm'], word_features_dropout)

        elif self.__skip_connection == 'concat':
            skip_connection = tf.concat([result['bilstm'], word_features_dropout], axis=2)

        else:
            raise ValueError("Invalid skip-connection type")

        result['bilstm_and_word_features'] = skip_connection
        return result, initializers


class DilatedCNNSharedNetworkBuilder(WECharCNNSharedNetworkBuilder):
    def __init__(self, config, features_config):
        WECharCNNSharedNetworkBuilder.__init__(self, config, features_config)
        self.__skip_connection = config.get('skip_connection', None)
        self.__cnn_kernels = config.get('cnn_kernels', None)

    def build(self, shapes, inputs, dropout):
        result, initializers = super().build(shapes, inputs, dropout)
        if self.__cnn_kernels is None:
            raise ValueError("Sent-level CNN layers are not specified!")

        cnn_input = tf.expand_dims(tf.nn.dropout(result['word_features'], dropout), axis=1)
        dilated_cnn = utils.apply_conv_layer(cnn_input, self.__cnn_kernels, var_name="dilated_cnn")
        result['dilated_cnn'] = tf.reshape(dilated_cnn, [tf.shape(dilated_cnn)[0], tf.shape(dilated_cnn)[2], self.__cnn_kernels[-1]["count"]])

        if self.__skip_connection is None:
            return result, initializers

        word_features_dropout = tf.nn.dropout(result['word_features'], dropout)

        if self.__skip_connection == 'sum':
            if result["dilated_cnn"].shape.as_list() != word_features_dropout.shape.as_list():
                raise ValueError(
                    "Cant sum because dilated_cnn result's shape and word's shape don't match: {} and {} ".
                    format(result["dilated_cnn"].shape, word_features_dropout.shape))
            skip_connection = tf.add(result['dilated_cnn'], word_features_dropout)

        elif self.__skip_connection == 'concat':
            skip_connection = tf.concat([result['dilated_cnn'], word_features_dropout], axis=2)
        else:
            raise ValueError("Invalid skip-connection type")

        result['dilated_cnn_and_word_features'] = skip_connection
        return result, initializers


class CharCNNBiLSTMCNNSharedNetworkBuilder(CharCNNBiLSTMSharedNetworkBuilder):
    def __init__(self, config, features_config):
        CharCNNBiLSTMSharedNetworkBuilder.__init__(self, config, features_config)
        self.__cnn_kernels = config['cnn_kernels']

    def build(self, shapes, inputs, dropout):
        result, initializers = super().build(shapes, inputs, dropout)

        if 'bilstm_and_word_features' in result:
            bilstm_result = result['bilstm_and_word_features']
        else:
            bilstm_result = result['bilstm']

        cnn_input = tf.nn.dropout(bilstm_result, dropout)
        cnn = utils.sent_level_cnn(cnn_input, self.__cnn_kernels)

        result['bilstm_cnn'] = cnn

        return result, initializers
