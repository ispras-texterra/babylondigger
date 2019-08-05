import tensorflow as tf
import numpy as np

from babylondigger.features.builder import PADDING
from babylondigger.neural.network import AbstractTaskNetworkBuilder
from babylondigger.features.provider import PosDictExtractorProvider, LemmaDictExtractorProvider, \
    LemmaTransformationExtractorProvider, LemmaCharExtractorProvider, SubwordEmbeddingExtractorProvider, \
    SubwordsExtractorProvider
from babylondigger.neural import utils


class DenseTaskNetworkBuilder(AbstractTaskNetworkBuilder):
    def __init__(self, labels_name, config):
        self.__labels_name = labels_name
        self.__shared_name = config['shared_name']
        self.__dense_size = config['dense_size']
        self.__loss_weights = config.get("loss_weights", None)
        self.__initializers = []

    def gold(self, shapes):
        return {
            self.__labels_name:  tf.placeholder(tf.int32, [None, None, len(shapes[self.__labels_name])], name=self.__labels_name)
        }

    def initializers(self):
        return self.__initializers

    def build(self, shapes, inputs, shared, gold, dropout):
        losses = {}
        outputs = []
        i = 0
        list_gold = tf.unstack(gold[self.__labels_name], axis=-1)
        for pos_i in shapes[self.__labels_name]:
            with tf.variable_scope(str(i)):
                logits = self._logits(shapes[self.__labels_name][pos_i],
                                 inputs,
                                 shared[self.__shared_name],
                                 dropout)
                outputs.append(self._output(inputs, logits, 'predicted_' + self.__labels_name))
                losses[pos_i] = self._loss(logits, list_gold[i], shared['sentence_mask'], shared['sentence_norm'])
            i += 1
        loss = self._sum_losses(losses)

        output = {
            self.__labels_name: tf.stack(
                outputs,
                name='Concatenated_output',
                axis=-1
            )
        }
        return output, loss

    def _sum_losses(self, losses):
        if self.__loss_weights is None:
            return tf.reduce_sum(tf.stack(losses), name='loss')

        weighted_losses = []
        for loss in losses:
            weight = self.__loss_weights[loss] if loss in self.__loss_weights else self.__loss_weights["default"]
            weighted_losses.append(tf.math.multiply(losses[loss], tf.convert_to_tensor(weight, dtype=tf.float32)))
        return tf.reduce_sum(tf.stack(weighted_losses), name='loss')

    def _logits(self, shapes, inputs, shared, dropout):
        dense = tf.layers.dense(shared, units=self.__dense_size, activation=tf.nn.relu,
                                name='dense_' + self.__labels_name)
        dense_with_dropout = tf.nn.dropout(dense, dropout)
        return tf.layers.dense(dense_with_dropout, units=shapes, name='logits_'+self.__labels_name)

    def _loss(self, logits, labels, sentence_mask, sentence_norm):
        raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_sum(raw_loss * sentence_mask) / sentence_norm

    def _output(self, inputs, logits, name):
        return tf.cast(tf.argmax(logits, axis=-1), tf.int32, name=name)


class PosDenseTaskNetworkBuilder(DenseTaskNetworkBuilder):
    def __init__(self, config, features_config):
        DenseTaskNetworkBuilder.__init__(self, 'pos_labels', config)
        self.__splitting = config["splitting"]

    def feature_providers(self):
        return [PosDictExtractorProvider('pos_labels', self.__splitting)]


class LemmaDenseTaskNetworkBuilder(DenseTaskNetworkBuilder):
    def __init__(self, config, features_config):
        DenseTaskNetworkBuilder.__init__(self, 'lemma_labels', config)

    def feature_providers(self):
        return [LemmaDictExtractorProvider('lemma_labels')]


class LemmaSuffixTaskNetworkBuilder(DenseTaskNetworkBuilder):
    def __init__(self, config, features_config):
        DenseTaskNetworkBuilder.__init__(self, 'lemma_rule_labels', config)
        self.__lemma_rules_config = features_config['lemma_suffix_transformation']

    def feature_providers(self):
        return [LemmaTransformationExtractorProvider('applicable_rules', 'lemma_rule_labels', self.__lemma_rules_config)]

    def inputs(self, shapes):
        return {
            'applicable_rules': tf.placeholder(tf.bool, [None, None, shapes['applicable_rules'][0]])
        }

    def _logits(self, shapes, inputs, shared, dropout):
        logits = super()._logits(shapes, inputs, shared, dropout)
        return tf.where(
            inputs['applicable_rules'],
            logits,
            tf.fill(tf.shape(logits), np.finfo(np.float32).min))


class LemmaCOMBOTaskNetworkBuilder(AbstractTaskNetworkBuilder):
    def __init__(self, config,  features_config):
        self.__labels_name = 'output_subwords'
        self.__features_config = features_config
        self.__shared_name = config['shared_name']
        self.__cnn_char_embed_size = features_config["input_subwords"]["embed_size"]
        self.__cnn_bilstm_dense_size = config["cnn_bilstm_dense_size"]
        self.__cnn_kernels = config["cnn_kernels"]
        self.__max_word_length = features_config["output_subwords"]["max_word_length"]
        self.__output_subwords_config = features_config["output_subwords"]
        self.__input_subwords_config = features_config["input_subwords"]
        self.__loss_weight = config.get("loss_weights")["LEMMA"]
        self.__initializers = []

    def gold(self, shapes):
        return {
            self.__labels_name:  tf.placeholder(tf.int32, [None, None, self.__max_word_length], name=self.__labels_name)
        }

    def feature_providers(self):
        feature_providers = [LemmaCharExtractorProvider('output_subwords', self.__output_subwords_config)]
        if "embedding_path" in self.__features_config["input_subwords"]:
            feature_providers.append(
                SubwordEmbeddingExtractorProvider('input_subwords', 'input_subwords', self.__input_subwords_config))
        else:
            feature_providers.append(
                SubwordsExtractorProvider('input_subwords', self.__input_subwords_config))
        return feature_providers

    def inputs(self, shapes):
        return {
            'input_subwords': tf.placeholder(tf.int32, [None, None, None], name='input_subwords')
        }

    def build(self, shapes, inputs, shared, gold, dropout):

        logits = self._logits(shapes, inputs, shared[self.__shared_name], dropout)
        loss = self._loss(logits, gold[self.__labels_name], shared['sentence_mask'], shared['sentence_norm'])
        output = {
            self.__labels_name: tf.concat(
                [self._output(logits, 'predicted_' + self.__labels_name)],
                name='Concatenated_output',
                axis=-1
            )
        }
        return output, loss

    def initializers(self):
        return self.__initializers

    def _logits(self, shapes, inputs, shared, dropout):
        ce = self._get_char_embeddings(inputs['input_subwords'], [shapes['input_subwords'][0], self.__cnn_char_embed_size])

        # reduce shared's output
        dense = tf.layers.dense(shared, units=self.__cnn_bilstm_dense_size, name="dense_cnn", activation=tf.tanh)
        dense = tf.expand_dims(dense, 2)

        # broadcast shared's output to input chars' shape
        ce_shape = tf.shape(ce)
        broadcast = tf.broadcast_to(dense, [ce_shape[0], ce_shape[1], self.__max_word_length, self.__cnn_bilstm_dense_size])

        # apply dilated convolutions to concatenated char embeddings and shared's output
        cnn_input = tf.reshape(tf.concat([broadcast, ce], axis=-1),
                               [-1, 1, self.__max_word_length, self.__cnn_char_embed_size + self.__cnn_bilstm_dense_size])
        cnn_output = utils.apply_conv_layer(cnn_input, self.__cnn_kernels, 'cnn_kernel')

        if self.__input_subwords_config.get("highway_layer", False):
            # apply highway layer
            ce = tf.reshape(ce, [-1, 1, self.__max_word_length, self.__cnn_char_embed_size])
            cnn_output = utils.highway(ce, cnn_output)

        # use conv1d to resize final char vectors to char dictionary size
        output_kernels = tf.get_variable(self.__labels_name, [1, 1, cnn_output.shape[-1], shapes[self.__labels_name][0]])
        logits = tf.nn.conv2d(cnn_output, output_kernels, [1, 1, 1, 1], "SAME")
        return tf.reshape(logits, [ce_shape[0], ce_shape[1], self.__max_word_length, shapes[self.__labels_name][0]])

    def _loss(self, logits, labels, sentence_mask, sentence_norm):
        raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        sentence_mask = tf.expand_dims(sentence_mask, -1)
        normed_loss = tf.truediv(tf.reduce_sum(raw_loss * sentence_mask), sentence_norm * self.__max_word_length, name="loss")
        weighted_loss = tf.math.multiply(normed_loss, tf.convert_to_tensor(self.__loss_weight, tf.float32))
        return weighted_loss

    def _output(self, logits, name):
        return tf.cast(tf.argmax(logits, axis=-1), tf.int32, name=name)

    def _get_char_embeddings(self, input_, shape):
        ce, ce_matrix = utils.embedding_layer('input_subwords_e', input_, shape, trainable=True)
        if "embedding_path" in self.__features_config["input_subwords"]:
            self.__initializers.append(utils.matrix_initializer('input_subwords', ce_matrix))

        def pad_to_max_len():
            return tf.pad(ce, [[0, 0], [0, 0], [0, self.__max_word_length - tf.shape(ce)[2]], [0, 0]],
                          'CONSTANT', constant_values=PADDING)

        def slice_to_max_len():
            return tf.slice(input_=ce, begin=[0, 0, 0, 0], size=[-1, -1, self.__max_word_length, -1])

        return tf.cond(tf.shape(ce)[2] > self.__max_word_length, slice_to_max_len, pad_to_max_len)


class LemmaSuffixBinaryTaskNetworkBuilder(AbstractTaskNetworkBuilder):
    def __init__(self, config, features_config):
        self.__labels_name = 'lemma_rule_labels'
        self.__shared_name = config['shared_name']
        self.__dense_size = config['dense_size']
        self.__lemma_rules_config = features_config['lemma_suffix_transformation']

    def gold(self, shapes):
        return {
            self.__labels_name:  tf.placeholder(tf.int32, [None, None, len(shapes[self.__labels_name])], name=self.__labels_name)
        }

    def feature_providers(self):
        return [LemmaTransformationExtractorProvider('applicable_rules', 'lemma_rule_labels', self.__lemma_rules_config)]

    def inputs(self, shapes):
        return {
            'applicable_rules': tf.placeholder(tf.bool, [None, None, shapes['applicable_rules'][0]])
        }

    def build(self, shapes, inputs, shared, gold, dropout):
        logits = self._logits(shapes, inputs, shared[self.__shared_name], dropout)

        output = self._output(inputs, logits, 'predicted_' + self.__labels_name)
        output = {
            self.__labels_name: tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[1], 1], name="bin_lemma_output")
        }
        list_gold = tf.unstack(gold[self.__labels_name], axis=-1)[0]
        gold_one_hots = tf.one_hot(list_gold, shapes[self.__labels_name]["LEMMA"])
        loss = self._loss(logits, gold_one_hots, shared['sentence_mask'], shared['sentence_norm'])
        return output, loss

    def _logits(self, shapes, inputs, shared, dropout):
        dense = tf.layers.dense(shared, units=self.__dense_size, name='dense_'+self.__labels_name)
        dense_with_dropout = tf.nn.dropout(dense, dropout)
        logits_list = []
        for rule_i in range(shapes[self.__labels_name]["LEMMA"]):
            rule_embedding = tf.one_hot(rule_i, shapes[self.__labels_name]["LEMMA"])
            rule_embedding_b = tf.broadcast_to(rule_embedding, [tf.shape(dense)[0], tf.shape(dense)[1], shapes[self.__labels_name]["LEMMA"]])
            logit_i = tf.layers.dense(tf.concat([dense_with_dropout, rule_embedding_b], axis=-1),
                                      units=1, name='binary_layer', reuse=tf.AUTO_REUSE)
            logits_list.append(logit_i)
        logits = tf.concat(logits_list, name='Concatenated_logits', axis=-1)

        return tf.where(
            inputs['applicable_rules'],
            logits,
            tf.fill(tf.shape(logits), np.finfo(np.float32).min))

    def _loss(self, logits, labels, sentence_mask, sentence_norm):
        raw_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_sum(raw_loss * tf.expand_dims(sentence_mask, axis=-1)) / sentence_norm

    def _output(self, inputs, logits, name):
        return tf.cast(tf.argmax(logits, axis=-1), tf.int32, name=name)
