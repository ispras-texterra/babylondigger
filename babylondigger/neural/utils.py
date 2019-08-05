from typing import List, Tuple, Dict

import tensorflow as tf
import math

from babylondigger.features.provider import SubwordsExtractorProvider, SubwordEmbeddingExtractorProvider
from babylondigger.neural.initializers import NeuralNetworkInitializer, MatrixInitializer, GlobalInitializer


############################### Layers ###############################


def embedding_layer(name: str, input_: tf.Tensor, shape: List[int], trainable: bool) -> Tuple[tf.Tensor, tf.Variable]:
    bound = (3 / shape[1]) ** 0.5
    we_matrix = tf.get_variable(name=name,
                                shape=shape,
                                initializer=tf.random_uniform_initializer(-bound, bound),
                                trainable=trainable)
    return tf.nn.embedding_lookup(we_matrix, input_), we_matrix


def apply_conv_layer(input_: tf.Tensor, kernels: List[Dict[str, int]], var_name: str):
    channel_count = input_.shape[-1]
    for i, conv_layer in enumerate(kernels):
        kernel = tf.get_variable(
            '{}_{}_s{}_c{}_d{}'.format(var_name, i, conv_layer['size'], conv_layer['count'], conv_layer['dilation']),
            [1, conv_layer['size'], channel_count, conv_layer['count']])
        input_ = tf.nn.relu(tf.nn.conv2d(input_, kernel, [1, 1, 1, 1], 'SAME',
                                         dilations=[1, 1, conv_layer['dilation'], 1]))
        channel_count = conv_layer['count']
    return input_


def subword_rnn(ids, input_: tf.Tensor, config, feature_name):
    max_word_length = tf.shape(input_)[2]
    max_sent_length = tf.shape(input_)[1]
    batch_size = tf.shape(input_)[0]
    ce_size = input_.shape[3]
    bilstm_input = tf.reshape(input_, [-1, max_word_length, ce_size])

    hidden_layer = bilstm_layer(bilstm_input, config["lstm_cell_size"], None, config["bilstm_layer_count"], scope=feature_name)

    dense = tf.layers.dense(hidden_layer, units=hidden_layer.get_shape()[-1], activation=tf.nn.relu)
    dense = tf.reshape(dense, [batch_size, max_sent_length, max_word_length, hidden_layer.get_shape()[-1]])
    return tf.reduce_max(dense, 2)


def subword_cnn(ids, input_: tf.Tensor, config, feature_name):
    max_sent_length = tf.shape(input_)[1]
    max_word_length = tf.shape(input_)[2]
    ce_size = input_.shape[3]
    reshaped = tf.reshape(input_, [-1, 1, max_word_length, ce_size])

    reshaped = apply_conv_layer(reshaped, config["cnn_kernels"], "{}_cnn".format(feature_name))
    pooled = [tf.reduce_max(reshaped, 2)]
    return tf.reshape(tf.concat(pooled, axis=-1), [-1, max_sent_length, config["cnn_kernels"][-1]['count']])


def subword_average(ids, input_: tf.Tensor, config, feature_name):
    zero = tf.constant(0)
    mask = tf.expand_dims(tf.not_equal(ids, zero), axis=-1)
    mask_shape = tf.shape(mask)
    input_shape = tf.shape(input_)
    broadcast_mask = tf.broadcast_to(mask, [mask_shape[0], mask_shape[1], mask_shape[2], input_shape[-1]])
    nonzero_input_ = tf.ragged.boolean_mask(input_, broadcast_mask).to_tensor()
    nonzero_input_reshaped = tf.reshape(nonzero_input_, [input_shape[0], input_shape[1], input_shape[2], input_.get_shape().as_list()[-1]])
    return tf.reduce_mean(nonzero_input_reshaped, axis=-2)


def subword_self_attention(ids, input_: tf.Tensor, config, feature_name):
    sum = tf.reduce_sum(input_, axis=-2)
    scale = math.sqrt(sum.get_shape().as_list()[-1])
    sum_scale = tf.scalar_mul(scale, sum)
    dot_product = tf.einsum('bsce,bse->bsc', input_, sum_scale)
    softmax = tf.nn.softmax(dot_product, axis=-1)
    scalar_mul = tf.multiply(input_, tf.expand_dims(softmax, axis=-1))
    return tf.reduce_sum(scalar_mul, axis=-2)


_subword_aggregators = {
    "rnn": subword_rnn,
    "cnn": subword_cnn,
    "average": subword_average,
    "self-attention": subword_self_attention
}


def subwords_features_builder(feature_name, config):

    if 'embedding_path' not in config:
        provider = SubwordsExtractorProvider(feature_name, config)
    else:
        provider = SubwordEmbeddingExtractorProvider(feature_name, feature_name, config)

    def input_builder():
        return feature_name, tf.placeholder(tf.int32, [None, None, None], name=feature_name)

    def features_builder(shapes, inputs, dropout):
        initializers = []
        sid = inputs[feature_name]
        if 'embedding_path' not in config:
            se_shape = [shapes[feature_name][0], config['embed_size']]
            se, _ = embedding_layer(feature_name, sid, se_shape, trainable=True)
        else:
            # with pre-trained embeddings
            se, se_matrix = embedding_layer(feature_name, sid, shapes[feature_name], config['trainable'])
            initializers.append(matrix_initializer(feature_name, se_matrix))

        subwords_features = _subword_aggregators[config["aggregator"]](sid, se, config.get("aggregator_config", {}), feature_name)

        if config.get("highway_layer", False):
            dense = tf.layers.dense(subwords_features, units=subwords_features.get_shape()[-1],
                                name="{}_dense_highway".format(feature_name), activation=tf.nn.relu)
            subwords_features = highway(subwords_features, dense)

        return (feature_name, subwords_features), initializers

    return {
        "extractor_provider": provider,
        "input_builder": input_builder,
        "features_builder": features_builder
    }


def bilstm_layer(input_: tf.Tensor, cell_size: int, sequence_lengths: tf.Tensor, bilstm_layer_count, dropout=.0, scope=""):
    hidden_layer = input_
    for i in range(bilstm_layer_count):
        (fw_lstm_output, bw_lstm_output), _ = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(cell_size), tf.nn.rnn_cell.LSTMCell(cell_size), hidden_layer, sequence_lengths, dtype=tf.float32,
            scope='bidirectional_rnn_{}-{}'.format(scope, i))
        if i == 0:
            hidden_layer = tf.layers.dropout(tf.concat([fw_lstm_output, bw_lstm_output], axis=-1), dropout)
        else:
            hidden_layer = tf.layers.dropout(
                tf.math.add(hidden_layer, tf.concat([fw_lstm_output, bw_lstm_output], axis=-1)), dropout)

    return hidden_layer


def sent_level_cnn(input_: tf.Tensor, kernels: List[Dict[str, int]]):
    max_sent_length = tf.shape(input_)[1]
    w_size = input_.shape[2]

    reshaped = tf.reshape(input_, [-1, 1, w_size, 1])
    reshaped = apply_conv_layer(reshaped, kernels, 'cnn_kernel')
    pooled = [tf.reduce_max(reshaped, 2)]
    return tf.reshape(tf.concat(pooled, axis=-1), [-1, max_sent_length, kernels[-1]['count']])


def highway(x, y):
    transform_gate = tf.layers.dense(x, units=x.get_shape()[-1], activation=tf.sigmoid)
    carry_gate = tf.subtract(1.0, transform_gate)
    return tf.add(tf.multiply(transform_gate, y), tf.multiply(carry_gate, x))

############################ Initializers ############################


def matrix_initializer(name: str, matrix: tf.Variable) -> NeuralNetworkInitializer:
    indices = tf.placeholder(tf.int32, [None])
    values = tf.placeholder(tf.float32, [None, matrix.shape[1]])
    update_op = tf.scatter_update(matrix, indices, values)
    return MatrixInitializer(name, indices, values, update_op)


def global_variables_initializer() -> NeuralNetworkInitializer:
    return GlobalInitializer(tf.global_variables_initializer())
