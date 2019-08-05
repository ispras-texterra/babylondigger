from typing import Dict, List, Tuple

import tensorflow as tf

from babylondigger.neural.initializers import NeuralNetworkInitializer, NeuralNetworkInitializerMapping
from babylondigger.neural.utils import global_variables_initializer
from babylondigger.features.provider import AbstractExtractorBuilderProvider

"""
Interface for TensorFlow computational graphs
"""

########################### Neural Network ###########################

_optimizers = {
    'GD': tf.train.GradientDescentOptimizer,
    'Adam': tf.train.AdamOptimizer,
    'RMSProp': tf.train.RMSPropOptimizer,
}

class NeuralNetwork(object):
    def __init__(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor], gold: Dict[str, tf.Tensor],
                 optimizer: tf.Operation, loss: tf.Tensor, initializers: List['NeuralNetworkInitializer']):
        self._inputs = inputs
        self._outputs = outputs
        self._gold = gold
        self._optimizer = optimizer
        self._initializers = initializers
        self._loss = loss

    def get_optimizer(self):
        return self._loss, self._optimizer

    def get_outputs(self) -> Dict[str, tf.Tensor]:
        return self._outputs

    def get_initializers(self) -> List['NeuralNetworkInitializer']:
        return self._initializers

    def get_feed_dict(self, input: Dict[str, object], gold: Dict[str, object] = None) -> Dict[tf.Tensor, object]:
        feed_dict = _fill_dict(self._inputs, input)
        if gold is not None:
            feed_dict.update(_fill_dict(self._gold, gold))
        return feed_dict

    def mapping(self) -> 'NeuralNetworkMapping':
        return NeuralNetworkMapping(self)


class NeuralNetworkMapping(object):
    def __init__(self, nn: NeuralNetwork):
        self.inputs = _tensor2name(nn._inputs)
        self.outputs = _tensor2name(nn._outputs)
        self.gold = _tensor2name(nn._gold)
        self.optimizer = nn._optimizer.name
        self.loss = nn._loss.name
        self.initializers = _convert_initializers(nn._initializers)

    def attach(self, tf_graph: tf.Graph) -> NeuralNetwork:
        return NeuralNetwork(
            _name2tensor(tf_graph, self.inputs),
            _name2tensor(tf_graph, self.outputs),
            _name2tensor(tf_graph, self.gold),
            tf_graph.get_operation_by_name(self.optimizer),
            tf_graph.get_tensor_by_name(self.loss),
            _attach_initializers(tf_graph, self.initializers)
        )

####################### Neural Network Builder #######################


class NeuralNetworkBuilder(object):

    def build(self, shapes) -> Tuple[NeuralNetwork, tf.Graph]:
        pass

    def feature_providers(self) -> List[AbstractExtractorBuilderProvider]:
        pass


_TensorMap = Dict[str, tf.Tensor]


class AbstractTaskNetworkBuilder(object):

    def feature_providers(self) -> List[AbstractExtractorBuilderProvider]:
        return []

    def inputs(self, shapes: Dict[str, List[int]]) -> _TensorMap:
        return {}

    def gold(self, shapes: Dict[str, List[int]]) -> _TensorMap:
        pass

    def build(self, shapes: Dict[str, List[int]], inputs: _TensorMap, shared: _TensorMap, gold: _TensorMap, dropout: tf.Tensor) -> Tuple[_TensorMap, tf.Tensor]:
        pass

    def initializers(self) -> List['NeuralNetworkInitializer']:
        pass


class AbstractSharedNetworkBuilder(object):

    def feature_providers(self) -> List[AbstractExtractorBuilderProvider]:
        pass

    def inputs(self, shapes: Dict[str, List[int]]) -> _TensorMap:
        pass

    def build(self, shapes: Dict[str, List[int]], inputs: _TensorMap, dropout: tf.Tensor) -> Tuple[_TensorMap, List['NeuralNetworkInitializer']]:
        pass


class MultitaskNetworkBuilder(NeuralNetworkBuilder):
    def __init__(self, shared_builder: AbstractSharedNetworkBuilder, task_builders: List[AbstractTaskNetworkBuilder], config, seed):
        self.__shared_builder = shared_builder
        self.__task_builders = task_builders
        #change
        self._random_seed = seed
        self._optimizer_dict = config['optimizer']

    def feature_providers(self):
        result = self.__shared_builder.feature_providers()
        for task_builder in self.__task_builders:
            result.extend(task_builder.feature_providers())
        return result

    def build(self, shapes) -> Tuple[NeuralNetwork, tf.Graph]:
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            # global configuration
            tf.set_random_seed(self._random_seed)

            gold = {
                'dropout_probability': tf.placeholder_with_default(1.0, [], name='dropout'),
                'learning_rate': tf.placeholder(tf.float32, shape=(), name='lr')
            }

            # shared part
            inputs = {}
            shared_inputs = self.__shared_builder.inputs(shapes)
            inputs.update(shared_inputs)

            shared, shared_initializers = self.__shared_builder.build(shapes, shared_inputs, gold['dropout_probability'])

            task_initializers = []
            outputs = {}
            losses = []
            # task specific part
            for task_builder in self.__task_builders:
                task_inputs = task_builder.inputs(shapes)
                inputs.update(task_inputs)
                task_inputs.update(shared_inputs)

                task_gold = task_builder.gold(shapes)
                gold.update(task_gold)

                output, loss = task_builder.build(shapes, task_inputs, shared, task_gold, gold['dropout_probability'])
                outputs.update(output)
                losses.append(loss)
                task_initializers.extend(task_builder.initializers())
            if len(losses) > 1:
                loss = tf.reduce_sum(tf.stack(losses), name='loss')
            elif len(losses) == 1:
                loss = tf.reduce_sum(losses[0], name='loss')
            else:
                raise ValueError()

            optimizer = self._optimizer(loss, gold['learning_rate'])

            initializers = [global_variables_initializer()]
            initializers.extend(shared_initializers)
            initializers.extend(task_initializers)

            return NeuralNetwork(inputs, outputs, gold, optimizer, loss, initializers), tf_graph

    def _optimizer(self, loss: tf.Tensor, learning_rate: tf.Tensor) -> tf.Operation:
        try:
            optimizer_type = _optimizers[self._optimizer_dict['type']]
        except KeyError:
            raise ValueError("No such optimizer: {}".format(self._optimizer_dict['type']))

        optimizer = optimizer_type(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)

        op = optimizer.apply_gradients(zip(gradients, variables), name='optimizer_op')
        return op


############################### Helpers ##############################


def _fill_dict(keys: Dict[str, tf.Tensor], values: Dict[str, object]) -> Dict[tf.Tensor, object]:
    result = {}
    for key, value in values.items():
        result[keys[key]] = value
    return result


def _tensor2name(tensors: Dict[str, tf.Tensor]) -> Dict[str, str]:
    return {key: tensor.name for key, tensor in tensors.items()}


def _name2tensor(cgraph: tf.Graph, names: Dict[str, str]) -> Dict[str, tf.Tensor]:
    return {key: cgraph.get_tensor_by_name(name) for key, name in names.items()}


def _convert_initializers(initializers: List[NeuralNetworkInitializer]) -> List[NeuralNetworkInitializerMapping]:
    return [initializer.mapping() for initializer in initializers]


def _attach_initializers(tf_graph: tf.Graph, mappings: List[NeuralNetworkInitializerMapping]) -> List[NeuralNetworkInitializer]:
    return [saver.attach(tf_graph) for saver in mappings]