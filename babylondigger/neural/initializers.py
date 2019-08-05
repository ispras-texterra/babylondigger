import tensorflow as tf


class NeuralNetworkInitializerMapping(object):
    def attach(self, tf_graph: tf.Graph) -> 'NeuralNetworkInitializer':
        pass


class NeuralNetworkInitializer(object):
    def initialize(self, session: tf.Session, **kwargs) -> None:
        pass

    def mapping(self) -> NeuralNetworkInitializerMapping:
        pass


class GlobalInitializer(NeuralNetworkInitializer):

    class GlobalInitializerMapping(NeuralNetworkInitializerMapping):
        def __init__(self, op_name: str):
            self.__op_name = op_name

        def attach(self, tf_graph: tf.Graph) -> 'GlobalInitializer':
            return GlobalInitializer(tf_graph.get_operation_by_name(self.__op_name))

    def __init__(self, init_op: tf.Operation):
        self.__init_op = init_op

    def initialize(self, session: tf.Session, **kwargs) -> None:
        with session.graph.as_default():
            session.run(self.__init_op)

    def mapping(self) -> NeuralNetworkInitializerMapping:
        return GlobalInitializer.GlobalInitializerMapping(self.__init_op.name)


class MatrixInitializer(NeuralNetworkInitializer):

    class MatrixInitializerMapping(NeuralNetworkInitializerMapping):
        def __init__(self, param_name: str, indices: str, values: str, update_op: str):
            self.__param_name = param_name
            self.__indices = indices
            self.__values = values
            self.__update_op = update_op

        def attach(self, tf_graph: tf.Graph) -> 'MatrixInitializer':
            return MatrixInitializer(
                self.__param_name, tf_graph.get_tensor_by_name(self.__indices),
                tf_graph.get_tensor_by_name(self.__values),
                tf_graph.get_tensor_by_name(self.__update_op)
            )

    def __init__(self, param_name: str, indices: tf.Tensor, values: tf.Tensor, update_op: tf.Tensor):
        self.__param_name = param_name
        self.__indices = indices
        self.__values = values
        self.__update_op = update_op

    def initialize(self, session: tf.Session, **kwargs):
        embeddings = kwargs[self.__param_name]
        indices, values = zip(*embeddings)
        session.run(self.__update_op, {self.__indices: indices, self.__values: values})

    def mapping(self) -> NeuralNetworkInitializerMapping:
        return MatrixInitializer.MatrixInitializerMapping(
            self.__param_name, self.__indices.name, self.__values.name, self.__update_op.name)
