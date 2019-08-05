import tensorflow as tf
import os.path
import pickle

from babylondigger.neural.network import NeuralNetwork


class Model(object):
    """
    A trainable model based on TensorFlow.
    """

    def __init__(self, network: NeuralNetwork, session: tf.Session):
        self._network = network
        self._session = session

    def initialize(self, **kwargs):
        for initializer in self._network.get_initializers():
            initializer.initialize(self._session, **kwargs)

    def train(self, features, labels):
        """
        Optimizes the model's computational graph's parameters based on given features and their labels.

        :param features: batch of input features for the graph
        :param labels: batch of corresponding labels for the input
        :return: computed value of the graph's optimizer after training on given batch
        """
        feed_dict = self._network.get_feed_dict(input=features, gold=labels)
        loss, _ = self._session.run(self._network.get_optimizer(), feed_dict=feed_dict)
        return loss

    def predict(self, features):
        """
        Predicts labels for given batch of input features.

        :param features: batch of input features for the graph
        :return: predicted labels for given input
        """
        feed_dict = self._network.get_feed_dict(input=features)
        return self._session.run(self._network.get_outputs(), feed_dict=feed_dict)

    def close(self):
        self._session.close()

    def save(self, path):
        """
        Saves the model and returns a loader function so that it can be restored and reused later.

        :param path: path where model files are saved
        :return: a function returning an identical model
        """
        with self._session.graph.as_default():
            tf.train.Saver().save(self._session, _tf_graph_path(path))
        network = self._network.mapping()
        with open(_mapping_path(path), 'wb') as f:
            pickle.dump(network, f)

        return lambda: Model.load(path)

    @classmethod
    def load(cls, path):
        tf_graph = tf.Graph()
        sess = tf.Session(graph=tf_graph)
        try:
            with tf_graph.as_default():
                tf_graph_path = _tf_graph_path(path)
                saver = tf.train.import_meta_graph(tf_graph_path+ '.meta')
                saver.restore(sess, tf_graph_path)
            with open(_mapping_path(path), 'rb') as f:
                network = pickle.load(f)
            return Model(network.attach(tf_graph), sess)
        except Exception as e:
            sess.close()
            raise e

    @classmethod
    def new(cls, network: NeuralNetwork, tf_graph: tf.Graph):
        return Model(network, tf.Session(graph=tf_graph))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _tf_graph_path(path: str) -> str:
    return os.path.join(path, 'graph')


def _mapping_path(path: str) -> str:
    return os.path.join(path, 'mapping')
