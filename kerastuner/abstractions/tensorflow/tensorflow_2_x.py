from kerastuner.abstractions.tensorflow import base
from kerastuner.abstractions.tensorflow import proxy
import tensorflow as tf


class GFileProxy_2_x(proxy.GFileProxy):
    """Proxies calls through to the appropriate tensorflow API."""

    def __getattr__(self, name):
        return getattr(tf.io.gfile, name)


class IOProxy_2_x(proxy.IOProxy):
    def __init__(self):
        self.gfile = GFileProxy_2_x()


class Tensorflow_2_x(base.Tensorflow):
    def __init__(self):
        self.io = IOProxy_2_x()
        self.python = proxy.PythonProxy()

    def save_savedmodel(self, model, path, tmp_path):
        tf.saved_model.save(model, path)
