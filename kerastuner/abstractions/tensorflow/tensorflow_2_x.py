from kerastuner.abstractions.tensorflow import proxy
import tensorflow as tf
from tensorflow.python import GraphDef
from tensorflow.python import Session as python_session
from tensorflow.io import gfile
import os
import sys
import subprocess
from tensorflow.python.tools import optimize_for_inference_lib


class GFileProxy_2_x(proxy.GFileProxy):
    """Proxies calls through to the appropriate tensorflow API."""

    def Open(self, *args, **kwargs):
        return gfile.GFile(*args, **kwargs)

    def makedirs(self, *args, **kwargs):
        return gfile.makedirs(*args, **kwargs)

    def exists(self, *args, **kwargs):
        return gfile.exists(*args, **kwargs)

    def rmtree(self, *args, **kwargs):
        return gfile.rmtree(*args, **kwargs)

    def glob(self, *args, **kwargs):
        return gfile.glob(*args, **kwargs)

    def remove(self, *args, **kwargs):
        return gfile.remove(*args, **kwargs)

    def copy(self, *args, **kwargs):
        return gfile.copy(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(gfile, name)


class IOProxy_2_x(proxy.IOProxy):
    def __init__(self):
        self.gfile = GFileProxy_2_x()


class Tensorflow_2_x(proxy.TensorflowProxy):
    def __init__(self):
        super(Tensorflow_2_x, self).__init__()
        self.io = IOProxy_2_x()
