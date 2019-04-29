import tensorflow
from tensorflow import python as tf_python

class GFileProxy(object):
    """Provides a subset of the tensorflow-2.0 API and proxies the calls through to the
    appropriate tensorflow API."""

    def Open(*args, **kwargs):
        pass

    def makedirs(*args, **kwargs):
        pass

    def exists(*args, **kwargs):
        pass

    def rmtree(*args, **kwargs):
        pass

    def glob(*args, **kwargs):
        pass

    def remove(*args, **kwargs):
        pass

    def copy(*args, **kwargs):
        pass


class IOProxy(object):
    def __init__(self):
        self.gfile = GFileProxy()

    def __getattr__(self, name):
        return getattr(tensorflow.io, name)


class PythonProxy(object):
    def __getattr__(self, name):
        return getattr(tensorflow.python, name)


class SavedModelProxy(object):
    def __getattr__(self, name):
        return getattr(tensorflow.saved_model, name)


class TensorflowProxy(object):
    """A proxy object for tensorflow calls.

    This proxy system aims to simplify the use of tensorflow across different
    versions (1.11-1.14, 2.x) despite the fact that the APIs have drastically
    changed.
    """

    def __init__(self):
        self.io = IOProxy()
        self.python = tf_python
        self.saved_model = SavedModelProxy()

    def __getattr__(self, name):
        return getattr(tensorflow, name)
