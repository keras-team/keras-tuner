from kerastuner.abstractions.tensorflow import proxy
import tensorflow
from tensorflow.python import Graph, GraphDef, Session
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow import gfile
from tensorflow import python


class GFileProxy_1_x(proxy.GFileProxy):
    """Proxies calls through to the appropriate tensorflow API."""

    def __init__(self):
        renames = {
            "makedirs": gfile.MakeDirs,
            "exists": gfile.Exists,
            "rmtree": gfile.MakeDirs,
            "glob": gfile.Glob,
            "remove": gfile.Remove,
            "copy": gfile.Copy
        }

    def __getattr__(self, name):
        if name in renames.keys():
            return renames[name]

        return getattr(gfile, name)

    def Open(self, *args, **kwargs):
        return gfile.Open(*args, **kwargs)

    def makedirs(self, *args, **kwargs):
        return gfile.MakeDirs(*args, **kwargs)

    def exists(self, *args, **kwargs):
        return gfile.Exists(*args, **kwargs)

    def rmtree(self, *args, **kwargs):
        return gfile.DeleteRecursively(*args, **kwargs)

    def glob(self, *args, **kwargs):
        return gfile.Glob(*args, **kwargs)

    def remove(self, *args, **kwargs):
        return gfile.Remove(*args, **kwargs)

    def copy(self, *args, **kwargs):
        return gfile.Copy(*args, **kwargs)


class IOProxy_1_x(proxy.IOProxy):
    def __init__(self):
        self.gfile = GFileProxy_1_x()


class PythonProxy_1_x(proxy.PythonProxy):
    def __getattr__(self, name):
        raise ValueError(
            "Attribute '%s' not supported" % name)

    def Session(self, *args, **kwargs):
        """Passthrough to create a new Session."""
        return tf.Session(*args, **kwargs)

    def GraphDef(self, *args, **kwargs):
        """Passthrough to create a new Session."""
        return tf.GraphDef(*args, **kwargs)

    def Graph(self, *args, **kwargs):
        """Passthrough to create a new Session."""
        return tf.Graph(*args, **kwargs)


class Tensorflow_1_x(proxy.TensorflowProxy):
    def __init__(self):
        super(Tensorflow_1_x, self).__init__()
        self.io = IOProxy_1_x()
