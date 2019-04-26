from kerastuner.abstractions.tensorflow import proxy
from kerastuner.abstractions.tensorflow import base
import tensorflow as tf

from tensorflow import gfile


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
        return getattr(tf.io, name)

    def Session(self, *args, **kwargs):
        """Passthrough to create a new Session."""
        return tf.Session(*args, **kwargs)


class Tensorflow_1_x(base.Tensorflow):
    def __init__(self):
        self.io = IOProxy_1_x()
        self.python = PythonProxy_1_x()

    def save_savedmodel(self, model, path, tmp_path):
        session = tf.keras.backend.get_session()
        tf.io.write_graph(session.graph, "/tmp/session/", "graph.pbtxt")
        tf.io.write_graph(tf.get_default_graph(),
                          "/tmp/session/", "graph.pbtxt")
        inputs = dict([(node.op.name, node) for node in model.inputs])
        outputs = dict([(node.op.name, node) for node in model.outputs])
        tf.compat.v1.saved_model.simple_save(session, path, inputs, outputs)
