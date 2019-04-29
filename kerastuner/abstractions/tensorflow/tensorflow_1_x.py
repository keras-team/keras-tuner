import gc

import tensorflow as tf
from tensorflow import gfile, python
from tensorflow.python import Graph, GraphDef, Session, ConfigProto
from tensorflow.tools.graph_transforms import TransformGraph

from kerastuner.abstractions.tensorflow import proxy


class GFileProxy_1_x(proxy.GFileProxy):
    """Proxies calls through to the appropriate tensorflow API."""

    def __init__(self):
        self.renames = {
            "makedirs": gfile.MakeDirs,
            "exists": gfile.Exists,
            "rmtree": gfile.MakeDirs,
            "glob": gfile.Glob,
            "remove": gfile.Remove,
            "copy": gfile.Copy
        }

    def __getattr__(self, name):
        if name in self.renames.keys():
            return self.renames[name]

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


class Utils_1_x(proxy.UtilsBase):
    def load_savedmodel(
            self,
            session,
            export_dir,
            tags=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):

        return tf.saved_model.load(
            session,
            tags,
            export_dir)

    def save_savedmodel(self, model, path, tmp_path):
        session = tf.keras.backend.get_session()
        self.write_graph(session.graph, tmp_path, "graph.pbtxt")
        self.write_graph(tf.get_default_graph(),
                         tmp_path, "graph.pbtxt")
        inputs = dict([(node.op.name, node) for node in model.inputs])
        outputs = dict([(node.op.name, node) for node in model.outputs])
        tf.compat.v1.saved_model.simple_save(session, path, inputs, outputs)

    def save_tflite(self, model, path, tmp_path, post_training_quantize=True):
        # First, create a SavedModel in the temporary directory
        savedmodel_path = os.path.join(tmp_path, "savedmodel")
        savedmodel_tmp_path = os.path.join(tmp_path, "savedmodel_tmp")

        input_ops = self.get_input_ops(model)
        output_ops = self.get_output_ops(model)

        save_savedmodel(model, savedmodel_path, savedmodel_tmp_path)

        self.convert_to_tflite(
            model, savedmodel_path, path, post_training_quantize=post_training_quantize)

    def convert_to_tflite(
            self,
            model,
            savedmodel_path,
            output_path,
            post_training_quantize):
        converter = tf.contrib.lite.TFLiteConverter.from_saved_model(
            savedmodel_path)
        converter.post_training_quantize = post_training_quantize
        write_file(path, converter.convert())

    def optimize_graph(
            self,
            frozen_model_path,
            input_ops,
            output_ops,
            input_types,
            toco_compatible):

        transforms = [
            "fold_constants",
            "fold_batch_norms",
            "fold_old_batch_norms"
        ]

        with tf.gfile.GFile(frozen_model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        transformed_graph_def = TransformGraph(
            graph_def, input_ops, output_ops,
            transforms)

        return transformed_graph_def

    def clear_tf_session(self):
        """Clear the tensorflow graph/session. Used to avoid OOM issues related to
        having numerous models."""

        tf.keras.backend.clear_session()
        gc.collect()

        cfg = ConfigProto()
        cfg.gpu_options.allow_growth = True  # pylint: disable=no-member
        tf.keras.backend.set_session(Session(config=cfg))
