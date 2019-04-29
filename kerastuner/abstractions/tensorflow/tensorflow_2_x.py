import os
import subprocess
import sys

import tensorflow as tf
from tensorflow.io import gfile
from tensorflow.python import ConfigProto, GraphDef, Session

from kerastuner.abstractions.tensorflow import proxy

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


class Utils_2_x(proxy.UtilsBase):
    def load_savedmodel(
            self,
            session,
            export_dir,
            tags=["serve"]):
        return tf.keras.experimental.load_from_saved_model(export_dir)

    def save_savedmodel(self, model, path, tmp_path):
        return tf.keras.experimental.export_saved_model(model, path)

    def save_keras_model(self, model, path, tmp_path):
        config_path = "%s-config.json" % path
        weights_path = "%s-weights.h5" % path
        weights_tmp = "%s-weights.h5" % tmp_path

        self.write_file(config_path, model.to_json())
        model.save_weights(weights_tmp, overwrite=True)

        # Move the file, potentially across filesystems.
        tf.io.gfile.copy(weights_tmp, weights_path, overwrite=True)
        tf.io.gfile.remove(weights_tmp)

    def save_keras_bundle_model(self, model, path, tmp_path):
        model.save(tmp_path)
        tf.io.gfile.copy(tmp_path, path, overwrite=True)
        tf.io.gfile.remove(tmp_path)

    def save_tflite(self, model, path, tmp_path, post_training_quantize=True):
        # First, create a SavedModel in the temporary directory
        keras_bundle_path = tmp_path + "keras_bundle.h5"
        tmp_keras_bundle_path = tmp_path + "keras_bundle_tmp"

        self.save_keras_bundle_model(
            model, keras_bundle_path, tmp_keras_bundle_path)
        self.convert_to_tflite(
            model, keras_bundle_path, path, post_training_quantize=post_training_quantize)

    def convert_to_tflite(self, model, savedmodel_path, output_path, post_training_quantize):
        output_file = os.path.join(output_path, "optimized_model.tflite")

        command = [
            sys.executable,
            "-m",
            "tensorflow.lite.python.tflite_convert",
            "--keras_model_file=%s" % savedmodel_path,
            "--output_file=%s" % output_file
        ]

        process = subprocess.Popen(command)
        process.wait()

    def optimize_graph(self, frozen_model_path, input_ops, output_ops, input_types, toco_compatible):
        # Parse the GraphDef, and determine the inputs and outputs
        with tf.io.gfile.GFile(frozen_model_path, "rb") as f:
            graph_def = GraphDef()
            graph_def.ParseFromString(f.read())

        # Convert the tensorflow dtypes into the enumeration values, as expected
        # by optimize_for_inference
        input_types = [type.as_datatype_enum for type in input_types]

        transformed_graph_def = optimize_for_inference_lib.optimize_for_inference(
            graph_def,
            input_ops,
            output_ops,
            input_types,
            toco_compatible)

        return transformed_graph_def

    def clear_tf_session(self):
        """Clear the tensorflow graph/session. Used to avoid OOM issues related to
        having numerous models."""

        tf.keras.backend.clear_session()
        gc.collect()

        cfg = ConfigProto()
        cfg.gpu_options.allow_growth = True  # pylint: disable=no-member
        tf.keras.backend.set_session( Session(config=cfg))
