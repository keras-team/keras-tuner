import os
import subprocess
import sys

from tensorflow.python.tools import optimize_for_inference_lib

from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import proxy
from kerastuner.abstractions.tensorflow.utils_base import UtilsBase


class Utils_2_x(UtilsBase):
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
            graph_def = tf.python.GraphDef()
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
