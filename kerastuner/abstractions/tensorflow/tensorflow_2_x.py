# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
import os
import subprocess
import sys

import tensorflow as tf
from tensorflow.compat.v1.io import gfile
from tensorflow.python import ConfigProto, GraphDef, Session
from tensorflow.keras.models import model_from_json, load_model
from kerastuner.abstractions.display import write_log
from kerastuner.abstractions.tensorflow import proxy
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel


class GFileProxy_2_x(proxy.GFileProxy):
    """Proxies calls through to the appropriate tensorflow API."""

    def Open(self, name, mode):
        """Open a file.

        Args:
            name (str): name of the file
            mode (str): one of 'r', 'w', 'a', 'r+', 'w+', 'a+'.
                Append 'b' for bytes mode.

        Returns:
            GFile - a GFile object representing the opened file.
        """
        return tf.compat.v1.io.gfile.GFile(name, mode)

    def makedirs(self, path):
        """Creates a directory and all parent/intermediate directories.

        It succeeds if path already exists and is writable.

        Args:
            path (str): string, name of the directory to be created

        Raises:
            errors.OpError: If the operation fails.
        """
        return tf.compat.v1.io.gfile.makedirs(path)

    def exists(self, path):
        """Determines whether a path exists or not.
        Args:
            path: string, a path

        Returns:
            True if the path exists, whether it's a file or a directory.
            False if the path does not exist and there
                are no filesystem errors.

        Raises:
            errors.OpError: Propagates any errors reported
                by the FileSystem API.
        """
        return tf.compat.v1.io.gfile.exists(path)

    def rmtree(self, path):
        """Deletes everything under path recursively.

        Args:
        path: string, a path

        Raises:
        errors.OpError: If the operation fails.
        """
        return tf.compat.v1.io.gfile.rmtree(path)

    def glob(self, pattern):
        """Returns a list of files that match the given pattern(s).

        Args:
        pattern: string or iterable of strings. The glob pattern(s).

        Returns:
        A list of strings containing filenames that match the given pattern(s).

        Raises:
        errors.OpError: If there are filesystem / directory listing errors.
        """
        return tf.compat.v1.io.gfile.glob(pattern)

    def remove(self, path):
        """Deletes the path located at 'path'.

        Args:
        path: string, a path

        Raises:
        errors.OpError: Propagates any errors reported by the FileSystem API.
            E.g., NotFoundError if the path does not exist.
        """
        return tf.compat.v1.io.gfile.remove(path)

    def copy(self, src, dst, overwrite=False):
        """Copies data from src to dst.

        Args:
        src: string, name of the file whose contents need to be copied
        dst: string, name of the file to which to copy to
        overwrite: boolean, if false its an error for newpath
            to be occupied by an existing file.

        Raises:
        errors.OpError: If the operation fails.
        """
        return tf.compat.v1.io.gfile.copy(src, dst, overwrite=overwrite)

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
        write_log("Saving weights to %s" % weights_tmp)
        model.save_weights(weights_tmp, overwrite=True)

        # Move the file, potentially across filesystems.
        tf.compat.v1.io.gfile.copy(weights_tmp, weights_path, overwrite=True)
        write_log("Moving weights to %s" % weights_path)
        tf.compat.v1.io.gfile.remove(weights_tmp)

    def save_keras_bundle_model(self, model, path, tmp_path):
        path += ".keras_bundle.h5"
        model.save(tmp_path)
        tf.compat.v1.io.gfile.copy(tmp_path, path, overwrite=True)
        tf.compat.v1.io.gfile.remove(tmp_path)

    def save_tflite(self, model, path, tmp_path, post_training_quantize=True):
        # First, create a SavedModel in the temporary directory
        keras_bundle_path = tmp_path + "keras_bundle.h5"
        tmp_keras_bundle_path = tmp_path + "keras_bundle_tmp"

        self.save_keras_bundle_model(
            model, keras_bundle_path, tmp_keras_bundle_path)
        self.convert_to_tflite(
            model, keras_bundle_path, path,
            post_training_quantize=post_training_quantize)

    def convert_to_tflite(self,
                          model,
                          savedmodel_path,
                          output_path,
                          post_training_quantize):
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

    def optimize_graph(self,
                       frozen_model_path,
                       input_ops,
                       output_ops,
                       input_types,
                       toco_compatible):
        # Parse the GraphDef, and determine the inputs and outputs
        with tf.compat.v1.io.gfile.GFile(frozen_model_path, "rb") as f:
            graph_def = GraphDef()
            graph_def.ParseFromString(f.read())

        # Convert the tensorflow dtypes into the enumeration values,
        # as expected by optimize_for_inference
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
        tf.keras.backend.set_session(Session(config=cfg))

    def _get_output_tensor_names_from_savedmodel(self, model, saved_model_path):
        """ Looks in the default_serving signature def of the saved model to
        determine the output tensor names for the given model.
        """
        saved_model_pb_file = os.path.join(saved_model_path, "saved_model.pb")

        with tf.compat.v1.io.gfile.GFile(saved_model_pb_file, "rb") as f:
            graph_bytes = f.read()

        sm = SavedModel()
        sm.ParseFromString(graph_bytes)

        name_map = {}

        for meta_graph in sm.meta_graphs:
            sig_def = meta_graph.signature_def["serving_default"]
            for name, tensor in sig_def.outputs.items():
                tensor_name = tensor.name
                # Drop the :0 suffix.
                tensor_name = tensor_name.split(":")[0]
                name_map[name] = tensor_name

        outputs = []
        for output_name in model.output_names:
            outputs.append(name_map[output_name])
        return outputs

    def save_frozenmodel(self, model, path, tmp_path):
        """Save a frozen SavedModel to the given path."""

        # First, create a SavedModel in the tmp directory.
        saved_model_path = tmp_path + "savedmodel"
        saved_model_tmp_path = tmp_path + "savedmodel_tmp"
        self.save_savedmodel(
            model, saved_model_path, saved_model_tmp_path)

        outputs = self._get_output_tensor_names_from_savedmodel(
            model, saved_model_path)
        output_tensor_names = ','.join(outputs)

        self.freeze_graph(
            saved_model_path, path, output_tensor_names)

    def save_model(self, model, path, export_type="keras", tmp_path="/tmp/"):
        KNOWN_OUTPUT_TYPES = [
            "keras",
            "keras_bundle",
            "tf",
            "tf_frozen"
        ]

        # Not yet supported for tf 2.0 - numerous issues with GPU models, and
        # other issues we haven't debugged yet.
        UNSUPPORTED_OUTPUT_TYPES = [
            "tf_lite",
            "tf_optimized",
        ]

        if export_type in UNSUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                "Output type '%s' is not currently supported "
                "when using tensorflow 2.x.  Valid output types are: %s" % (
                    export_type, str(KNOWN_OUTPUT_TYPES)))

        # Convert PosixPath to string, if necessary.
        path = str(path)
        tmp_path = str(tmp_path)

        if export_type == "keras":
            self.save_keras_model(model, path, tmp_path)
        elif export_type == "keras_bundle":
            self.save_keras_bundle_model(model, path, tmp_path)
        elif export_type == "tf":
            self.save_savedmodel(model, path, tmp_path)
        elif export_type == "tf_frozen":
            self.save_frozenmodel(model, path, tmp_path)
        elif export_type == "tf_optimized":
            self.save_optimized_model(model, path, tmp_path)
        elif export_type == "tf_lite":
            self.save_tflite(model, path, tmp_path)
        else:
            raise ValueError("Output type '%s' not in known types '%s'" % (
                export_type, str(KNOWN_OUTPUT_TYPES)))
