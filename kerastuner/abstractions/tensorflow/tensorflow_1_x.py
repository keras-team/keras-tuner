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
import os

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.python import Session, ConfigProto
from tensorflow.tools.graph_transforms import TransformGraph

from kerastuner.abstractions.tensorflow import proxy


class GFileProxy_1_x(proxy.GFileProxy):
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
        return tf.io.gfile.GFile(name, mode)

    def makedirs(self, path):
        """Creates a directory and all parent/intermediate directories.

        It succeeds if path already exists and is writable.

        Args:
            path (str): string, name of the directory to be created

        Raises:
            errors.OpError: If the operation fails.
        """
        return tf.io.gfile.makedirs(path)

    def exists(self, path):
        """Determines whether a path exists or not.
        Args:
            path: string, a path

        Returns:
            True if the path exists, whether it's a file or a directory.
            False if the path does not exist
                and there are no filesystem errors.

        Raises:
            errors.OpError: Propagates any errors
                reported by the FileSystem API.
        """
        return tf.io.gfile.exists(path)

    def rmtree(self, path):
        """Deletes everything under path recursively.

        Args:
        path: string, a path

        Raises:
        errors.OpError: If the operation fails.
        """
        return tf.io.gfile.rmtree(path)

    def glob(self, pattern):
        """Returns a list of files that match the given pattern(s).

        Args:
        pattern: string or iterable of strings. The glob pattern(s).

        Returns:
        A list of strings containing filenames that match the given pattern(s).

        Raises:
        errors.OpError: If there are filesystem / directory listing errors.
        """
        return tf.io.gfile.glob(pattern)

    def remove(self, path):
        """Deletes the path located at 'path'.

        Args:
        path: string, a path

        Raises:
        errors.OpError: Propagates any errors reported by the FileSystem API.
            NotFoundError if the path does not exist.
        """
        return tf.io.gfile.remove(path)

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
        return tf.io.gfile.copy(src, dst, overwrite=overwrite)

    def __getattr__(self, name):
        return getattr(tf.io.gfile, name)


class IOProxy_1_x(proxy.IOProxy):
    def __init__(self):
        self.gfile = GFileProxy_1_x()


class PythonProxy_1_x(proxy.PythonProxy):
    def __getattr__(self, name):
        raise ValueError("Attribute '%s' not supported" % name)

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
            tags=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
        return tf.saved_model.load(session, tags, export_dir)

    def save_savedmodel(self, model, path, tmp_path):
        session = tf.keras.backend.get_session()
        self.write_graph(session.graph, tmp_path, "graph.pbtxt")
        self.write_graph(tf.get_default_graph(), tmp_path, "graph.pbtxt")
        inputs = dict([(node.op.name, node) for node in model.inputs])
        outputs = dict([(node.op.name, node) for node in model.outputs])
        tf.compat.v1.saved_model.simple_save(session, path, inputs, outputs)

    def save_tflite(self, model, path, tmp_path, post_training_quantize=True):
        sess = K.get_session()

        self.write_file(
            path + ".tflite",
            tf.lite.TFLiteConverter.from_session(sess, model.inputs,
                                                 model.outputs).convert())

    def convert_to_tflite(self, model, savedmodel_path, output_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
        self.write_file(output_path, converter.convert())

    def optimize_graph(self, frozen_model_path, input_ops, output_ops,
                       input_types, toco_compatible):

        transforms = [
            "fold_constants", "fold_batch_norms", "fold_old_batch_norms"
        ]

        with tf.io.gfile.GFile(frozen_model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        transformed_graph_def = TransformGraph(graph_def, input_ops,
                                               output_ops, transforms)

        return transformed_graph_def

    def clear_tf_session(self):
        """Clear the tensorflow graph/session.

        Used to avoid OOM issues related to having numerous models.
        """

        tf.keras.backend.clear_session()
        gc.collect()

        cfg = ConfigProto()
        cfg.gpu_options.allow_growth = True  # pylint: disable=no-member
        tf.keras.backend.set_session(Session(config=cfg))

    def save_model(self,
                   model,
                   path,
                   export_type="keras",
                   tmp_path="/tmp/",
                   **kwargs):
        """Save the provided model to the given path.

        Args:
            model(Model): the Keras model to be saved.
            path (str): the directory in which to write the model.
            export_type (str, optional): Defaults to "keras". What format
                of model to export:

                # Tensorflow 1.x/2.x
                "keras" - Save as separate config (JSON) and weights (HDF5)
                    files.
                "keras_bundle" - Saved in Keras's native format (HDF5), via
                    save_model()

                # Currently only supported in Tensorflow 1.x
                "tf" - Saved in tensorflow's SavedModel format. See:
                    https://www.tensorflow.org/alpha/guide/saved_model
                "tf_frozen" - A SavedModel, where the weights are stored
                    in the model file itself, rather than a variables
                    directory. See:
                    https://www.tensorflow.org/guide/extend/model_files
                "tf_optimized" - A frozen SavedModel, which has
                    additionally been transformed via tensorflow's graph
                    transform library to remove training-specific nodes
                    and operations.  See:
                    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms
                "tf_lite" - A TF Lite model.
            tmp_path (str, optional): directory in which
                to store temporary files.
        """

        KNOWN_OUTPUT_TYPES = [
            "keras", "keras_bundle", "tf", "tf_frozen", "tf_optimized",
            "tf_lite"
        ]

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
            raise ValueError("Output type '%s' not in known types '%s'" %
                             (export_type, str(KNOWN_OUTPUT_TYPES)))
