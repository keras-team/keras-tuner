from abc import abstractmethod
import gc
import json
import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
from tensorflow import python as tf_python

from abc import abstractmethod


class GFileProxy(object):
    """Provides a subset of the tensorflow-2.0 API and proxies the
    calls through to the appropriate tensorflow API."""

    @abstractmethod
    def Open(self, name, mode):
        """Open a file.

        Args:
            name (str): name of the file
            mode (str): one of 'r', 'w', 'a', 'r+', 'w+', 'a+'. Append 'b' for
                bytes mode.

        Returns:
            GFile - a GFile object representing the opened file.
        """

    @abstractmethod
    def makedirs(self, path):
        """Creates a directory and all parent/intermediate directories.

        It succeeds if path already exists and is writable.

        Args:
            path (str): string, name of the directory to be created

        Raises:
            errors.OpError: If the operation fails.
        """

    @abstractmethod
    def exists(self, path):
        """Determines whether a path exists or not.
        Args:
            path: string, a path

        Returns:
            Boolean: True if the path exists, whether it's a file or a
               directory. False if the path does not exist and there are no
               filesystem errors.

        Raises:
            errors.OpError: Propagates any errors reported by the FileSystem API.
        """

    @abstractmethod
    def rmtree(self, path):
        """Deletes everything under path recursively.

        Args:
            path: string, a path

        Raises:
            errors.OpError: If the operation fails.
        """

    @abstractmethod
    def glob(self, pattern):
        """Returns a list of files that match the given pattern(s).

        Args:
            pattern: string or iterable of strings. The glob pattern(s).

        Returns:
            A list of strings containing filenames that match the given pattern(s).

        Raises:
            errors.OpError: If there are filesystem / directory listing errors.
        """

    @abstractmethod
    def remove(self, path):
        """Deletes the path located at 'path'.

        Args:
            path: string, a path

        Raises:
            errors.OpError: Propagates any errors reported by the FileSystem API.
        """

    @abstractmethod
    def copy(self, src, dst, overwrite=False):
        """Copies data from src to dst.

        Args:
            src: string, name of the file whose contents need to be copied
            dst: string, name of the file to which to copy to
            overwrite: boolean, if false its an error for newpath to be occupied
                by an existing file.

        Raises:
            errors.OpError: If the operation fails.
        """


class IOProxy(object):
    """Proxy for tf.io. In general, the GFile module is redirected to a
    custom proxy object, and all other calls are directly passed to `tf.io`"""

    def __init__(self):
        self.gfile = GFileProxy()

    def __getattr__(self, name):
        return getattr(tf.io, name)


class PythonProxy(object):
    """Proxy for tf.python. In general, the calls are passed directly to
    tf.python. Howerver, items in tf.python which exist in only one of tf 1.x
    or tf 2.x may have be forwarded to the appropriate module.
    """

    def __getattr__(self, name):
        return getattr(tf.python, name)


class TensorflowProxy(object):
    """A proxy object for tensorflow calls.

    This proxy system aims to simplify the use of tensorflow across different
    versions(1.11-1.14, 2.x) despite the fact that the APIs have drastically
    changed.
    """

    def __init__(self):
        self.io = IOProxy()
        self.python = tf_python

    def __getattr__(self, name):
        return getattr(tf, name)


class UtilsBase(object):
    """Base implementation of tensorflow utilities."""

    def __init__(self, tf_proxy):
        """Construct a utils class with the given tensorflow proxy."""
        self.tf_proxy = tf_proxy

    def compute_model_size(self, model):
        "Compute the size of a given model, in terms of the number of parameters."
        params = [self.tf_proxy.keras.backend.count_params(
            p) for p in set(model.trainable_weights)]
        return int(np.sum(params))

    def serialize_loss(self, loss):
        """Serialize the loss information for a model.

        Example:
            `json_loss_config = json.dumps(serialize_loss(model.loss))`

        Args:
            loss - One of the following:
                (str): Name of one of the loss functions known to Keras.
                (Callable): A function or callable object. Must be registered as a
                Keras loss.
                (dict): A dictionary mapping output nodes to loss functions in
                    string or callable form. Loss functions must be represented
                    as a str or Callable, as above.
                (list): A list of loss functions, applied to the output nodes.
                    Loss functions must be represented as a str or Callable,
                    as above.
        """
        if isinstance(loss, str):
            return loss
        elif isinstance(loss, list):
            loss_out = []
            for l in loss:
                loss_out.append(self.serialize_loss(l))
            return loss_out
        elif isinstance(loss, dict):
            loss_out = {}
            for k, v in loss.items():
                loss_out[k] = self.serialize_loss(v)
            return loss_out
        else:
            return self.tf_proxy.keras.losses.serialize(loss)

    def deserialize_loss(self, loss):
        """ Deserialize a model loss, serialized by serialize_loss, above,
            returning a single loss function, list of losses, or dict of
            lossess, depending on what was serialized.

            Args:
                loss: JSON configuration representing the loss or losses.
        """

        if isinstance(loss, dict):
            loss_out = {}
            for output, l in loss.items():
                loss_out[output] = self.tf_proxy.keras.losses.deserialize_loss(
                    l)
            return loss_out
        elif isinstance(loss, list):
            loss_out = []
            for l in loss:
                loss_out.append(self.tf_proxy.keras.losses.deserialize_loss(l))
            return loss_out
        else:
            return self.tf_proxy.keras.losses.deserialize(loss)

    def freeze_graph(
            self,
            saved_model_path,
            output_graph_path,
            output_tensor_names):
        """
        Create a frozen version of the specified saved model.

        This cannot be done in the main session, as the names will conflict
        with the existing model. Rather than resorting to manipulating
        sessions in a error-prone and messy fashion, we simply spawn a
        subprocess to freeze the graph.

        Args:
            saved_model_path: str, path to the saved model directory.
            output_graph_path: str, path to the file where the output graph
                will be written.
            output_tensor_names: str, comma separated list of tensors to set
                as the outputs for the graph.
        """

        command = [
            sys.executable,
            "-m",
            "tensorflow.python.tools.freeze_graph",
            "--input_saved_model_dir=%s" % saved_model_path,
            "--output_node_names=%s" % output_tensor_names,
            "--placeholder_type_enum=3",  # int32
            "--output_graph=%s" % output_graph_path]

        process = subprocess.Popen(command)
        process.wait()

    def reload_model(
            self,
            config_file,
            weights_file,
            results_file,
            compile=False):
        """ Reconstructs a model from the outputs of a kerastuner run.

        Args:
            config_file(string): Configuration filename.
            weights_file(string): Keras weights filename.
            results_file(string): Results filename.
            compile(bool, optional): Defaults to False. If True, the optimizer
                and loss will be read from the Instance, and the model will be
                compiled.

        Returns:
            tf.keras.models.Model: The(optionally compiled) Model.
        """

        # Reconstruct the model.
        config = self.read_file(config_file)
        model = self.tf_proxy.keras.models.model_from_json(config)
        model.load_weights(weights_file)

        # If compilation is requested, we need to reload the results file to find
        # which optimizer and losses the model used.
        if compile:
            results_file = json.loads(self.read_file(results_file))
            loss = self.deserialize_loss(results_file["loss_config"])
            optimizer = self.tf_proxy.keras.optimizers.deserialize(
                results_file["optimizer_config"])
            model.compile(loss=loss, optimizer=optimizer)

        return model

    def get_input_ops(self, model):
        "Get the names of the input operations for the given model."
        return [node.op.name for node in model.inputs]

    def get_input_tensors(self, model):
        "Get the names of the input operations for the given model."
        return [node.op for node in model.inputs]

    def get_output_tensors(self, model):
        "Get the names of the output operations for the given model."
        return [node.op for node in model.outputs]

    def get_input_types(self, model):
        "Get the dtypes of the input operations for the given model."
        outputs = []
        for node in model.inputs:
            if hasattr(node, "dtype"):
                outputs.append(node.dtype)
            elif hasattr(node.op, "dtype"):
                outputs.append(node.op.dtype)

        return outputs

    def get_output_ops(self, model):
        "Get the names of the output operations for the given model."
        if isinstance(model.output, list):
            return [x.op.name for x in model.output]
        else:
            return [model.output.op.name]

    def write_graph(self, graph_def, filename, as_text=True):
        "Write the given graph_def to the specified file."
        self.tf_proxy.io.write_graph(graph_def,
                                     logdir=os.path.dirname(filename),
                                     name=os.path.basename(filename),
                                     as_text=as_text)

    def load_savedmodel(
            self,
            session,
            export_dir,
            tags=None):
        """Load a SavedModel into the given session."""
        return self.tf_proxy.saved_model.load(
            session,
            tags,
            export_dir)

    def save_savedmodel(self, model, path, tmp_path):
        """Write a SavedModel to the given path."""
        # Implemented by subclasses
        raise NotImplementedError

    def save_keras_model(self, model, path, tmp_path):
        """Save the given model as separate config and weights files."""
        config_path = "%s-config.json" % path
        weights_path = "%s-weights.h5" % path
        weights_tmp = "%s-weights.h5" % tmp_path

        self.write_file(config_path, model.to_json())
        model.save_weights(weights_tmp, overwrite=True)

        # Move the file, potentially across filesystems.
        self.tf_proxy.io.gfile.copy(
            weights_tmp, weights_path, overwrite=True)

        if self.tf_proxy.io.gfile.exists(weights_tmp):
            self.tf_proxy.io.gfile.remove(weights_tmp)

    def save_keras_bundle_model(self, model, path, tmp_path):
        """Save a Keras model bundle (config + weights) to the given path."""
        model.save(tmp_path)
        self.tf_proxy.io.gfile.copy(tmp_path, path, overwrite=True)
        self.tf_proxy.io.gfile.remove(tmp_path)

    def save_frozenmodel(self, model, path, tmp_path):
        """Save a frozen SavedModel to the given path."""

        # First, create a SavedModel in the tmp directory.
        saved_model_path = tmp_path + "savedmodel"
        saved_model_tmp_path = tmp_path + "savedmodel_tmp"
        self.save_savedmodel(
            model, saved_model_path, saved_model_tmp_path)

        # Extract the output tensor names, which are needed in the freeze_graph
        # call to determine which nodes are actually needed in the final graph.
        ops = self.get_output_ops(model)
        output_tensor_names = ','.join(ops)

        self.freeze_graph(
            saved_model_path, path, output_tensor_names)

    def save_optimized_model(
            self,
            model,
            output_path,
            tmp_path,
            toco_compatible=False):
        """Save an optimized, frozen SavedModel to the given path."""

        # To save an optimized model, we first freeze the model, then apply
        # the optimize_for_inference library.
        frozen_path = tmp_path + "_frozen"
        frozen_tmp_path = tmp_path + "_frozen_tmp"
        self.save_frozenmodel(model, frozen_path, frozen_tmp_path)

        # Parse the GraphDef, and determine the inputs and outputs
        with self.tf_proxy.io.gfile.Open(frozen_path, "rb") as f:
            graph_def = self.tf_proxy.python.GraphDef()
            graph_def.ParseFromString(f.read())
        input_ops = self.get_input_ops(model)
        input_types = self.get_input_types(model)
        output_ops = self.get_output_ops(model)

        transformed_graph_def = self.optimize_graph(
            frozen_path, input_ops, output_ops, input_types, toco_compatible)

        self.write_graph(
            transformed_graph_def,
            os.path.join(output_path, "optimized_graph.pb"),
            as_text=False)

    @abstractmethod
    def save_model(self, model, path, output_type="keras", tmp_path="/tmp/", **kwargs):
        """Save the provided model to the given path.

        Args:
            model(Model): the Keras model to be saved.
            path (str): the directory in which to write the model.
            output_type (str, optional): Defaults to "keras". What format
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
            tmp_path (str, optional): directory in which to store temporary files.
        """

    def write_file(self, path, contents):
        with self.tf_proxy.io.gfile.Open(path, 'w') as output:
            output.write(contents)

    def read_file(self, path, mode='r'):
        with self.tf_proxy.io.gfile.Open(path, mode) as i:
            return i.read()

    def create_directory(self, path, remove_existing=False):
        # Create the directory if it doesn't exist.
        if not self.tf_proxy.io.gfile.exists(path):
            self.tf_proxy.io.gfile.makedirs(path)

        # If it does exist, and remove_existing it specified, the directory will be
        # removed and recreated.
        elif remove_existing:
            self.tf_proxy.io.gfile.rmtree(path)
            self.tf_proxy.io.gfile.makedirs(path)

    def optimize_graph(
            self,
            frozen_model_path,
            input_ops,
            output_ops,
            input_types,
            toco_compatible):
        # Implemented by subclasses
        raise NotImplementedError
