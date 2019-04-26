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
        return getattr(tf.io, name)

class PythonProxy(object):
    def __getattr__(self, name):
        return getattr(tf.python, name)


class TensorflowProxy(object):
    """A proxy object for tensorflow calls.

    This proxy system aims to simplify the use of tensorflow across different
    versions (1.11-1.14, 2.x) despite the fact that the APIs have drastically
    changed.
    """

    def __init__(self):
        self.io = IOProxy()
        self.python = PythonProxy()

    def compute_model_size(self, model):
        "Compute the size of a given model"
        pass

    def clear_tf_session(self):
        """Clear the tensorflow graph/session. Used to avoid OOM issues related to
        having numerous models."""
        pass

    def serialize_optimizer(self, optimizer):
        """Serialize an optimizer."""


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
        pass

    def deserialize_loss(self, loss):
        """ Deserialize a model loss, serialized by serialize_loss, above,
            returning a single loss function, list of losses, or dict of
            lossess, depending on what was serialized.

            Args:
                loss: JSON configuration representing the loss or losses.
        """
        pass


    def save_savedmodel(model, path, tmp_path):
        pass

    def freeze_graph(self, saved_graph_path, output_graph_path, output_node_names):
        "Freeze a saved model with the specified output nodes."
        pass

    def optimize_graph(self, frozen_model_path, input_ops, output_ops, input_types, toco_compatible):
        """Optimize the graph for inference, removing unnecessary nodes,
        folding constants, etc. to reduce the size of the model and
        increase the inference throughput."""
        pass

    def convert_to_tflite(self,savedmodel_path, output_path, post_training_quantize=True):
        """Convert a SavedModel file to tflite format."""

    def reload_model(config_file, weights_file, results_file, compile=False):
        """ Reconstructs a model from the files persisted by the tuner.

        Args:
            config_file (string): Configuration filename.
            weights_file (string): Keras weights filename.
            results_file (string): Results filename.
            compile (bool, optional): Defaults to False. If True, the optimizer
                and loss will be read from the Instance, and the model will be
                compiled.

        Returns:
            tf.keras.models.Model: The (optionally compiled) Model.
        """

        # Reconstruct the model.
        config = read_file(config_file)
        model = tf.keras.models.model_from_json(config)
        model.load_weights(weights_file)

        # If compilation is requested, we need to reload the results file to find
        # which optimizer and losses the model used.
        if compile:
            results_file = json.loads(read_file(results_file))
            loss = deserialize_loss(results_file["loss_config"])
            optimizer = tf.keras.optimizers.deserialize(
                results_file["optimizer_config"])
            model.compile(loss=loss, optimizer=optimizer)

        return model

    def get_input_ops(self, model):
        """Return the input op names for the model."""
        pass

    def get_input_types(self, model):
        """Return the input types for the model"""
        pass

    def get_output_ops(self, model):
        """Return the output op names for the model, as a list."""
