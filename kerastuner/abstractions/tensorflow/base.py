import tensorflow
from tensorflow import python
from tensorflow.python import Session, Graph
from tensorflow.python import tools
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.saved_model import save
import subprocess
import sys


from kerastuner.abstractions.tensorflow import proxy

class Tensorflow(proxy.TensorflowProxy):
    def compute_model_size(self, model):
        "Compute the size of a given model"
        params = [K.count_params(p) for p in set(model.trainable_weights)]
        return int(np.sum(params))

    def clear_tf_session(self):
        """Clear the tensorflow graph/session. Used to avoid OOM issues related to
        having numerous models."""

        K.clear_session()
        gc.collect()

        cfg = ConfigProto()
        cfg.gpu_options.allow_growth = True  # pylint: disable=no-member
        K.set_session(Session(config=cfg))


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
        if isinstance(loss, six.string_types):
            return loss
        elif isinstance(loss, list):
            loss_out = []
            for l in loss:
                loss_out.append(serialize_loss(l))
            return loss_out
        elif isinstance(loss, dict):
            loss_out = {}
            for k, v in loss.items():
                loss_out[k] = serialize_loss(l)
            return loss_out
        else:
            return tf.keras.losses.serialize(loss)


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
                loss_out[output] = tf.keras.losses.deserialize_loss(l)
            return loss_out
        elif isinstance(loss, list):
            loss_out = []
            for l in loss:
                loss_out.append(tf.keras.losses.deserialize_loss(l))
            return loss_out
        else:
            return tf.keras.losses.deserialize(loss)



    def freeze_graph(self, saved_model_path, output_graph_path, output_tensor_names):

        # Freeze the temporary graph into the final output file.
        #
        # Note: This needs to be done in an empty session, otherwise names from
        # the loaded model (e.g. Adam/...) will conflict with the graph that is
        # inside of freeze_graph


        command = [
            sys.executable,
            "-m",
            "tensorflow.python.tools.freeze_graph",
            "--input_saved_model_dir=%s" % saved_model_path,
            "--output_node_names=%s" % output_tensor_names,
            "--output_graph=%s" % output_graph_path]

        print("Freezing graph with:", " ".join(command))
        process = subprocess.Popen(command)


        process.wait()
        print("Done.")

        # with Session().as_default() as sess:
        #     with Graph().as_default() as _:
        #         freeze_graph.freeze_graph(
        #             None,  # No input graph, read from saved model.
        #             None,  # No input saver
        #             True,  # Input is binary
        #             "",  # No input checkpoint
        #             # Name of the output tensor for the graph.
        #             output_tensor_names,
        #             "",  # No restore op
        #             "",  # No filename tensor
        #             path,  # Output file for the frozen model.
        #             True,  # Clear devices
        #             "",  # No init nodes
        #             "",  # No var whitelist
        #             "",  # No var blacklist
        #             "",  # No input meta graph
        #             saved_model_path)  # Saved model path

    def optimize_graph(self, frozen_model_path, input_ops, output_ops, input_types, toco_compatible):

        # Parse the GraphDef, and determine the inputs and outputs
        with tf.gfile.GFile(frozen_model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        transformed_graph_def = optimize_for_inference_lib.optimize_for_inference(
            graph_def,
            input_ops,
            output_ops,
            input_types,
            toco_compatible)

        self.write_graph(transformed_graph_def, filename, as_text=False)

    def convert_to_tflite(self,savedmodel_path, output_path, post_training_quantize=True):
        converter = tf.contrib.lite.TFLiteConverter.from_saved_model(
            savedmodel_path)
        converter.post_training_quantize = post_training_quantize
        write_file(path, converter.convert())

    def reload_model(self, config_file, weights_file, results_file, compile=False):
        """ Reconstructs a model from the persisted files.

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
        return [node.op.name for node in model.inputs]

    def get_input_types(self, model):
        return [node.op.dtype for node in model.inputs]

    def get_output_ops(self, model):
        if isinstance(model.output, list):
            return [x.op.name for x in model.output]
        else:
            return [model.output.op.name]

    def write_graph(graph_def, filename, as_text=True):
        tf.train.write_graph(graph_def,
                         logdir=os.path.dirname(filename),
                         as_text=True,
                         name=os.path.basename(filename))
