import os

import six
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph

from kerastuner.abstractions.display import warning
from kerastuner.abstractions.io import read_file, write_file, copy_file, rm


def serialize_loss(loss):
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


def deserialize_loss(loss):
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


def save_keras_model(model, path, tmp_path):
    config_path = "%s-config.json" % path
    weights_path = "%s-weights.h5" % path
    weights_tmp = "%s-weights.h5" % tmp_path

    write_file(config_path, model.to_json())

    # Save to the tmp directory, which is assumed to be local. The final
    # directory may be, e.g. a gs:// URL, which h5py cannot handle.
    model.save_weights(weights_tmp, overwrite=True)

    # Move the file, potentially across filesystems.
    copy_file(weights_tmp, weights_path, overwrite=True)
    rm(weights_tmp)


def save_keras_bundle_model(model, path, tmp_path):
    model.save(tmp_path)
    copy_file(tmp_path, path, overwrite=True)
    rm(tmp_path)


def save_savedmodel(model, path, tmp_path):
    # Build the signature, which maps the user-specified names to the actual tensors
    # in the graph.
    input_dict = {}
    output_dict = {}
    for input_layer in model.inputs:
        input_dict[input_layer.name] = input_layer
    for output_layer in model.outputs:
        output_dict[output_layer.name] = output_layer
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs=input_dict, outputs=output_dict)

    # Using the existing Keras session (so we have access to the entire graph),
    # save the MetaGraphDef containing the signature, proper tags, and the underlying
    # GraphDef.
    with K.get_session().as_default() as sess:
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
            })
        builder.save()


def _get_input_ops(model):
    return [node.op.name for node in model.inputs]


def _get_output_ops(model):
    if isinstance(model.output, list):
        return [x.op.name for x in model.output]
    else:
        return [model.output.op.name]


def save_frozenmodel(model, path, tmp_path):
    # First, create a SavedModel in the tmp directory.
    saved_model_path = tmp_path + "savedmodel"
    saved_model_tmp_path = tmp_path + "savedmodel_tmp"
    save_savedmodel(model, saved_model_path, saved_model_tmp_path)

    # Extract the output tensor names, which are needed in the freeze_graph
    # call to determine which nodes are actually needed in the final graph.
    output_tensor_names = ','.join(_get_output_ops(model))

    # Freeze the temporary graph into the final output file.
    #
    # Note: This needs to be done in an empty session, otherwise names from
    # the loaded model (e.g. Adam/...) will conflict with the graph that is
    # inside of freeze_graph
    with tf.Session().as_default() as sess:
        with tf.Graph().as_default() as _:
            freeze_graph(
                None,  # No input graph, read from saved model.
                None,  # No input saver
                True,  # Input is binary
                "",  # No input checkpoint
                # Name of the output tensor for the graph.
                output_tensor_names,
                "",  # No restore op
                "",  # No filename tensor
                path,  # Output file for the frozen model.
                True,  # Clear devices
                "",  # No init nodes
                "",  # No var whitelist
                "",  # No var blacklist
                "",  # No input meta graph
                saved_model_path)  # Saved model path


def save_minimizedmodel(model, filename, tmp_path, graph_transforms=None):
    # A minimized model requires we first create the frozen model, in the
    # temporary directory.
    frozen_path = tmp_path + "_frozen"
    frozen_tmp_path = tmp_path + "_frozen_tmp"
    save_frozenmodel(model, frozen_path, frozen_tmp_path)

    # Default graph transforms that are usually innocuous. Allow users to
    # override the transformations if desired.
    if graph_transforms is None:
        graph_transforms = [
            "strip_unused_nodes",
            "fold_constants",
            "fold_batch_norms",
            "fold_old_batch_norms"
        ]

    # Parse the GraphDef, and determine the inputs and outputs
    with tf.gfile.GFile(frozen_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    input_ops = _get_input_ops(model)
    output_ops = _get_output_ops(model)

    # Transform the graph (e.g. removing unused nodes, etc.) and write the results.
    transformed_graph_def = TransformGraph(graph_def,
                                           input_ops,
                                           output_ops,
                                           graph_transforms)
    tf.train.write_graph(transformed_graph_def,
                         logdir=os.path.dirname(filename),
                         as_text=False,
                         name=os.path.basename(filename))


def save_tflite(model, path, tmp_path, post_training_quantize=True):
    # First, create a SavedModel in the temporary directory
    savedmodel_path = tmp_path + "savedmodel"
    savedmodel_tmp_path = tmp_path + "savedmodel_tmp"
    save_savedmodel(model, savedmodel_path, savedmodel_tmp_path)

    # Convert the saved model to TF Lite, with quantization.
    converter = tf.contrib.lite.TFLiteConverter.from_saved_model(
        savedmodel_path)
    converter.post_training_quantize = post_training_quantize
    write_file(path, converter.convert())


def save_model(model, path, output_type="keras", tmp_path="/tmp/", **kwargs):
    """
    A model saver object capable of saving a model in multiple formats.

    Args:
        model (tf.keras.models.Model): The model to be saved.
        output_type (str, optional): Defaults to "keras". The format in
            which to save the model. Valid options are:

            "keras" - Save as separate config (JSON) and weights (HDF5)
                files.
            "keras_bundle" - Saved in Keras's native format (HDF5), via
                save_model()
            "tf" - Saved in tensorflow's SavedModel format. See:
                https://www.tensorflow.org/alpha/guide/saved_model
            "tf_frozen" - A SavedModel, where the weights are stored
                in the model file itself, rather than a variables
                directory. See:
                https://www.tensorflow.org/guide/extend/model_files
            "tf_optimized" - A frozen SavedModel, which has
                additionally been transformed via tensorflow's graph
                transform library to remove training-specific nodes and
                operations.  See:
                https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms
            "tf_lite" - A TF Lite model.
    """

    KNOWN_OUTPUT_TYPES = [
        "keras",
        "keras_bundle",
        "tf",
        "tf_frozen",
        "tf_optimized",
        "tf_lite"]

    # Convert PosixPath to string, if necessary.
    path = str(path)
    tmp_path = str(tmp_path)

    if output_type == "keras":
        save_keras_model(model, path, tmp_path)
    elif output_type == "keras_bundle":
        save_keras_bundle_model(model, path, tmp_path)
    elif output_type == "tf":
        save_savedmodel(model, path, tmp_path)
    elif output_type == "tf_frozen":
        save_frozenmodel(model, path, tmp_path)
    elif output_type == "tf_optimized":
        save_minimizedmodel(model, path, tmp_path)
    elif output_type == "tf_lite":
        save_tflite(model, path, tmp_path)
    else:
        raise ValueError("Output type '%s' not in known types '%s'" % (
            output_type, str(KNOWN_OUTPUT_TYPES)))


def reload_model(config_file, weights_file, results_file, compile=False):
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
    # FIXME: merge with instance serialization
    if compile:
        results_file = json.loads(read_file(results_file))
        loss = deserialize_loss(results_file["loss_config"])
        optimizer = tf.keras.optimizers.deserialize(
            results_file["optimizer_config"])
        model.compile(loss=loss, optimizer=optimizer)

    return model
