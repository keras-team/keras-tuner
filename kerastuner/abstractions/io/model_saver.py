import os

import six

import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow
from tensorflow import python
from tensorflow.python import Session, Graph
from tensorflow.python import tools
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.saved_model import save


from kerastuner.abstractions.display import warning
from kerastuner.abstractions.io import read_file, write_file, copy_file, rm
from kerastuner.abstractions.tensorflow import TENSORFLOW


def save_keras_model(model, path, tmp_path):
    config_path = "%s-config.json" % path
    weights_path = "%s-weights.h5" % path
    weights_tmp = "%s-weights.h5" % tmp_path

    write_file(config_path, model.to_json())
    model.save_weights(weights_tmp, overwrite=True)

    # Move the file, potentially across filesystems.
    copy_file(weights_tmp, weights_path, overwrite=True)
    rm(weights_tmp)


def save_keras_bundle_model(model, path, tmp_path):
    model.save(tmp_path)
    copy_file(tmp_path, path, overwrite=True)
    rm(tmp_path)


def save_savedmodel(model, path, tmp_path):
    TENSORFLOW.save_savedmodel(model, path, tmp_path)


def save_frozenmodel(model, path, tmp_path):
    # First, create a SavedModel in the tmp directory.
    saved_model_path = tmp_path + "savedmodel"
    saved_model_tmp_path = tmp_path + "savedmodel_tmp"
    TENSORFLOW.save_savedmodel(
        model, saved_model_path, saved_model_tmp_path)

    # Extract the output tensor names, which are needed in the freeze_graph
    # call to determine which nodes are actually needed in the final graph.
    ops = TENSORFLOW.get_output_ops(model)
    output_tensor_names = ','.join(ops)

    TENSORFLOW.freeze_graph(
        saved_model_path, path, output_tensor_names)


def save_optimized_model(model, filename, tmp_path, toco_compatible=False):
    # To save an optimized model, we first freeze the model, then apply
    # the optimize_for_inference library.
    frozen_path = tmp_path + "_frozen"
    frozen_tmp_path = tmp_path + "_frozen_tmp"
    save_frozenmodel(model, frozen_path, frozen_tmp_path)

    # Parse the GraphDef, and determine the inputs and outputs
    with TENSORFLOW.io.gfile.Open(frozen_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    input_ops = TENSORFLOW.get_input_ops(model)
    input_types = TENSORFLOW.get_input_types(model)
    output_ops = TENSORFLOW.get_output_ops(model)

    TENSORFLOW.optimize_graph(frozen_path, input_ops, output_ops, input_types, toco_compatible)
    TENSORFLOW.write_graph(transformed_graph_def, filename, as_text=False)

 

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

    print("Saving as %s" % output_type)
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
        save_optimized_model(model, path, tmp_path)
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
    if compile:
        results_file = json.loads(read_file(results_file))
        loss = deserialize_loss(results_file["loss_config"])
        optimizer = tf.keras.optimizers.deserialize(
            results_file["optimizer_config"])
        model.compile(loss=loss, optimizer=optimizer)

    return model
