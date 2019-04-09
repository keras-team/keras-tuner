import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph
from kerastuner.abstractions.io import write_file


class ModelSaver(object):
    def __init__(
            self,
            model,
            output_type="keras"):
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

        self.model = model
        self.output_type = output_type

        KNOWN_OUTPUT_TYPES = [
            "keras", "keras_h5", "tf", "tf_frozen", "tf_optimized", "tf_lite"]
        assert self.output_type in KNOWN_OUTPUT_TYPES, "Output type '%s' not in known types '%s'" % (
            output_type, str(KNOWN_OUTPUT_TYPES))

    def _save_keras_model(self, path):
        config_path = "%s-config.json" % path
        weights_path = "%s-weights.h5" % path

        write_file(config_path, self.model.to_json())
        self.model.save_weights(weights_path, overwrite=True)

    def _save_keras_h5_model(self, path):
        self.model.save(path)

    def _save_savedmodel(self, path):
        input_dict = {}
        output_dict = {}

        for input_layer in self.model.inputs:
            input_dict[input_layer.name] = input_layer
        for output_layer in self.model.outputs:
            output_dict[output_layer.name] = output_layer

        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs=input_dict, outputs=output_dict)

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

    def _get_input_ops(self):
        return [node.op.name for node in self.model.inputs]

    def _get_output_ops(self):
        if isinstance(self.model.output, list):
            return [x.op.name for x in self.model.output]
        else:
            return [self.model.output.op.name]

    def _save_frozenmodel(self, path):
        saved_model_path = path + "_savedmodel"
        self._save_savedmodel(saved_model_path)

        output_tensor_names = ','.join(self._get_output_ops())

        freeze_graph(
            None,  # No input graph, read from saved model.
            None,  # No input saver
            True,  # Input is binary
            "",  # No input checkpoint
            output_tensor_names,  # Name of the output tensor for the graph.
            "",  # No restore op
            "",  # No filename tensor
            path,  # Output file for the frozen model.
            True,  # Clear devices
            "",  # No init nodes
            "",  # No var whitelist
            "",  # No var blacklist
            "",  # No input meta graph
            saved_model_path)  # Saved model path

    def _save_minimizedmodel(self, filename):
        frozen_path = filename + "_frozen"
        self._save_frozenmodel(frozen_path)

        # FIXME - this should be configurable.
        graph_transforms = [
            "strip_unused_nodes",
            "fold_constants",
            "fold_batch_norms",
            "fold_old_batch_norms"
        ]

        with tf.gfile.GFile(frozen_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        input_ops = self._get_input_ops()
        output_ops = self._get_output_ops()

        transformed_graph_def = TransformGraph(graph_def,
                                               input_ops,
                                               output_ops,
                                               graph_transforms)

        tf.train.write_graph(transformed_graph_def,
                             logdir=os.path.dirname(filename),
                             as_text=False,
                             name=os.path.basename(filename))

    def _save_tflite(self, path):
        savedmodel_path = path + "_savedmodel"
        self._save_savedmodel(savedmodel_path)
        print(tf.contrib.lite.TFLiteConverter)
        print(dir(tf.contrib.lite.TFLiteConverter))
        converter = tf.contrib.lite.TFLiteConverter.from_saved_model(
            savedmodel_path)
        converter.post_training_quantize = True
        write_file(path, converter.convert())

    def save_model(self, path):
        if self.output_type == "keras":
            self._save_keras_model(path)
        if self.output_type == "keras_h5":
            self._save_keras_h5_model(path)
        if self.output_type == "tf":
            self._save_savedmodel(path)
        if self.output_type == "tf_frozen":
            self._save_frozenmodel(path)
        if self.output_type == "tf_optimized":
            self._save_minimizedmodel(path)
        if self.output_type == "tf_lite":
            self._save_tflite(path)


def save_model(model, path, output_type):
    saver = ModelSaver(model, output_type)
    saver.save_model(path)
