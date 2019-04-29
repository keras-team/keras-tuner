from tensorflow import gfile
from tensorflow.tools.graph_transforms import TransformGraph

from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import proxy
from kerastuner.abstractions.tensorflow.utils_base import UtilsBase


class Utils_1_x(UtilsBase):
    def load_savedmodel(
            self,
            session,
            export_dir,
            tags=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):

        return tf.saved_model.load(
            session,
            tags,
            export_dir)

    def save_savedmodel(self, model, path, tmp_path):
        session = tf.keras.backend.get_session()
        self.write_graph(session.graph, tmp_path, "graph.pbtxt")
        self.write_graph(tf.get_default_graph(),
                         tmp_path, "graph.pbtxt")
        inputs = dict([(node.op.name, node) for node in model.inputs])
        outputs = dict([(node.op.name, node) for node in model.outputs])
        tf.compat.v1.saved_model.simple_save(session, path, inputs, outputs)

    def save_tflite(self, model, path, tmp_path, post_training_quantize=True):
        # First, create a SavedModel in the temporary directory
        savedmodel_path = os.path.join(tmp_path, "savedmodel")
        savedmodel_tmp_path = os.path.join(tmp_path, "savedmodel_tmp")

        input_ops = self.get_input_ops(model)
        output_ops = self.get_output_ops(model)

        save_savedmodel(model, savedmodel_path, savedmodel_tmp_path)

        self.convert_to_tflite(
            model, savedmodel_path, path, post_training_quantize=post_training_quantize)

    def convert_to_tflite(
            self,
            model,
            savedmodel_path,
            output_path,
            post_training_quantize):
        converter = tf.contrib.lite.TFLiteConverter.from_saved_model(
            savedmodel_path)
        converter.post_training_quantize = post_training_quantize
        write_file(path, converter.convert())

    def optimize_graph(
            self,
            frozen_model_path,
            input_ops,
            output_ops,
            input_types,
            toco_compatible):

        transforms = [
            "fold_constants",
            "fold_batch_norms",
            "fold_old_batch_norms"
        ]

        with tf.gfile.GFile(frozen_model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        transformed_graph_def = TransformGraph(
            graph_def, input_ops, output_ops,
            transforms)

        return transformed_graph_def
