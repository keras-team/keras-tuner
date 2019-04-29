import json
import os

import numpy as np
import pytest

from kerastuner.abstractions.tensorflow import MAJOR_VERSION, MINOR_VERSION
from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils

K = tf.keras.backend


_TRAIN_ARR_1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
_TRAIN_ARR_2 = np.array([5, 4, 3, 2, 1], dtype=np.float32)
_OUT_ARR_1 = _TRAIN_ARR_1 - _TRAIN_ARR_2
_OUT_ARR_2 = _TRAIN_ARR_1 * _TRAIN_ARR_2


@pytest.fixture(autouse=True)
def clear_session():
    K.clear_session()


@pytest.fixture(scope="function")
def model():
    i1 = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="i1")
    i2 = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="i2")

    t1 = tf.keras.layers.Dense(4, name="dense_i1")(i1)
    t2 = tf.keras.layers.Dense(4, name="dense_i2")(i2)

    c = tf.keras.layers.Concatenate()([t1, t2])

    o1 = tf.keras.layers.Dense(1, name="o1")(c)
    o2 = tf.keras.layers.Dense(1, name="o2")(c)

    model = tf.keras.models.Model(inputs=[i1, i2], outputs=[o1, o2])
    model.compile(optimizer="adam", loss={
        "o1": "mse",
        "o2": "mse"
    })

    x = {
        "i1": _TRAIN_ARR_1,
        "i2": _TRAIN_ARR_2
    }
    y = {
        "o1": _OUT_ARR_1,
        "o2": _OUT_ARR_2
    }
    model.fit(x, y, epochs=2)
    return model


@pytest.fixture(scope="module")
def training_data():
    return (
        {
            "i1": _TRAIN_ARR_1,
            "i2": _TRAIN_ARR_2
        },
        {
            "o1": _OUT_ARR_1,
            "o2": _OUT_ARR_2,
        }
    )


@pytest.fixture(scope="module")
def feed_dict():
    return {
        "i1:0": np.expand_dims(_TRAIN_ARR_1, axis=-1),
        "i2:0": np.expand_dims(_TRAIN_ARR_2, axis=-1)
    }


@pytest.fixture(scope="module")
def output_names():
    return [
        "o1/BiasAdd:0",
        "o2/BiasAdd:0"
    ]


def test_save_keras_bundle(tmp_path, model, training_data):
    save_path = os.path.join(str(tmp_path), "model_output")
    tmp_path = os.path.join(str(tmp_path), "model_output_tmp")
    x, y = training_data

    tf_utils.save_model(
        model, save_path, tmp_path=tmp_path, output_type="keras_bundle")

    loaded = tf.keras.models.load_model(save_path)

    orig_out = model.predict(x)
    loaded_out = loaded.predict(x)

    assert np.allclose(orig_out, loaded_out)


def test_save_keras(tmp_path, model, training_data):
    save_path = os.path.join(str(tmp_path), "model_output")
    tmp_path = os.path.join(str(tmp_path), "model_output_tmp")
    x, y = training_data

    tf_utils.save_model(model, save_path, tmp_path=tmp_path, output_type="keras")

    config = tf_utils.read_file(save_path + "-config.json")

    loaded = tf.keras.models.model_from_json(config)
    loaded.load_weights(save_path + "-weights.h5")
    loaded.compile(optimizer="adam", loss={
        "o1": "mse",
        "o2": "mse"
    })

    orig_out = model.predict(x)
    loaded_out = loaded.predict(x)

    assert np.allclose(orig_out, loaded_out)


def test_save_tf(
        tmp_path,
        model,
        training_data,
        feed_dict,
        output_names):

    # Not currently supported for tf2.0
    if MAJOR_VERSION == 2:
      return

    save_path = os.path.join(str(tmp_path), "model_output")
    tmp_path = os.path.join(str(tmp_path), "model_output_tmp")
    x, y = training_data

    tf_utils.save_model(model, save_path, tmp_path=tmp_path, output_type="tf")

    orig_out = model.predict(x)

    if MAJOR_VERSION >= 2:
        sess = None
        loaded_model = tf_utils.load_savedmodel(
            sess,
            export_dir=save_path,
            tags=["serve"])
        loaded_out = loaded_model.predict(x)
        assert np.allclose(orig_out, loaded_out)
    else:
        with tf.python.Session().as_default() as sess:
            loaded_model = tf_utils.load_savedmodel(
                sess,
                export_dir=save_path,
                tags=["serve"])

            node_map = {}
            for node in sess.graph.as_graph_def().node:
                node_map[node.name] = node

            output_node = node_map["o2/BiasAdd"]
            output_node.input[0] == "o2/MatMul"
            output_node.input[1] == "o2/BiasAdd/ReadVariableOp"

            output_node = node_map["o1/BiasAdd"]
            output_node.input[0] == "o1/MatMul"
            output_node.input[1] == "o1/BiasAdd/ReadVariableOp"


def test_save_frozen(
        tmp_path,
        model,
        training_data,
        feed_dict,
        output_names):
    # Not currently supported for tf2.0
    if MAJOR_VERSION == 2:
      return

    save_path = os.path.join(str(tmp_path), "model_output")
    tmp_path = os.path.join(str(tmp_path), "model_output_tmp")
    x, y = training_data

    orig_out = model.predict(x)

    tf_utils.save_model(model, save_path, tmp_path=tmp_path, output_type="tf_frozen")

    tf.keras.backend.clear_session()

    graph = tf.python.Graph()
    sess = tf.python.Session(graph=graph)
    with sess.as_default() as default_sess:
        with sess.graph.as_default() as default_graph:
            graph_def = tf.python.GraphDef()
            graph_def.ParseFromString(tf_utils.read_file(save_path, "rb"))
            tf.import_graph_def(
                graph_def, name="", return_elements=None)

            loaded_out = sess.run(output_names, feed_dict=feed_dict)

    assert np.allclose(orig_out, loaded_out)


def test_save_optimized(
        tmp_path,
        model,
        training_data,
        feed_dict,
        output_names):

    # Not currently supported for tf2.0
    if MAJOR_VERSION == 2:
      return

    save_path = os.path.join(str(tmp_path), "model_output")
    tmp_save_path = os.path.join(str(save_path), "tmp_model_output")

    x, y = training_data
    orig_out = model.predict(x)

    tf_utils.save_model(
        model,
        save_path,
        tmp_path=tmp_save_path,
        output_type="tf_optimized")

    graph = tf.python.Graph()
    sess = tf.python.Session(graph=graph)
    with sess.as_default() as default_sess:
        with sess.graph.as_default() as default_graph:
            graph_def = tf.python.GraphDef()
            graph_def.ParseFromString(
                tf_utils.read_file(os.path.join(
                    save_path, "optimized_graph.pb"), "rb"))
            tf.import_graph_def(
                graph_def, name="", return_elements=None)

            loaded_out = sess.run(output_names, feed_dict=feed_dict)

    assert np.allclose(orig_out, loaded_out)


def test_save_tf_lite(
        tmp_path,
        model,
        training_data,
        feed_dict,
        output_names):

    # Not currently supported for tf2.0
    if MAJOR_VERSION == 2:
      return

    # There are bugs with saving as tf.lite in early version
    # see: https://github.com/tensorflow/tensorflow/issues/17349
    if MAJOR_VERSION == 1 and MINOR_VERSION <= 13:
        return

    save_path = os.path.join(str(tmp_path), "model_output")
    save_file = os.path.join(save_path, "optimized_model.tflite")

    print("Creating:", save_path)
    os.makedirs(save_path)

    x, y = training_data
    orig_out = model.predict(x)

    tf_utils.save_model(model, save_path, tmp_path=tmp_path, output_type="tf_lite")

    with tf.python.Session().as_default() as sess:
        with tf.Graph().as_default() as _:
            interpreter = tf.lite.Interpreter(model_path=save_file)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            for i in range(len(x["i1"])):
                i1 = np.expand_dims(np.expand_dims(
                    x["i1"][i], axis=-1), axis=-1)
                i2 = np.expand_dims(np.expand_dims(
                    x["i2"][i], axis=-1), axis=-1)

                interpreter.set_tensor(
                    input_details[0]['index'], i1)
                interpreter.set_tensor(
                    input_details[1]['index'], i2)
                interpreter.invoke()
                o1 = interpreter.get_tensor(output_details[0]['index'])
                o2 = interpreter.get_tensor(output_details[1]['index'])

                assert np.allclose(orig_out[0][i], o1)
                assert np.allclose(orig_out[1][i], o2)
