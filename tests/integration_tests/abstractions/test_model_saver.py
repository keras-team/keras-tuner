import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.models import Model, load_model, model_from_json
from kerastuner.abstractions.io import save_model, read_file
import numpy as np
import os
import pytest
import json


_TRAIN_ARR_1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
_TRAIN_ARR_2 = np.array([5, 4, 3, 2, 1], dtype=np.float32)
_OUT_ARR_1 = _TRAIN_ARR_1 - _TRAIN_ARR_2
_OUT_ARR_2 = _TRAIN_ARR_1 * _TRAIN_ARR_2


@pytest.fixture(autouse=True)
def clear_session():
    K.clear_session()


@pytest.fixture(scope="function")
def model():
    i1 = Input(shape=(1,), dtype=tf.float32, name="i1")
    i2 = Input(shape=(1,), dtype=tf.float32, name="i2")

    t1 = Dense(4, name="dense_i1")(i1)
    t2 = Dense(4, name="dense_i2")(i2)

    c = Concatenate()([t1, t2])

    o1 = Dense(1, name="o1")(c)
    o2 = Dense(1, name="o2")(c)

    model = Model(inputs=[i1, i2], outputs=[o1, o2])
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

    save_model(model, save_path, tmp_path=tmp_path, output_type="keras_bundle")

    loaded = load_model(save_path)

    orig_out = model.predict(x)
    loaded_out = loaded.predict(x)

    assert np.allclose(orig_out, loaded_out)


def test_save_keras(tmp_path, model, training_data):
    save_path = os.path.join(str(tmp_path), "model_output")
    tmp_path = os.path.join(str(tmp_path), "model_output_tmp")
    x, y = training_data

    save_model(model, save_path, tmp_path=tmp_path, output_type="keras")

    config = read_file(save_path + "-config.json")

    orig_sess = K.get_session()

    loaded = model_from_json(config)
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
    save_path = os.path.join(str(tmp_path), "model_output")
    tmp_path = os.path.join(str(tmp_path), "model_output_tmp")
    x, y = training_data

    save_model(model, save_path, tmp_path=tmp_path, output_type="tf")

    orig_out = model.predict(x)

    with tf.Session().as_default() as sess:
        with tf.Graph().as_default() as _:
            meta_graph_def = tf.saved_model.load(
                sess,
                tags=[tf.saved_model.tag_constants.SERVING],
                export_dir=save_path)

            loaded_out = sess.run(output_names, feed_dict=feed_dict)

    assert np.allclose(orig_out, loaded_out)


def test_save_frozen(
        tmp_path,
        model,
        training_data,
        feed_dict,
        output_names):
    save_path = os.path.join(str(tmp_path), "model_output")
    tmp_path = os.path.join(str(tmp_path), "model_output_tmp")
    x, y = training_data

    orig_out = model.predict(x)

    save_model(model, save_path, tmp_path=tmp_path, output_type="tf_frozen")

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default() as default_sess:
        with sess.graph.as_default() as default_graph:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(read_file(save_path, "rb"))
            tf.import_graph_def(graph_def, name="", return_elements=None)

            loaded_out = sess.run(output_names, feed_dict=feed_dict)

    assert np.allclose(orig_out, loaded_out)


def test_save_optimized(
        tmp_path,
        model,
        training_data,
        feed_dict,
        output_names):

    save_path = os.path.join(str(tmp_path), "model_output")

    x, y = training_data
    orig_out = model.predict(x)

    save_model(model, save_path, tmp_path=tmp_path, output_type="tf_optimized")

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default() as default_sess:
        with sess.graph.as_default() as default_graph:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(read_file(save_path, "rb"))
            tf.import_graph_def(graph_def, name="", return_elements=None)

            loaded_out = sess.run(output_names, feed_dict=feed_dict)

    assert np.allclose(orig_out, loaded_out)


def test_save_tf_lite(
        tmp_path,
        model,
        training_data,
        feed_dict,
        output_names):

    ver, sub_version, _ = tf.__version__.split('.')
    # there are bugs with saving as tf.lite in early version
    # see: https://github.com/tensorflow/tensorflow/issues/17349
    if int(ver) == 1 and int(sub_version) < 13:
        return

    save_path = os.path.join(str(tmp_path), "model_output")

    x, y = training_data
    orig_out = model.predict(x)

    save_model(model, save_path, tmp_path=tmp_path, output_type="tf_lite")

    with tf.Session().as_default() as sess:
        with tf.Graph().as_default() as _:
            interpreter = tf.contrib.lite.Interpreter(model_path=save_path)
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
