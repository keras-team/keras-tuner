"""Keras Tuner CIFAR10 example for the TensorFlow blog post."""

import kerastuner as kt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds


def build_model(hp):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = inputs
    for i in range(hp.Int("conv_blocks", 3, 5, default=3)):
        filters = hp.Int("filters_" + str(i), 32, 256, step=32)
        for _ in range(2):
            x = layers.Convolution2D(filters, kernel_size=(3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        if hp.Choice("pooling_" + str(i), ["avg", "max"]) == "max":
            x = layers.MaxPool2D()(x)
        else:
            x = layers.AvgPool2D()(x)
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(
        hp.Int("hidden_size", 30, 100, step=10, default=50), activation="relu"
    )(x)
    x = layers.Dropout(hp.Float("dropout", 0, 0.5, step=0.1, default=0.5))(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = kt.Hyperband(
    build_model, objective="val_accuracy", max_epochs=30, hyperband_iterations=2
)

data = tfds.load("cifar10")
train_ds, test_ds = data["train"], data["test"]


def standardize_record(record):
    return tf.cast(record["image"], tf.float32) / 255.0, record["label"]


train_ds = train_ds.map(standardize_record).batch(64).shuffle(10000)
test_ds = test_ds.map(standardize_record).batch(64)

tuner.search(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
)

best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
