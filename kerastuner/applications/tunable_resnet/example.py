"""Example on how to use Tunable Resnet."""

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from kerastuner.applications.tunable_resnet import resnet, resnet_single_model
from kerastuner.tuners.randomsearch import RandomSearch


# Import the Cifar10 dataset.
NUM_CLASSES = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Import an hypertunable version of Resnet.
model_fn = resnet(
    input_shape=x_train.shape[1:],
    num_classes=NUM_CLASSES)

# Find the best model.
tuner = RandomSearch(
    model_fn,
    epoch_budget=10,
    max_epochs=3,
    project='Tunable Resnet',
    architecture='Resnet',
    validation_data=(x_test, y_test),
    max_params=50000000)
tuner.summary()
tuner.search(x_train, y_train, validation_split=0.01)
tuner.display_result_summary()
