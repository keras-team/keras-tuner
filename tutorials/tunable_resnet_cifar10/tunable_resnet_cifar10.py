"""Example on how to use Tunable Resnet."""

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from kerastuner.applications.tunable_resnet import TunableResNet
from kerastuner.tuners.randomsearch import RandomSearch


# Import the Cifar10 dataset.
NUM_CLASSES = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Import an hypertunable version of Resnet.
model_fn = TunableResNet(
    input_shape=x_train.shape[1:],
    num_classes=NUM_CLASSES)

# Initialize the hypertuner: we should find the model that maximixes the
# validation accuracy, training each model for three epochs for a max of
# 12 epochs of total training time.
tuner = RandomSearch(
    model_fn,
    objective='val_acc',
    epoch_budget=12,
    max_epochs=3,
    project='Cifar10',
    architecture='Resnet',
    validation_data=(x_test, y_test),
    max_params=50000000)

# Display search overview.
tuner.summary()

# Performs the hypertuning.
tuner.search(x_train, y_train, validation_split=0.01)

# Show the best models, their hyperparameters, and the resulting metrics.
tuner.results_summary()

# Export the top 2 models, in keras format format.
tuner.save_best_models(num_models=1)
