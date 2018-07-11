import subprocess
import random
import numpy as np
import functools

# standard Keras import
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from tensorflow.examples.tutorials.mnist import input_data

# hypertune imports
from kerastuner.distributions import Range, Choice, Boolean, Fixed
from kerastuner.tuners import RandomSearch

DRY_RUN = False  # DRY_RUN: True don't train the models,  DRY_RUN: False: train models.

DATASET = random.choice(['MNIST'])  # Choose between RANDOM and MNIST
USERNAME = subprocess.check_output(
    "gcloud auth  list 2>/dev/null | awk '/^*/ {print $2}'",
    shell=True).strip()
# USERNAME = 'invernizzi.l@gmail.com'

# just a simple model to demo how easy it is to use KerasTuner
if DATASET == 'RANDOM':
    input_width = 20
    output_width = 1
    x_train = np.random.random((1000, input_width))
    y_train = np.random.randint(2, size=(1000, output_width))
elif DATASET == 'MNIST':
    input_width = 784
    output_width = 10
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels


def build_architecture(hidden_dims):
    # Input layer
    IL_UNITS = Range('input dims', 1, 4, 2)
    # Hidden layer
    L2_UNITS = Range('hidden dims', *hidden_dims)
    L2_ACTIVATION = Choice('hidden activation', ['relu', 'tanh'])
    L2_OPTIONAL = Boolean('use hidden layer')
    # Last layer
    LL_UNITS = Fixed('ouput dims', output_width)
    LL_ACTIVATION = Choice('output activation', ['sigmoid', 'tanh'])
    # Compile options
    LOSS = Choice('loss', ['binary_crossentropy', 'mse'])

    model = Sequential()
    model.add(Dense(IL_UNITS, input_shape=(input_width, )))
    if L2_OPTIONAL:
        model.add(Dense(L2_UNITS, activation=L2_ACTIVATION))
    model.add(Dense(LL_UNITS, activation=LL_ACTIVATION))
    model.compile(optimizer='adam', loss=LOSS, metrics=['accuracy'])
    return model


# Display architecture structure.
build_architecture(hidden_dims=(1, 2, 3)).summary()

PARAMS = [
    {
        'architecture': 'FancyArch',
        'build_params': {
            'hidden_dims': (16, 32, 2)
        },
        'epoch_budget': 30,
        'max_epochs': 20,
    },
    {
        'architecture': 'CheapArch',
        'build_params': {
            'hidden_dims': (1, 3, 2)
        },
        'epoch_budget': 10,
        'max_epochs': 2,
    },
]

random.shuffle(PARAMS)
for params in PARAMS:
    hypermodel = RandomSearch(
        functools.partial(build_architecture, **params['build_params']),
        epoch_budget=params['epoch_budget'],
        max_epochs=params['max_epochs'],
        dry_run=DRY_RUN,
        project=DATASET,
        architecture=params['architecture'],
        username=USERNAME,
        gs_dir='gs://keras-tuner.appspot.com',
        metrics=[  # Metrics to track across the runs and display
            ('loss', 'min'), ('val_loss', 'min'), ('acc', 'max'), ('val_acc',
                                                                   'max')
        ])
    hypermodel.search(x_train, y_train, validation_split=0.01)
