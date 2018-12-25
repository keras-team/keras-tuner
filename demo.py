import numpy as np

# standard Keras import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping

# hypertune imports
from kerastuner.distributions import Range, Choice, Boolean, Fixed
from kerastuner.tuners import RandomSearch

# just a simple model to demo how easy it is to use KerasTuner 
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
DRY_RUN = False # DRY_RUN: True don't train the models,  DRY_RUN: False: train models.
username = "demo@kerastuner.io"


def model_fn():
  # Input layer
  IL_UNITS = Range('input dims', 16, 32, 2, group='inputs')
  # Hidden layer
  L2_UNITS = Range('hidden dims', 16, 32, 2, group="hidden_layers")
  L2_ACTIVATION = Choice('hidden activation', ['relu', 'tanh'], group="hidden_layers")
  L2_OPTIONAL = Boolean('use hidden layer', group="hidden_layers")
  # Last layer
  LL_UNITS = Fixed('ouput dims', 1, group="output")
  LL_ACTIVATION = Choice('output activation', ['sigmoid', 'tanh'], group="output")
  # Compile options
  LOSS = Choice('loss', ['binary_crossentropy', 'mse'], group="optimizer")

  model = Sequential()
  model.add(Dense(IL_UNITS, input_shape=(20,)))
  if L2_OPTIONAL:
    model.add(Dense(L2_UNITS, activation=L2_ACTIVATION))
  model.add(Dense(LL_UNITS, activation=LL_ACTIVATION))
  optimizer = Adam(lr=0.001)
  model.compile(optimizer=optimizer, loss=LOSS, metrics=['accuracy'])
  return model

# which metrics to track across the runs and display
METRIC_TO_REPORT = [('loss', 'min'), ('val_loss', 'min'), ('acc', 'max'), ('val_acc', 'max')]
hypermodel = RandomSearch(model_fn, epoch_budget=90, max_epochs=10, dry_run=DRY_RUN, project="kerastuner-demo", architecture="MLP", info={'model_type':'MLP'}, metrics=METRIC_TO_REPORT)
hypermodel.summary()
hypermodel.backend(username=username)
hypermodel.search(x_train, y_train, validation_split=0.01)
