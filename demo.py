import numpy as np
# standard Keras import
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping

# hypertune imports
from kerastuner.distributions import Range, Choice, Fixed, Boolean
from kerastuner.tuners import RandomSearch

# just a simple model to demo how easy it is to use KerasTuner 
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
DRY_RUN = False # DRY_RUN: True don't train the models,  DRY_RUN: False: train models.

def model_fn():
  # Initial layer
  IL_UNITS = Range(16, 32, 2)
  # Hidden layer
  L2_UNITS = Range(16, 32, 2)
  L2_ACTIVATION = Choice('relu', 'tanh')
  L2_OPTIONAL = Boolean()
  # Last layer
  LL_UNITS = Fixed(1)
  LL_ACTIVATION = Choice('sigmoid', 'tanh')
  # Compile options
  LOSS = Choice('binary_crossentropy', 'mse')

  model = Sequential()
  model.add(Dense(IL_UNITS, input_shape=(20,)))
  if L2_OPTIONAL:
    model.add(Dense(L2_UNITS, activation=L2_ACTIVATION))
  model.add(Dense(LL_UNITS, activation=LL_ACTIVATION))
  model.compile(optimizer='adam', loss=LOSS, metrics=['accuracy'])
  return model

# which metrics to track across the runs and display
METRIC_TO_REPORT = [('loss', 'min'), ('val_loss', 'min'), ('acc', 'max'), ('val_acc', 'max')]
hypermodel = RandomSearch(model_fn, epoch_budget=90, max_epochs=10, dry_run=DRY_RUN, model_name="kerastuner-demo", metrics=METRIC_TO_REPORT)
hypermodel.search(x_train, y_train, validation_split=0.01)