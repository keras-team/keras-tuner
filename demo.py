import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD

from kerastuner.distributions import Range, Choice, Fixed, Bool
from kerastuner.tuners import RandomSearch

x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))

def model_fn():
  # Initial layer
  IL_UNITS = Range(16, 32, 2)
  # Hidden layer
  L2_UNITS = Range(16, 32, 2)
  L2_ACTIVATION = Choice(['relu', 'tanh'])
  L2_OPTIONAL = Bool()
  # Last layer
  LL_UNITS = Fixed(1)
  LL_ACTIVATION = Choice(['sigmoid', 'tanh'])
  # Compile options
  OPTIMIZER = Choice([Adam(), SGD()])
  LOSS = Choice(['binary_crossentropy', 'mse'])
  METRICS = Fixed(['accuracy'])

  model = Sequential()
  model.add(Dense(IL_UNITS, input_shape=(20,)))
  if L2_OPTIONAL:
    model.add(Dense(L2_UNITS, activation=L2_ACTIVATION))
  model.add(Dense(LL_UNITS, activation=LL_ACTIVATION))
  model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
  return model

#model = model_fn()
#model.summary()
mmodel = RandomSearch(model_fn, iterations=5)
#mmodel = RandomSearch(model_fn, iterations=20)
mmodel.search(x_train, y_train, validation_split=0.01)
mmodel.statistics()