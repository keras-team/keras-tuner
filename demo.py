import random
import numpy as np
from hypertuner.distributions import Range, Choice, Fixed
from hypertuner.layers import MDense
from hypertuner.tuners import RandomSearch

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))

# Initial layer
IL_UNITS = Range(16, 32, 2)
IL_INPUT_SHAPE = Fixed(20)

# Hidden layer
L2_UNITS = Range(16, 32, 2)
L2_ACTIVATION = Choice(['relu', 'tanh'])
L2_OPTIONAL = True

# Last layer
LL_UNITS = Fixed(1)
LL_ACTIVATION = Choice(['sigmoid', 'tanh'])

# Compile options
OPTIMIZER = Choice([Adam(), SGD()])
LOSS = Choice(['binary_crossentropy', 'mse'])
METRICS = Fixed(['accuracy'])

# Meta model
mmodel = RandomSearch(iterations=20)
mmodel.add(MDense(IL_UNITS, input_shape=IL_INPUT_SHAPE))
mmodel.add(MDense(L2_UNITS, activation=L2_ACTIVATION, optional=L2_OPTIONAL))
mmodel.add(MDense(LL_UNITS, activation=['sigmoid']))
mmodel.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
mmodel.summary()
mmodel.search(x_train, y_train, validation_split=0.01)
mmodel.statistics()