# standard imports
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# hypertune imports
from kerastuner.distributions import Range, Choice, Boolean, Fixed
from kerastuner.tuners import RandomSearch

# Random data tp feed our model to show how easy it is to use KerasTuner
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))

# use https://fixme to track results
api_key = "fixme"


def model_fn():
    # Input layer
    IL_UNITS = Range('input dims', 16, 32, 2, group='inputs')

    # Hidden layer
    L2_UNITS = Range('dims', 16, 32, 2, group="hidden_layers")
    L2_ACTIVATION = Choice('activation', ['relu', 'tanh'], group="hidden_layers")
    L2_OPTIONAL = Boolean('use', group="hidden_layers")

    # Last layer
    LL_UNITS = Fixed('ouput dims', 1, group="output")
    LL_ACTIVATION = Choice('activation', ['sigmoid', 'tanh'], group="output")

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


hypermodel = RandomSearch(model_fn, epoch_budget=90, max_epochs=10)
hypermodel.summary()
hypermodel.search(x_train, y_train, validation_split=0.01)
