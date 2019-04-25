
# standard imports
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# hypertune imports
from kerastuner.distributions import Range, Choice, Boolean, Fixed
from kerastuner.tuners import RandomSearch
# Random data to feed the model to show how easy it is to use KerasTuner
x_train = np.random.random((10000, 20))
y_train = np.random.randint(2, size=(10000, 1))
# You can use http://keras-tuner.appspot.com to track results on the web, and
# get notifications. To do so, grab an API key on that site, and fill it here.
api_key = ''
def model_fn():
    # Input layer
    IL_UNITS = Range('dims', 16, 32, 2, group='inputs')
    # Hidden layer
    L2_UNITS = Range('dims', 16, 32, 2, group="hidden_layers")
    L2_ACTIVATION = Choice(
        'activation', ['relu', 'tanh'], group="hidden_layers")
    L2_OPTIONAL = Boolean('2nd hidden layer', group="hidden_layers")
    # Last layer
    LL_UNITS = Fixed('dims', 1, group="output")
    LL_ACTIVATION = Choice('activation', ['sigmoid', 'tanh'], group="output")
    # Compile options
    LOSS = Choice('loss', ['binary_crossentropy', 'mse'], group="optimizer")
    model = Sequential()
    model.add(Dense(IL_UNITS, input_shape=(20,)))
    model.add(Dense(L2_UNITS, activation=L2_ACTIVATION))
    if L2_OPTIONAL:
        model.add(Dense(L2_UNITS, activation=L2_ACTIVATION))
    model.add(Dense(LL_UNITS, activation=LL_ACTIVATION))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss=LOSS)#, metrics=['accuracy'])
    return model
# train 3 models over 3 epochs
hypermodel = RandomSearch(model_fn, 'loss', epoch_budget=4, max_epochs=2)
hypermodel.summary()
if api_key:
    hypermodel.enable_cloud(
        api_key=api_key,
        # url='http://localhost:5000/api/'
    )
hypermodel.search(x_train, y_train, validation_split=0.01)
# Show the best models, their hyperparameters, and the resulting metrics.
#hypermodel.display_result_summary()

hypermodel.save_best_model(output_type="tf_optimized")
