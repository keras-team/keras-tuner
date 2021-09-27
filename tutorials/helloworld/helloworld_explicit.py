import keras_tuner as kt
from tensorflow import keras, ones
from pprint import pprint
import numpy as np
from functools import partial


def last_value_task(n_input=20, n_train=1000) -> dict:
    """ 
        The task is learning the last value in a vector
        
    :param n_input:      Dimension of input
    :param n_train:      Number of training samples
  
    """
    n_output = 1
    x_train = np.random.randn(n_train, 1, n_input)
    y_train = x_train[:, :, n_input - 1]
    x_val = np.random.randn(n_train, 1, n_input)
    y_val = x_val[:, :, n_input - 1]
    x_test = np.random.randn(n_train, 1, n_input)
    y_test = x_test[:,:, n_input-1]
    return dict(n_input=n_input,n_output=n_output, n_train=n_train, x_train=x_train, x_test=x_test, x_val=x_val, y_train=y_train, y_test=y_test, y_val=y_val )


def tunertrain(d, n_input, epochs=50):
    """ Use keras-tuner to select the number of activations and units

        (Illustrates a more explicit calling pattern that ensures the model can be reconstructed,
         and possibly avoids confusion between hp names and keras arguments)

    """

    def build_model(hp, n_input:int):
        model = keras.Sequential()
        model.add(keras.layers.Dense(
            units= hp.Choice('units', [8, 16, 24, 32, 64]),
            activation=hp.Choice('activation_1',['linear','relu']),
            input_shape=(1, n_input) ))
        model.add(keras.layers.Dense(8, activation=hp.Choice('activation_2',['linear','relu'])))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(loss='mse')
        return model

    p_build = partial(build_model, n_input=n_input)
    tuner = kt.Hyperband(
        p_build,
        objective='val_loss',
        overwrite=True,
        max_epochs=100)

    tuner.search(d['x_train'], d['y_train'], epochs=epochs, validation_data=(d['x_val'], d['y_val']))
    print(tuner.results_summary())
    best_model = tuner.get_best_models()[0]
    return best_model


def verify_model_fit(d, model):
    """ Illustrates that the best_model can be recreated successfully """
    y_test_hat = model.predict(d['x_test'])
    test_error = float(keras.metrics.mean_squared_error(y_test_hat[:, 0, 0], d['y_test'][:, 0]))
    y_val_hat = model.predict(d['x_val'])
    val_error = float(keras.metrics.mean_squared_error(y_val_hat[:, 0, 0], d['y_val'][:, 0]))
    y_train_hat = model.predict(d['x_train'])
    train_error = float(keras.metrics.mean_squared_error(y_train_hat[:, 0, 0], d['y_train'][:, 0]))
    summary = {"train_error": train_error,
            "val_error": val_error,
            "test_error": test_error}
    return summary


if __name__=='__main__':
   n_input = 20
   d = last_value_task(n_input=20, n_train=1000)
   best_model = tunertrain(d=d,n_input=n_input)
   summary = verify_model_fit(d=d, model=best_model)
   pprint(summary)
