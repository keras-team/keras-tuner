# Keras Tuner

A hyperparameter tuner for [Keras](https://keras.io), specifically for `tf.keras` with TensorFlow 2.0.

Full documentation and tutorials available on the [Keras Tuner website.](https://keras-team.github.io/keras-tuner/)

## Installation

Requirements:

- Python 3.6
- TensorFlow 2.0

Install latest release:

```
pip install -U keras-tuner
```

Install from source:

```
git clone https://github.com/keras-team/keras-tuner.git
cd keras-tuner
pip install .
```


## Usage: the basics

Here's how to perform hyperparameter tuning for a single-layer dense neural network using random search.

First, we define a model-building function. It takes an argument `hp` from which you can
sample hyperparameters, such as `hp.Int('units', min_value=32, max_value=512, step=32)`
(an integer from a certain range).

This function returns a compiled model.

```python
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model
```

Next, instantiate a tuner. You should specify the model-building function,
the name of the objective to optimize (whether to minimize or maximize is automatically inferred
for built-in metrics), the total number of trials (`max_trials`) to test, and the number
of models that should be built and fit for each trial (`executions_per_trial`).

Available tuners are `RandomSearch` and `Hyperband`.

**Note:** the purpose of having multiple executions per trial is to reduce results variance
and therefore be able to more accurately assess the performance of a model. If you want to get
results faster, you could set `executions_per_trial=1` (single round of training for each model configuration).

```python
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')
```

You can print a summary of the search space:

```python
tuner.search_space_summary()
```

Then, start the search for the best hyperparameter configuration.
The call to `search` has the same signature as `model.fit()`.

```python
tuner.search(x, y,
             epochs=5,
             validation_data=(val_x, val_y))
```

Here's what happens in `search`: models are built iteratively by calling the model-building function,
which populates the hyperparameter space (search space) tracked by the `hp` object. The tuner
progressively explores the space, recording metrics for each configuration.

When search is over, you can retrieve the best model(s):

```python
models = tuner.get_best_models(num_models=2)
```

Or print a summary of the results:

```python
tuner.results_summary()
```

You will also find detailed logs, checkpoints, etc, in the folder `my_dir/helloworld`, i.e. `directory/project_name`.


## The search space may contain conditional hyperparameters

Below, we have a `for` loop creating a tunable number of layers,
which themselves involve a tunable `units` parameter.

This can be pushed to any level of parameter interdependency, including recursion.

Note that all parameter names should be unique (here, in the loop over `i`,
we name the inner parameters `'units_' + str(i)`).

```python
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model
```

## You can use a HyperModel subclass instead of a model-building function

This makes it easy to share and reuse hypermodels.

A `HyperModel` subclass only needs to implement a `build(self, hp)` method.

```python
from kerastuner import HyperModel


class MyHyperModel(HyperModel):

    def __init__(self, classes):
        self.classes = classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
        model.add(layers.Dense(self.classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model


hypermodel = MyHyperModel(classes=10)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             epochs=5,
             validation_data=(val_x, val_y))
```

## Keras Tuner includes pre-made tunable applications: HyperResNet and HyperXception

These are ready-to-use hypermodels for computer vision.

They come pre-compiled with `loss="categorical_crossentropy"` and `metrics=["accuracy"]`.

```python
from kerastuner.applications import HyperResNet
from kerastuner.tuners import Hyperband

hypermodel = HyperResNet(input_shape=(128, 128, 3), classes=10)

tuner = Hyperband(
    hypermodel,
    objective='val_accuracy',
    max_epochs=40,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             validation_data=(val_x, val_y))
```

## You can easily restrict the search space to just a few parameters

If you have an existing hypermodel, and you want to search over only a few parameters
(such as the learning rate), you can do so by passing a `hyperparameters` argument
to the tuner constructor, as well as `tune_new_entries=False` to specify that parameters
that you didn't list in `hyperparameters` should not be tuned. For these parameters, the default
value gets used.

```python
from kerastuner import HyperParameters

hypermodel = HyperXception(input_shape=(128, 128, 3), classes=10)

hp = HyperParameters()
# This will override the `learning_rate` parameter with your
# own selection of choices
hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

tuner = Hyperband(
    hypermodel,
    hyperparameters=hp,
    # `tune_new_entries=False` prevents unlisted parameters from being tuned
    tune_new_entries=False,
    objective='val_accuracy',
    max_epochs=40,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             validation_data=(val_x, val_y))
```

Want to know what parameter names are available? Read the code.


## About parameter default values

Whenever you register a hyperparameter inside a model-building function or the `build` method of a hypermodel,
you can specify a default value:

```python
hp.Int('units',
       min_value=32,
       max_value=512,
       step=32,
       default=128)
```

If you don't, hyperparameters always have a default default (for `Int`, it is equal to `min_value`).


## Fixing values in a hypermodel

What if you want to do the reverse -- tune all available parameters in a hypermodel, **except** one (the learning rate)?

Pass a `hyperparameters` argument with a `Fixed` entry (or any number of `Fixed` entries), and specify `tune_new_entries=True`.


```python
hypermodel = HyperXception(input_shape=(128, 128, 3), classes=10)

hp = HyperParameters()
hp.Fixed('learning_rate', value=1e-4)

tuner = Hyperband(
    hypermodel,
    hyperparameters=hp,
    tune_new_entries=True,
    objective='val_accuracy',
    max_epochs=40,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             validation_data=(val_x, val_y))
```

## Overriding compilation arguments

If you have a hypermodel for which you want to change the existing optimizer, loss, or metrics, you can do so by passing these arguments
to the tuner constructor:

```python
hypermodel = HyperXception(input_shape=(128, 128, 3), classes=10)

tuner = Hyperband(
    hypermodel,
    optimizer=keras.optimizers.Adam(1e-3),
    loss='mse',
    metrics=[keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')],
    objective='val_precision',
    max_epochs=40,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             validation_data=(val_x, val_y))
```


## Citation

BibTex entry:
```bibtex
@misc{omalley2019kerastuner,
	title        = {Keras {Tuner}},
	author       = {O'Malley, Tom and Bursztein, Elie and Long, James and Chollet, Fran\c{c}ois and Jin, Haifeng and Invernizzi, Luca and others},
	year         = 2019,
	howpublished = {\url{https://github.com/keras-team/keras-tuner}}
}
```
