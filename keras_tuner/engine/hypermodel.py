# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"HyperModel base class."

from keras_tuner import errors


class HyperModel:
    """Defines a search space of models.

    A search space is a collection of models. The `build` function will build
    one of the models from the space using the given `HyperParameters` object.

    Users should subclass the `HyperModel` class to define their search spaces
    by overriding `build()`, which creates and returns the Keras model.
    Optionally, you may also override `fit()` to customize the training process
    of the model.

    Examples:

    In `build()`, you can create the model using the hyperparameters.

    ```python
    class MyHyperModel(kt.HyperModel):
        def build(self, hp):
            model = keras.Sequential()
            model.add(keras.layers.Dense(
                hp.Choice('units', [8, 16, 32]),
                activation='relu'))
            model.add(keras.layers.Dense(1, activation='relu'))
            model.compile(loss='mse')
            return model
    ```

    When overriding `HyperModel.fit()`, if you use `model.fit()` to train your
    model, which returns the training history, you can return it directly. You
    may use `hp` to specify any hyperparameters to tune.

    ```python
    class MyHyperModel(kt.HyperModel):
        def build(self, hp):
            ...

        def fit(self, hp, model, *args, **kwargs):
            return model.fit(
                *args,
                epochs=hp.Int("epochs", 5, 20),
                **kwargs)
    ```

    If you have a customized training process, you can return the objective
    value as a float.

    If you want to keep track of more metrics, you can return a dictionary of
    the metrics to track.

    ```python
    class MyHyperModel(kt.HyperModel):
        def build(self, hp):
            ...

        def fit(self, hp, model, *args, **kwargs):
            ...
            return {
                "loss": loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }
    ```

    Args:
        name: Optional string, the name of this HyperModel.
        tunable: Boolean, whether the hyperparameters defined in this
            hypermodel should be added to search space. If `False`, either the
            search space for these parameters must be defined in advance, or
            the default values will be used. Defaults to True.
    """

    def __init__(self, name=None, tunable=True):
        self.name = name
        self.tunable = tunable

        self._build = self.build
        self.build = self._build_wrapper

    def build(self, hp):
        """Builds a model.

        Args:
            hp: A `HyperParameters` instance.

        Returns:
            A model instance.
        """
        raise NotImplementedError

    def _build_wrapper(self, hp, *args, **kwargs):
        if not self.tunable:
            # Copy `HyperParameters` object so that new entries are not added
            # to the search space.
            hp = hp.copy()
        return self._build(hp, *args, **kwargs)

    def declare_hyperparameters(self, hp):
        pass

    def fit(self, hp, model, *args, **kwargs):
        """Train the model.

        Args:
            hp: HyperParameters.
            model: `keras.Model` built in the `build()` function.
            **kwargs: All arguments passed to `Tuner.search()` are in the
                `kwargs` here. It always contains a `callbacks` argument, which
                is a list of default Keras callback functions for model
                checkpointing, tensorboard configuration, and other tuning
                utilities. If `callbacks` is passed by the user from
                `Tuner.search()`, these default callbacks will be appended to
                the user provided list.

        Returns:
            A `History` object, which is the return value of `model.fit()`, a
            dictionary, or a float.

            If return a dictionary, it should be a dictionary of the metrics to
            track. The keys are the metric names, which contains the
            `objective` name. The values should be the metric values.

            If return a float, it should be the `objective` value.
        """
        return model.fit(*args, **kwargs)


class DefaultHyperModel(HyperModel):
    """Produces HyperModel from a model building function."""

    def __init__(self, build, name=None, tunable=True):
        super().__init__(name=name)
        self.build = build


def get_hypermodel(hypermodel):
    """Gets a HyperModel from a HyperModel or callable."""
    if hypermodel is None:
        return None

    if isinstance(hypermodel, HyperModel):
        return hypermodel

    if not callable(hypermodel):
        raise errors.FatalValueError(
            "The `hypermodel` argument should be either "
            "a callable with signature `build(hp)` returning a model, "
            "or an instance of `HyperModel`."
        )
    return DefaultHyperModel(hypermodel)
