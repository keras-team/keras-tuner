# Copyright 2019 The Keras Tuner Authors
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import gc
import traceback

import numpy as np
from tensorflow import keras

from .. import config as config_module


class HyperModel(object):
    """Defines a search space of models.

    A search space is a collection of models. The `build` function will build
    one of the models from the space using the given `HyperParameters` object.

    Users should subclass the `HyperModel` class to define their search spaces
    by overriding the `.build(...)` function.

    Examples:

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


class DefaultHyperModel(HyperModel):
    """Produces HyperModel from a model building function."""

    def __init__(self, build, name=None, tunable=True):
        super(DefaultHyperModel, self).__init__(name=name)
        self.build = build


class KerasHyperModel(HyperModel):
    """Builds and compiles a Keras Model with optional compile overrides."""

    def __init__(
        self,
        hypermodel,
        max_model_size=None,
        optimizer=None,
        loss=None,
        metrics=None,
        distribution_strategy=None,
        **kwargs
    ):
        super(KerasHyperModel, self).__init__(**kwargs)
        self.hypermodel = get_hypermodel(hypermodel)
        self.max_model_size = max_model_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.distribution_strategy = distribution_strategy

        self._max_fail_streak = 5

    def build(self, hp):
        for i in range(self._max_fail_streak + 1):
            # clean-up TF graph from previously stored (defunct) graph
            keras.backend.clear_session()
            gc.collect()

            # Build a model, allowing max_fail_streak failed attempts.
            try:
                with maybe_distribute(self.distribution_strategy):
                    model = self.hypermodel.build(hp)
            except:
                if config_module.DEBUG:
                    traceback.print_exc()

                print("Invalid model %s/%s" % (i, self._max_fail_streak))

                if i == self._max_fail_streak:
                    raise RuntimeError("Too many failed attempts to build model.")
                continue

            # Stop if `build()` does not return a valid model.
            if not isinstance(model, keras.models.Model):
                raise RuntimeError(
                    "Model-building function did not return "
                    "a valid Keras Model instance, found {}".format(model)
                )

            # Check model size.
            size = maybe_compute_model_size(model)
            if self.max_model_size and size > self.max_model_size:
                print("Oversized model: {} parameters -- skipping".format(size))
                if i == self._max_fail_streak:
                    raise RuntimeError("Too many consecutive oversized models.")
                continue
            break

        return self._compile_model(model)

    def _compile_model(self, model):
        with maybe_distribute(self.distribution_strategy):
            if self.optimizer or self.loss or self.metrics:
                compile_kwargs = {
                    "optimizer": model.optimizer,
                    "loss": model.loss,
                    "metrics": model.metrics,
                }
                if self.loss:
                    compile_kwargs["loss"] = self.loss
                if self.optimizer:
                    compile_kwargs["optimizer"] = self.optimizer
                if self.metrics:
                    compile_kwargs["metrics"] = self.metrics
                model.compile(**compile_kwargs)
            return model


def maybe_compute_model_size(model):
    """Compute the size of a given model, if it has been built."""
    if model.built:
        params = [keras.backend.count_params(p) for p in model.trainable_weights]
        return int(np.sum(params))
    return 0


@contextlib.contextmanager
def maybe_distribute(distribution_strategy):
    """Distributes if distribution_strategy is set."""
    if distribution_strategy is None:
        yield
    else:
        with distribution_strategy.scope():
            yield


def get_hypermodel(hypermodel):
    """Gets a HyperModel from a HyperModel or callable."""
    if isinstance(hypermodel, HyperModel):
        return hypermodel
    else:
        if not callable(hypermodel):
            raise ValueError(
                "The `hypermodel` argument should be either "
                "a callable with signature `build(hp)` returning a model, "
                "or an instance of `HyperModel`."
            )
        return DefaultHyperModel(hypermodel)
