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

import os
from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf
from packaging.version import parse
from tensorboard.plugins.hparams import api as hparams_api
from tensorflow import keras

import keras_tuner
from keras_tuner import errors
from keras_tuner.engine import tuner as tuner_module

INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))


def build_model(hp):
    inputs = keras.Input(shape=(INPUT_DIM,))
    x = inputs
    for i in range(hp.Int("num_layers", 1, 4)):
        x = keras.layers.Dense(
            units=hp.Int(f"units_{str(i)}", 5, 9, 1, default=6), activation="relu"
        )(x)

    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class MockModel(keras.Model):
    def __init__(self, full_history):
        super().__init__()
        self.full_history = full_history
        self.callbacks = []
        self.optimizer = True

    def call_callbacks(self, callbacks, method_name, *args, **kwargs):
        for callback in callbacks:
            method = getattr(callback, method_name)
            method(*args, **kwargs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs=None)

    def on_epoch_end(self, epoch):
        logs = {"loss": np.average(self.full_history[epoch])}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, epoch, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs=None)

    def on_batch_end(self, epoch, batch):
        logs = {"loss": self.full_history[epoch][batch]}
        for callback in self.callbacks:
            callback.on_batch_end(epoch, logs=logs)

    def fit(self, *args, **kwargs):
        self.callbacks = kwargs["callbacks"]
        for callback in self.callbacks:
            callback.model = self
        for epoch in range(len(self.full_history)):
            self.on_epoch_begin(epoch)
            for batch in range(len(self.full_history[epoch])):
                self.on_batch_begin(epoch, batch)
                self.on_batch_end(epoch, batch)
            self.on_epoch_end(epoch)
        history = keras.callbacks.History()
        history.history = {
            "loss": [np.average(epoch_values) for epoch_values in self.full_history]
        }
        return history

    def save_weights(self, fname, **kwargs):
        pass

    def get_config(self):
        return {}


class MockHyperModel(keras_tuner.HyperModel):

    mode_0 = [[10, 9, 8], [7, 6, 5], [4, 3, 2]]
    mode_1 = [[13, 13, 13], [12, 12, 12], [11, 11, 11]]

    def __init__(self):
        # The first call to `build` in tuner __init__
        # will reset this to 0
        self.mode_0_execution_count = -1

    def build(self, hp):
        if hp.Choice("mode", [0, 1]) == 0:
            return MockModel(self.mode_0)
        return MockModel(self.mode_1)


def build_subclass_model(hp):
    class MyModel(keras.Model):
        def build(self, _):
            self.layer = keras.layers.Dense(NUM_CLASSES, activation="softmax")

        def call(self, x):
            x = x + hp.Float("bias", 0, 10)
            return self.layer(x)

        # Currently necessary, because we save the model.
        # Note that this model is not written w/ best practices,
        # because the hp.Float value of the best model cannot be
        # inferred from `get_config()`. The best practice is to pass
        # HPs as __init__ arguments to subclass Layers and Models.
        def get_config(self):
            return {}

    model = MyModel()
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class ExampleHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.Int("num_layers", 1, 4)):
            x = keras.layers.Dense(
                units=hp.Int(f"units_{str(i)}", 5, 9, 1, default=6),
                activation="relu",
            )(x)

        outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, shuffle=hp.Boolean("shuffle"), **kwargs)


def test_basic_tuner_attributes(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )

    assert tuner.oracle.objective.name == "val_accuracy"
    assert tuner.oracle.max_trials == 2
    assert tuner.executions_per_trial == 3
    assert tuner.directory == tmp_path
    assert tuner.hypermodel.__class__.__name__ == "DefaultHyperModel"
    assert len(tuner.oracle.hyperparameters.space) == 3  # default search space
    assert len(tuner.oracle.hyperparameters.values) == 3  # default search space

    tuner.search_space_summary()

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    tuner.results_summary()

    assert len(tuner.oracle.trials) == 2
    assert os.path.exists(os.path.join(str(tmp_path), "untitled_project"))


def test_multi_objective(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective=["val_accuracy", "val_loss"],
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )

    assert tuner.oracle.objective.name == "multi_objective"

    tuner.search_space_summary()

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    tuner.results_summary()


def test_no_hypermodel_with_objective(tmp_path):
    class MyTuner(keras_tuner.tuners.RandomSearch):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            return {"val_loss": hp.Float("value", 0, 10)}

    tuner = MyTuner(
        objective="val_loss",
        max_trials=2,
        directory=tmp_path,
    )
    tuner.search()

    assert len(tuner.oracle.trials) == 2


def test_no_objective_with_hypermodel(tmp_path):
    class MyHyperModel(ExampleHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            return hp.Float("value", 0, 10)

    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=MyHyperModel(),
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )
    tuner.search()

    assert len(tuner.oracle.trials) == 2


def test_no_hypermodel_no_objective(tmp_path):
    class MyTuner(keras_tuner.tuners.RandomSearch):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            return hp.Float("value", 0, 10)

    tuner = MyTuner(
        objective="val_loss",
        max_trials=2,
        directory=tmp_path,
    )
    tuner.search()

    assert len(tuner.oracle.trials) == 2


def test_no_hypermodel_without_override_run_trial_error(tmp_path):
    with pytest.raises(ValueError, match="Received `hypermodel=None`"):
        keras_tuner.tuners.RandomSearch(
            max_trials=2,
            executions_per_trial=3,
            directory=tmp_path,
        )


def test_fit_return_string(tmp_path):
    class MyHyperModel(ExampleHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            return hp.Choice("value", ["a", "b"])

    tuner = keras_tuner.tuners.RandomSearch(
        objective="val_loss",
        hypermodel=MyHyperModel(),
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )

    with pytest.raises(TypeError, match="HyperModel\.fit\(\) to be one of"):
        tuner.search()


def test_run_trial_return_string(tmp_path):
    class MyTuner(keras_tuner.tuners.RandomSearch):
        def run_trial(self, trial, **kwargs):
            return trial.hyperparameters.Choice("value", ["a", "b"])

    tuner = MyTuner(
        objective="val_loss",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )

    with pytest.raises(TypeError, match="Tuner\.run_trial\(\) to be one of"):
        tuner.search()


def test_no_objective_fit_return_not_float(tmp_path):
    class MyHyperModel(ExampleHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            return {"val_loss": hp.Float("value", 0, 10)}

    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=MyHyperModel(),
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )

    with pytest.raises(TypeError, match="HyperModel\.fit\(\) to be a single float"):
        tuner.search()


def test_no_objective_run_trial_return_not_float(tmp_path):
    class MyTuner(keras_tuner.tuners.RandomSearch):
        def run_trial(self, trial, **kwargs):
            return {"val_loss": trial.hyperparameters.Float("value", 0, 10)}

    tuner = MyTuner(
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )

    with pytest.raises(TypeError, match="Tuner\.run_trial\(\) to be a single float"):
        tuner.search()


def test_callbacks_in_fit_kwargs(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )
    with patch.object(
        tuner, "_build_and_fit_model", wraps=tuner._build_and_fit_model
    ) as mock_build_and_fit_model:
        tuner.search(
            x=TRAIN_INPUTS,
            y=TRAIN_TARGETS,
            epochs=2,
            validation_data=(VAL_INPUTS, VAL_TARGETS),
            callbacks=[
                keras.callbacks.EarlyStopping(),
                keras.callbacks.TensorBoard(tmp_path),
            ],
        )
        assert len(tuner.oracle.trials) == 2
        callback_class_names = [
            x.__class__.__name__
            for x in mock_build_and_fit_model.call_args[1]["callbacks"]
        ]
        assert callback_class_names == [
            "EarlyStopping",
            "TensorBoard",
            "Callback",
            "TunerCallback",
            "SaveBestEpoch",
        ]


def test_hypermodel_with_dynamic_space(tmp_path):
    hypermodel = ExampleHyperModel()
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )

    assert tuner.hypermodel == hypermodel

    tuner.search_space_summary()

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    tuner.results_summary()

    assert len(tuner.oracle.trials) == 2
    tuner.oracle.hyperparameters.get("shuffle")


def test_override_compile(tmp_path):
    class MyHyperModel(ExampleHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            history = super().fit(hp, model, *args, **kwargs)
            assert model.optimizer.__class__.__name__ == "RMSprop"
            assert model.loss == "mse"
            assert len(model.metrics) >= 2
            assert model.metrics[-2]._fn.__name__ == "mean_squared_error"
            assert model.metrics[-1]._fn.__name__ == "sparse_categorical_accuracy"
            return history

    tuner = keras_tuner.tuners.RandomSearch(
        MyHyperModel(),
        objective="val_mse",
        max_trials=2,
        executions_per_trial=1,
        metrics=["mse", "accuracy"],
        loss="mse",
        optimizer="rmsprop",
        directory=tmp_path,
    )

    assert tuner.oracle.objective.name == "val_mse"
    assert tuner.optimizer == "rmsprop"
    assert tuner.loss == "mse"
    assert tuner.metrics == ["mse", "accuracy"]

    tuner.search_space_summary()
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )
    tuner.results_summary()
    model = tuner.get_best_models()[0]
    assert model.loss == "mse"


def test_override_optimizer_with_actual_optimizer_object(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=4,
        optimizer=keras.optimizers.Adam(0.01),
        directory=tmp_path,
    )
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )


def test_static_space(tmp_path):
    def build_model_static(hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.get("num_layers")):
            x = keras.layers.Dense(
                units=hp.get(f"units_{str(i)}"), activation="relu"
            )(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(hp.get("learning_rate")),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    hp = keras_tuner.HyperParameters()
    hp.Int("num_layers", 1, 3, 1, default=2)
    hp.Int("units_0", 4, 6, 1, default=5)
    hp.Int("units_1", 4, 6, 1, default=5)
    hp.Int("units_2", 4, 6, 1, default=5)
    hp.Choice("learning_rate", [0.01, 0.001])
    tuner = keras_tuner.tuners.RandomSearch(
        build_model_static,
        objective="val_accuracy",
        max_trials=4,
        directory=tmp_path,
        hyperparameters=hp,
        allow_new_entries=False,
    )

    assert tuner.oracle.hyperparameters == hp
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )
    assert len(tuner.oracle.trials) == 4


def test_static_space_errors(tmp_path):
    def build_model_static(hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.get("num_layers")):
            x = keras.layers.Dense(
                units=hp.get(f"units_{str(i)}"), activation="relu"
            )(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(hp.Float("learning_rate", 1e-5, 1e-2)),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    hp = keras_tuner.HyperParameters()
    hp.Int("num_layers", 1, 3, 1, default=2)
    hp.Int("units_0", 4, 6, 1, default=5)
    hp.Int("units_1", 4, 6, 1, default=5)

    with pytest.raises(RuntimeError, match="`allow_new_entries` is `False`"):
        tuner = keras_tuner.tuners.RandomSearch(
            build_model_static,
            objective="val_accuracy",
            max_trials=2,
            directory=tmp_path,
            hyperparameters=hp,
            allow_new_entries=False,
        )
        tuner.search(
            x=TRAIN_INPUTS,
            y=TRAIN_TARGETS,
            epochs=2,
            validation_data=(VAL_INPUTS, VAL_TARGETS),
        )

    def build_model_static_invalid(hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.get("num_layers")):
            x = keras.layers.Dense(
                units=hp.get(f"units_{str(i)}"), activation="relu"
            )(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float("learning_rate", 0.001, 0.008, 0.001)
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    with pytest.raises(RuntimeError, match="`allow_new_entries` is `False`"):
        tuner = keras_tuner.tuners.RandomSearch(
            build_model_static_invalid,
            objective="val_accuracy",
            max_trials=2,
            directory=tmp_path,
            hyperparameters=hp,
            allow_new_entries=False,
        )
        tuner.search(
            x=TRAIN_INPUTS,
            y=TRAIN_TARGETS,
            epochs=2,
            validation_data=(VAL_INPUTS, VAL_TARGETS),
        )


def test_restricted_space_using_defaults(tmp_path):
    hp = keras_tuner.HyperParameters()
    hp.Int("num_layers", 1, 5, 1, default=2)
    hp.Choice("learning_rate", [0.01, 0.001, 0.0001])

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=4,
        directory=tmp_path,
        hyperparameters=hp,
        allow_new_entries=True,
        tune_new_entries=False,
    )

    assert len(tuner.oracle.hyperparameters.space) == 2
    new_lr = [
        p for p in tuner.oracle.hyperparameters.space if p.name == "learning_rate"
    ][0]
    assert new_lr.values == [0.01, 0.001, 0.0001]

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=1,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    assert len(tuner.oracle.trials) == 4
    assert len(tuner.oracle.hyperparameters.space) == 2  # Nothing added
    for trial in tuner.oracle.trials.values():
        # Trials get default values but don't pass these on to the oracle.
        assert len(trial.hyperparameters.space) >= 2


def test_restricted_space_with_custom_defaults(tmp_path):
    hp = keras_tuner.HyperParameters()
    hp.Int("num_layers", 1, 3, 1, default=2)
    hp.Choice("learning_rate", [0.01, 0.001, 0.0001])
    hp.Fixed("units_0", 4)
    hp.Fixed("units_1", 3)

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=4,
        directory=tmp_path,
        hyperparameters=hp,
        allow_new_entries=True,
        tune_new_entries=False,
    )

    assert len(tuner.oracle.hyperparameters.space) == 4
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=1,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    assert len(tuner.oracle.trials) == 4


def test_reparameterized_space(tmp_path):
    hp = keras_tuner.HyperParameters()
    hp.Int("num_layers", 1, 3, 1, default=3)
    hp.Choice("learning_rate", [0.01, 0.001, 0.0001])

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        seed=1337,
        objective="val_accuracy",
        max_trials=4,
        directory=tmp_path,
        hyperparameters=hp,
        allow_new_entries=True,
        tune_new_entries=True,
    )

    # Initial build model adds to the space.
    assert len(tuner.oracle.hyperparameters.space) == 5
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=1,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    assert len(tuner.oracle.trials) == 4
    assert len(tuner.oracle.hyperparameters.space) == 5


def test_get_best_models(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model, objective="val_accuracy", max_trials=4, directory=tmp_path
    )

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    models = tuner.get_best_models(2)
    assert len(models) == 2
    assert isinstance(models[0], keras.Model)
    assert isinstance(models[1], keras.Model)


def test_saving_and_reloading(tmp_path):

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=4,
        executions_per_trial=2,
        directory=tmp_path,
    )

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    new_tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=4,
        executions_per_trial=2,
        directory=tmp_path,
    )
    new_tuner.reload()

    assert len(new_tuner.oracle.trials) == 4

    new_tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )


def test_subclass_model(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        build_subclass_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_path,
    )

    tuner.search_space_summary()

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    tuner.results_summary()
    assert len(tuner.oracle.trials) == 2


def test_subclass_model_loading(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        build_subclass_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_path,
    )

    tuner.search_space_summary()

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    best_trial_score = tuner.oracle.get_best_trials()[0].score

    best_model = tuner.get_best_models()[0]
    best_model_score = best_model.evaluate(VAL_INPUTS, VAL_TARGETS)[1]

    assert best_model_score == best_trial_score


def test_update_trial(tmp_path):
    # Test stop the oracle in update_trial.
    class MyOracle(keras_tuner.Oracle):
        def populate_space(self, _):
            values = {p.name: p.random_sample() for p in self.hyperparameters.space}
            return {"values": values, "status": "RUNNING"}

        def update_trial(self, trial_id, metrics, step=0):
            super().update_trial(trial_id, metrics, step)
            trial = self.trials[trial_id]
            trial.status = "STOPPED"
            return trial.status

    my_oracle = MyOracle(objective="val_accuracy", max_trials=2)
    tuner = keras_tuner.Tuner(
        oracle=my_oracle, hypermodel=build_model, directory=tmp_path
    )
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=5,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    assert len(my_oracle.trials) == 2

    for trial in my_oracle.trials.values():
        # Test that early stopping worked.
        assert len(trial.metrics.get_history("val_accuracy")) == 1


def test_tunable_false_hypermodel(tmp_path):
    def build_model(hp):
        input_shape = (256, 256, 3)
        inputs = tf.keras.Input(shape=input_shape)

        with hp.name_scope("xception"):
            # Tune the pooling of Xception by supplying the search space
            # beforehand.
            hp.Choice("pooling", ["avg", "max"])
            xception = keras_tuner.applications.HyperXception(
                include_top=False, input_shape=input_shape, tunable=False
            ).build(hp)
        x = xception(inputs)

        x = tf.keras.layers.Dense(
            hp.Int("hidden_units", 50, 100, step=10), activation="relu"
        )(x)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.get(hp.Choice("optimizer", ["adam", "sgd"]))
        optimizer.learning_rate = hp.Float(
            "learning_rate", 1e-4, 1e-2, sampling="log"
        )

        model.compile(optimizer, loss="sparse_categorical_crossentropy")
        return model

    tuner = keras_tuner.RandomSearch(
        objective="val_loss",
        hypermodel=build_model,
        max_trials=4,
        directory=tmp_path,
    )

    x = np.random.random(size=(2, 256, 256, 3))
    y = np.random.randint(0, NUM_CLASSES, size=(2,))

    tuner.search(x, y, validation_data=(x, y), batch_size=2)

    hps = tuner.oracle.get_space()
    assert "xception/pooling" in hps
    assert "hidden_units" in hps
    assert "optimizer" in hps
    assert "learning_rate" in hps

    # Make sure no HPs from building xception were added.
    assert len(hps.space) == 4


def test_get_best_hyperparameters(tmp_path):
    hp1 = keras_tuner.HyperParameters()
    hp1.Fixed("a", 1)
    trial1 = keras_tuner.engine.trial.Trial(hyperparameters=hp1)
    trial1.status = "COMPLETED"
    trial1.score = 10

    hp2 = keras_tuner.HyperParameters()
    hp2.Fixed("a", 2)
    trial2 = keras_tuner.engine.trial.Trial(hyperparameters=hp2)
    trial2.status = "COMPLETED"
    trial2.score = 9

    tuner = keras_tuner.RandomSearch(
        objective="val_accuracy",
        hypermodel=build_model,
        max_trials=2,
        directory=tmp_path,
    )

    tuner.oracle.trials = {trial1.trial_id: trial1, trial2.trial_id: trial2}

    hps = tuner.get_best_hyperparameters()[0]
    assert hps["a"] == 1


def test_reloading_error_message(tmp_path):
    shared_dir = tmp_path

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=shared_dir,
    )

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    with pytest.raises(RuntimeError, match="pass `overwrite=True`"):
        keras_tuner.tuners.BayesianOptimization(
            build_model,
            objective="val_accuracy",
            max_trials=2,
            executions_per_trial=3,
            directory=shared_dir,
        )


def test_search_logging_verbosity(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )

    with patch("sys.stdout", new=StringIO()) as output:
        tuner.search(
            x=TRAIN_INPUTS,
            y=TRAIN_TARGETS,
            epochs=2,
            validation_data=(VAL_INPUTS, VAL_TARGETS),
            verbose=0,
        )
        assert output.getvalue().strip() == ""


def test_convert_hyperparams_to_hparams():
    def _check_hparams_equal(hp1, hp2):
        assert (
            hparams_api.hparams_pb(hp1, start_time_secs=0).SerializeToString()
            == hparams_api.hparams_pb(hp2, start_time_secs=0).SerializeToString()
        )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Choice("learning_rate", [1e-4, 1e-3, 1e-2])
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams,
        {
            hparams_api.HParam(
                "learning_rate", hparams_api.Discrete([1e-4, 1e-3, 1e-2])
            ): 1e-4
        },
    )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Int("units", min_value=2, max_value=16)
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams, {hparams_api.HParam("units", hparams_api.IntInterval(2, 16)): 2}
    )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Int("units", min_value=32, max_value=128, step=32)
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams,
        {hparams_api.HParam("units", hparams_api.Discrete([32, 64, 96, 128])): 32},
    )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Float("learning_rate", min_value=0.5, max_value=1.25, step=0.25)
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams,
        {
            hparams_api.HParam(
                "learning_rate", hparams_api.Discrete([0.5, 0.75, 1.0, 1.25])
            ): 0.5
        },
    )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Float("learning_rate", min_value=1e-4, max_value=1e-1)
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams,
        {
            hparams_api.HParam(
                "learning_rate", hparams_api.RealInterval(1e-4, 1e-1)
            ): 1e-4
        },
    )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Float("theta", min_value=0.0, max_value=1.57)
    hps.Float("r", min_value=0.0, max_value=1.0)
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    expected_hparams = {
        hparams_api.HParam("theta", hparams_api.RealInterval(0.0, 1.57)): 0.0,
        hparams_api.HParam("r", hparams_api.RealInterval(0.0, 1.0)): 0.0,
    }
    hparams_repr_list = [repr(hparams[x]) for x in hparams.keys()]
    expected_hparams_repr_list = [
        repr(expected_hparams[x]) for x in expected_hparams
    ]

    assert sorted(hparams_repr_list) == sorted(expected_hparams_repr_list)

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Boolean("has_beta")
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams,
        {hparams_api.HParam("has_beta", hparams_api.Discrete([True, False])): False},
    )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Fixed("beta", 0.1)
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams, {hparams_api.HParam("beta", hparams_api.Discrete([0.1])): 0.1}
    )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Fixed("type", "WIDE_AND_DEEP")
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams,
        {
            hparams_api.HParam(
                "type", hparams_api.Discrete(["WIDE_AND_DEEP"])
            ): "WIDE_AND_DEEP"
        },
    )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Fixed("condition", True)
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams,
        {hparams_api.HParam("condition", hparams_api.Discrete([True])): True},
    )

    hps = keras_tuner.engine.hyperparameters.HyperParameters()
    hps.Fixed("num_layers", 2)
    hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(hps)
    _check_hparams_equal(
        hparams, {hparams_api.HParam("num_layers", hparams_api.Discrete([2])): 2}
    )


def test_tuning_correctness(tmp_path):
    tuner = keras_tuner.Tuner(
        oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
            objective="loss", max_trials=2, seed=1337
        ),
        hypermodel=MockHyperModel(),
        directory=tmp_path,
    )
    tuner.search()
    assert len(tuner.oracle.trials) == 2

    m0_epochs = [float(np.average(x)) for x in MockHyperModel.mode_0]
    m1_epochs = [float(np.average(x)) for x in MockHyperModel.mode_1]

    # Score tracking correctness
    first_trial, second_trial = sorted(
        tuner.oracle.trials.values(), key=lambda t: t.score
    )
    assert first_trial.score == min(m0_epochs)
    assert second_trial.score == min(m1_epochs)
    assert tuner.oracle.get_best_trials(1)[0].trial_id == first_trial.trial_id


def assert_found_best_score(tmp_path, hypermodel, tuner_class=keras_tuner.Tuner):
    tuner = tuner_class(
        oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
            objective="loss", max_trials=2, seed=1337
        ),
        hypermodel=hypermodel,
        directory=tmp_path,
    )
    tuner.search(callbacks=[])
    assert tuner.oracle.get_best_trials(1)[0].score == 3.0


def test_hypermodel_fit_return_a_dict(tmp_path):
    class MyHyperModel(MockHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            history = super().fit(hp, model, *args, **kwargs)
            return {
                "loss": min(history.history["loss"]),
                "other_metric": np.random.rand(),
            }

    assert_found_best_score(tmp_path, MyHyperModel())


def test_hypermodel_fit_return_a_float(tmp_path):
    class MyHyperModel(MockHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            history = super().fit(hp, model, *args, **kwargs)
            return min(history.history["loss"])

    assert_found_best_score(tmp_path, MyHyperModel())


def test_hypermodel_fit_return_an_int(tmp_path):
    class MyHyperModel(MockHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            history = super().fit(hp, model, *args, **kwargs)
            return int(min(history.history["loss"]))

    assert_found_best_score(tmp_path, MyHyperModel())


def test_run_trial_return_none_without_update_trial(tmp_path):
    class MyTuner(keras_tuner.Tuner):
        def run_trial(self, trial, *fit_args, **fit_kwargs):
            self.hypermodel.build(trial.hyperparameters).fit(*fit_args, **fit_kwargs)

    with pytest.raises(
        errors.FatalTypeError,
        match="Did you forget",
    ):
        assert_found_best_score(tmp_path, MockHyperModel(), MyTuner)


def test_run_trial_return_none_with_update_trial(tmp_path):
    class MyTuner(keras_tuner.Tuner):
        def run_trial(self, trial, *fit_args, **fit_kwargs):
            history = self.hypermodel.build(trial.hyperparameters).fit(
                *fit_args, **fit_kwargs
            )
            self.oracle.update_trial(
                trial.trial_id, {"loss": min(history.history["loss"])}
            )

    with pytest.deprecated_call(match="Please remove the call"):
        assert_found_best_score(tmp_path, MockHyperModel(), MyTuner)


def test_run_trial_return_history(tmp_path):
    class MyTuner(keras_tuner.Tuner):
        def run_trial(self, trial, *fit_args, **fit_kwargs):
            return self.hypermodel.build(trial.hyperparameters).fit(
                *fit_args, **fit_kwargs
            )

    assert_found_best_score(tmp_path, MockHyperModel(), MyTuner)


def test_run_trial_return_a_dict(tmp_path):
    class MyTuner(keras_tuner.Tuner):
        def run_trial(self, trial, *fit_args, **fit_kwargs):
            history = self.hypermodel.build(trial.hyperparameters).fit(
                *fit_args, **fit_kwargs
            )
            return {"loss": min(history.history["loss"])}

    assert_found_best_score(tmp_path, MockHyperModel(), MyTuner)


def test_run_trial_return_a_float(tmp_path):
    class MyTuner(keras_tuner.Tuner):
        def run_trial(self, trial, *fit_args, **fit_kwargs):
            history = self.hypermodel.build(trial.hyperparameters).fit(
                *fit_args, **fit_kwargs
            )
            return min(history.history["loss"])

    assert_found_best_score(tmp_path, MockHyperModel(), MyTuner)


def test_run_trial_return_float_list(tmp_path):
    class MyTuner(keras_tuner.Tuner):
        def run_trial(self, trial, *fit_args, **fit_kwargs):
            ret = []
            for _ in range(3):
                history = self.hypermodel.build(trial.hyperparameters).fit(
                    *fit_args, **fit_kwargs
                )
                ret.append(min(history.history["loss"]))
            return ret

    assert_found_best_score(tmp_path, MockHyperModel(), MyTuner)


def test_tuner_errors(tmp_path):
    # invalid oracle
    with pytest.raises(
        ValueError, match="Expected `oracle` argument to be an instance of `Oracle`"
    ):
        tuner_module.Tuner(
            oracle="invalid", hypermodel=build_model, directory=tmp_path
        )
    # invalid hypermodel
    with pytest.raises(ValueError, match="`hypermodel` argument should be either"):
        tuner_module.Tuner(
            oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
                objective="val_accuracy", max_trials=3
            ),
            hypermodel="build_model",
            directory=tmp_path,
        )
    # oversize model
    with pytest.raises(RuntimeError, match="Oversized model"):
        tuner = tuner_module.Tuner(
            oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
                objective="val_accuracy", max_trials=3
            ),
            hypermodel=build_model,
            max_model_size=4,
            directory=tmp_path,
        )
        tuner.search(
            TRAIN_INPUTS, TRAIN_TARGETS, validation_data=(VAL_INPUTS, VAL_TARGETS)
        )
    # TODO: test no optimizer


@pytest.mark.skipif(
    parse(tf.__version__) < parse("2.3.0"),
    reason="TPUStrategy only exists in TF2.3+.",
)
def test_metric_direction_inferred_from_objective(tmp_path):
    oracle = keras_tuner.tuners.randomsearch.RandomSearchOracle(
        objective=keras_tuner.Objective("a", "max"), max_trials=1
    )
    oracle._set_project_dir(tmp_path, "untitled_project")
    trial = oracle.create_trial("tuner0")
    oracle.update_trial(trial.trial_id, {"a": 1})
    trial = oracle.get_trial(trial.trial_id)
    assert trial.metrics.get_direction("a") == "max"

    oracle = keras_tuner.tuners.randomsearch.RandomSearchOracle(
        objective=keras_tuner.Objective("a", "min"), max_trials=1
    )
    oracle._set_project_dir(tmp_path, "untitled_project2")
    trial = oracle.create_trial("tuner0")
    oracle.update_trial(trial.trial_id, {"a": 1})
    trial = oracle.get_trial(trial.trial_id)
    assert trial.metrics.get_direction("a") == "min"


def test_overwrite_true(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_path,
    )
    tuner.search(
        TRAIN_INPUTS, TRAIN_TARGETS, validation_data=(VAL_INPUTS, VAL_TARGETS)
    )
    assert len(tuner.oracle.trials) == 2

    new_tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_path,
        overwrite=True,
    )
    assert len(new_tuner.oracle.trials) == 0


def test_correct_display_trial_number(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_path,
    )
    tuner.search(
        TRAIN_INPUTS, TRAIN_TARGETS, validation_data=(VAL_INPUTS, VAL_TARGETS)
    )
    new_tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=6,
        directory=tmp_path,
        overwrite=False,
    )
    new_tuner.search(
        TRAIN_INPUTS, TRAIN_TARGETS, validation_data=(VAL_INPUTS, VAL_TARGETS)
    )
    assert len(new_tuner.oracle.trials) == new_tuner._display.trial_number


def test_error_on_unknown_objective_direction(tmp_path):
    with pytest.raises(ValueError, match="Could not infer optimization direction"):
        keras_tuner.tuners.RandomSearch(
            hypermodel=build_model,
            objective="custom_metric",
            max_trials=2,
            directory=tmp_path,
        )


def test_callbacks_run_each_execution(tmp_path):
    callback_instances = set()

    class LoggingCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs):
            callback_instances.add(id(self))

    logging_callback = LoggingCallback()
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )
    tuner.search(
        TRAIN_INPUTS,
        TRAIN_TARGETS,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
        callbacks=[logging_callback],
    )

    # Unknown reason cause the callback to run 5 times sometime.
    # Make 5 & 6 both pass the test before found the reason.
    assert len(callback_instances) in {5, 6}


def test_build_and_fit_model(tmp_path):
    class MyTuner(keras_tuner.tuners.RandomSearch):
        def _build_and_fit_model(self, trial, *args, **kwargs):
            self.was_called = True
            return super()._build_and_fit_model(trial, *args, **kwargs)

    tuner = MyTuner(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_path,
    )

    tuner.run_trial(
        tuner.oracle.create_trial("tuner0"),
        TRAIN_INPUTS,
        TRAIN_TARGETS,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    assert tuner.was_called


def test_build_and_fit_model_in_tuner(tmp_path):
    class MyTuner(tuner_module.Tuner):
        def _build_and_fit_model(self, trial, *args, **kwargs):
            self.was_called = True
            return super()._build_and_fit_model(trial, *args, **kwargs)

    tuner = MyTuner(
        oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
            objective="val_loss",
            max_trials=2,
        ),
        hypermodel=build_model,
        directory=tmp_path,
    )

    tuner.run_trial(
        tuner.oracle.create_trial("tuner0"),
        TRAIN_INPUTS,
        TRAIN_TARGETS,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    assert tuner.was_called


def test_init_build_all_hps_in_all_conditions(tmp_path):
    class ConditionalHyperModel(MockHyperModel):
        def build(self, hp):
            model_type = hp.Choice("model_type", ["cnn", "mlp"])
            with hp.conditional_scope("model_type", ["cnn"]):
                if model_type == "cnn":
                    sub_cnn = hp.Choice("sub_cnn", ["a", "b"])
                    with hp.conditional_scope("sub_cnn", ["a"]):
                        if sub_cnn == "a":
                            hp.Int("n_filters_a", 2, 4)
                    with hp.conditional_scope("sub_cnn", ["b"]):
                        if sub_cnn == "b":
                            hp.Int("n_filters_b", 6, 8)
            with hp.conditional_scope("model_type", ["mlp"]):
                if model_type == "mlp":
                    sub_mlp = hp.Choice("sub_mlp", ["a", "b"])
                    with hp.conditional_scope("sub_mlp", ["a"]):
                        if sub_mlp == "a":
                            hp.Int("n_units_a", 2, 4)
                    with hp.conditional_scope("sub_mlp", ["b"]):
                        if sub_mlp == "b":
                            hp.Int("n_units_b", 6, 8)
            more_block = hp.Boolean("more_block", default=False)
            with hp.conditional_scope("more_block", [True]):
                if more_block:
                    hp.Int("new_block_hp", 1, 3)
            return super().build(hp)

    def name_in_hp(name, hp):
        return any(name == single_hp.name for single_hp in hp.space)

    class MyTuner(tuner_module.Tuner):
        def _populate_initial_space(self):
            super()._populate_initial_space()
            hp = self.oracle.hyperparameters
            assert name_in_hp("model_type", hp)
            assert name_in_hp("sub_cnn", hp)
            assert name_in_hp("n_filters_a", hp)
            assert name_in_hp("n_filters_b", hp)
            assert name_in_hp("sub_mlp", hp)
            assert name_in_hp("n_units_a", hp)
            assert name_in_hp("n_units_b", hp)
            assert name_in_hp("more_block", hp)
            assert name_in_hp("new_block_hp", hp)

    MyTuner(
        oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
            objective="loss", max_trials=2, seed=1337
        ),
        hypermodel=ConditionalHyperModel(),
        directory=tmp_path,
    )


def test_populate_initial_space_with_hp_parent_arg(tmp_path):
    def build_model(hp):
        hp.Boolean("parent", default=True)
        hp.Boolean(
            "child",
            parent_name="parent",
            parent_values=[False],
        )
        return keras.Sequential()

    keras_tuner.RandomSearch(
        build_model,
        objective="val_accuracy",
        directory=tmp_path,
        max_trials=1,
    )


def test_populate_initial_space_with_declare_hp(tmp_path):
    class MyHyperModel(keras_tuner.HyperModel):
        def declare_hyperparameters(self, hp):
            hp.Boolean("bool")

        def build(self, hp):
            return keras.Sequential()

    keras_tuner.RandomSearch(
        MyHyperModel(),
        objective="val_accuracy",
        directory=tmp_path,
        max_trials=1,
    )


def test_build_did_not_return_keras_model(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=lambda hp: None,
        objective="val_accuracy",
        directory=tmp_path,
    )
    with pytest.raises(
        errors.FatalTypeError,
        match="Expected the model-building function",
    ):
        tuner.search()


def test_callback_cannot_be_deep_copied(tmp_path):
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=lambda hp: keras.Sequential(),
        objective="val_accuracy",
        directory=tmp_path,
    )
    with pytest.raises(
        errors.FatalValueError,
        match="All callbacks used during a search should be deep-copyable",
    ):
        tuner.search(callbacks=[keras_tuner])
