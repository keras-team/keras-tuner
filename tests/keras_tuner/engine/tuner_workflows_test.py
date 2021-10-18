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
from tensorboard.plugins.hparams import api as hparams_api
from tensorflow import keras

import keras_tuner

INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))


@pytest.fixture(scope="function")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("integration_test", numbered=True)


def build_model(hp):
    inputs = keras.Input(shape=(INPUT_DIM,))
    x = inputs
    for i in range(hp.Int("num_layers", 1, 4)):
        x = keras.layers.Dense(
            units=hp.Int("units_" + str(i), 5, 9, 1, default=6), activation="relu"
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
                units=hp.Int("units_" + str(i), 5, 9, 1, default=6),
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


def test_basic_tuner_attributes(tmp_dir):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir,
    )

    assert tuner.oracle.objective.name == "val_accuracy"
    assert tuner.oracle.max_trials == 2
    assert tuner.executions_per_trial == 3
    assert tuner.directory == tmp_dir
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
    assert os.path.exists(os.path.join(str(tmp_dir), "untitled_project"))


def test_no_hypermodel_with_objective(tmp_dir):
    class MyTuner(keras_tuner.tuners.RandomSearch):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            return {"val_loss": hp.Float("value", 0, 10)}

    tuner = MyTuner(
        objective="val_loss",
        max_trials=2,
        directory=tmp_dir,
    )
    tuner.search()

    assert len(tuner.oracle.trials) == 2


def test_no_objective_with_hypermodel(tmp_dir):
    class MyHyperModel(ExampleHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            return hp.Float("value", 0, 10)

    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=MyHyperModel(),
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir,
    )
    tuner.search()

    assert len(tuner.oracle.trials) == 2


def test_no_hypermodel_no_objective(tmp_dir):
    class MyTuner(keras_tuner.tuners.RandomSearch):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            return hp.Float("value", 0, 10)

    tuner = MyTuner(
        objective="val_loss",
        max_trials=2,
        directory=tmp_dir,
    )
    tuner.search()

    assert len(tuner.oracle.trials) == 2


def test_no_hypermodel_without_override_run_trial_error(tmp_dir):
    with pytest.raises(ValueError, match="Received `hypermodel=None`"):
        keras_tuner.tuners.RandomSearch(
            max_trials=2,
            executions_per_trial=3,
            directory=tmp_dir,
        )


def test_no_objective_return_not_single_value_error(tmp_dir):
    class MyHyperModel(ExampleHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            return {"val_loss": hp.Float("value", 0, 10)}

    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=MyHyperModel(),
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir,
    )

    with pytest.raises(TypeError, match="to be a single float"):
        tuner.search()


def test_callbacks_in_fit_kwargs(tmp_dir):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir,
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
                keras.callbacks.TensorBoard(tmp_dir),
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
            "ModelCheckpoint",
        ]


def test_hypermodel_with_dynamic_space(tmp_dir):
    hypermodel = ExampleHyperModel()
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir,
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


def test_override_compile(tmp_dir):
    class MyHyperModel(ExampleHyperModel):
        def fit(self, hp, model, *args, **kwargs):
            history = super().fit(hp, model, *args, **kwargs)
            assert model.optimizer.__class__.__name__ == "RMSprop"
            assert model.loss == "sparse_categorical_crossentropy"
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
        loss="sparse_categorical_crossentropy",
        optimizer="rmsprop",
        directory=tmp_dir,
    )

    assert tuner.oracle.objective.name == "val_mse"
    assert tuner.optimizer == "rmsprop"
    assert tuner.loss == "sparse_categorical_crossentropy"
    assert tuner.metrics == ["mse", "accuracy"]

    tuner.search_space_summary()
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )
    tuner.results_summary()
    tuner.hypermodel.build(tuner.oracle.hyperparameters)


def test_static_space(tmp_dir):
    def build_model_static(hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.get("num_layers")):
            x = keras.layers.Dense(
                units=hp.get("units_" + str(i)), activation="relu"
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
        directory=tmp_dir,
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


def test_static_space_errors(tmp_dir):
    def build_model_static(hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.get("num_layers")):
            x = keras.layers.Dense(
                units=hp.get("units_" + str(i)), activation="relu"
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
            directory=tmp_dir,
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
                units=hp.get("units_" + str(i)), activation="relu"
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
            directory=tmp_dir,
            hyperparameters=hp,
            allow_new_entries=False,
        )
        tuner.search(
            x=TRAIN_INPUTS,
            y=TRAIN_TARGETS,
            epochs=2,
            validation_data=(VAL_INPUTS, VAL_TARGETS),
        )


def test_restricted_space_using_defaults(tmp_dir):
    hp = keras_tuner.HyperParameters()
    hp.Int("num_layers", 1, 5, 1, default=2)
    hp.Choice("learning_rate", [0.01, 0.001, 0.0001])

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=4,
        directory=tmp_dir,
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


def test_restricted_space_with_custom_defaults(tmp_dir):
    hp = keras_tuner.HyperParameters()
    hp.Int("num_layers", 1, 3, 1, default=2)
    hp.Choice("learning_rate", [0.01, 0.001, 0.0001])
    hp.Fixed("units_0", 4)
    hp.Fixed("units_1", 3)

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=4,
        directory=tmp_dir,
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


def test_reparameterized_space(tmp_dir):
    hp = keras_tuner.HyperParameters()
    hp.Int("num_layers", 1, 3, 1, default=3)
    hp.Choice("learning_rate", [0.01, 0.001, 0.0001])

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        seed=1337,
        objective="val_accuracy",
        max_trials=4,
        directory=tmp_dir,
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


def test_get_best_models(tmp_dir):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model, objective="val_accuracy", max_trials=4, directory=tmp_dir
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


def test_saving_and_reloading(tmp_dir):

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=4,
        executions_per_trial=2,
        directory=tmp_dir,
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
        directory=tmp_dir,
    )
    new_tuner.reload()

    assert len(new_tuner.oracle.trials) == 4

    new_tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )


def test_subclass_model(tmp_dir):
    tuner = keras_tuner.tuners.RandomSearch(
        build_subclass_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_dir,
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


def test_subclass_model_loading(tmp_dir):
    tuner = keras_tuner.tuners.RandomSearch(
        build_subclass_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_dir,
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


def test_update_trial(tmp_dir):
    class MyOracle(keras_tuner.Oracle):
        def populate_space(self, _):
            values = {p.name: p.random_sample() for p in self.hyperparameters.space}
            return {"values": values, "status": "RUNNING"}

        def update_trial(self, trial_id, metrics, step=0):
            if step == 3:
                trial = self.trials[trial_id]
                trial.status = "STOPPED"
                return trial.status
            return super(MyOracle, self).update_trial(trial_id, metrics, step)

    my_oracle = MyOracle(objective="val_accuracy", max_trials=2)
    tuner = keras_tuner.Tuner(
        oracle=my_oracle, hypermodel=build_model, directory=tmp_dir
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


def test_objective_formats():
    obj = keras_tuner.engine.oracle._format_objective("accuracy")
    assert obj == keras_tuner.Objective("accuracy", "max")

    obj = keras_tuner.engine.oracle._format_objective(
        keras_tuner.Objective("score", "min")
    )
    assert obj == keras_tuner.Objective("score", "min")

    obj = keras_tuner.engine.oracle._format_objective(
        [keras_tuner.Objective("score", "max"), keras_tuner.Objective("loss", "min")]
    )
    assert obj == [
        keras_tuner.Objective("score", "max"),
        keras_tuner.Objective("loss", "min"),
    ]

    obj = keras_tuner.engine.oracle._format_objective(["accuracy", "loss"])
    assert obj == [
        keras_tuner.Objective("accuracy", "max"),
        keras_tuner.Objective("loss", "min"),
    ]


def test_tunable_false_hypermodel(tmp_dir):
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
        objective="val_loss", hypermodel=build_model, max_trials=4, directory=tmp_dir
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


def test_get_best_hyperparameters(tmp_dir):
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
        directory=tmp_dir,
    )

    tuner.oracle.trials = {trial1.trial_id: trial1, trial2.trial_id: trial2}

    hps = tuner.get_best_hyperparameters()[0]
    assert hps["a"] == 1


def test_reloading_error_message(tmp_dir):
    shared_dir = tmp_dir

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


def test_search_logging_verbosity(tmp_dir):
    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir,
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
        repr(expected_hparams[x]) for x in expected_hparams.keys()
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
