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

import pytest
import os

import numpy as np

from tensorflow import keras
import kerastuner


INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test', numbered=True)


def build_model(hp):
    inputs = keras.Input(shape=(INPUT_DIM,))
    x = inputs
    for i in range(hp.Int('num_layers', 1, 4)):
        x = keras.layers.Dense(
            units=hp.Int('units_' + str(i), 5, 9, 1, default=6),
            activation='relu')(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def build_subclass_model(hp):
    class MyModel(keras.Model):
        def build(self, _):
            self.layer = keras.layers.Dense(
                NUM_CLASSES, activation='softmax')

        def call(self, x):
            x = x + hp.Float('bias', 0, 10)
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
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


class ExampleHyperModel(kerastuner.HyperModel):

    def build(self, hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.Int('num_layers', 1, 4)):
            x = keras.layers.Dense(
                units=hp.Int('units_' + str(i), 5, 9, 1, default=6),
                activation='relu')(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model


def test_basic_tuner_attributes(tmp_dir):
    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir)

    assert tuner.oracle.objective.name == 'val_accuracy'
    assert tuner.oracle.max_trials == 2
    assert tuner.executions_per_trial == 3
    assert tuner.directory == tmp_dir
    assert tuner.hypermodel.__class__.__name__ == 'DefaultHyperModel'
    assert len(tuner.oracle.hyperparameters.space) == 3  # default search space
    assert len(tuner.oracle.hyperparameters.values) == 3  # default search space

    tuner.search_space_summary()

    tuner.search(x=TRAIN_INPUTS,
                 y=TRAIN_TARGETS,
                 epochs=2,
                 validation_data=(VAL_INPUTS, VAL_TARGETS))

    tuner.results_summary()

    assert len(tuner.oracle.trials) == 2
    assert os.path.exists(os.path.join(tmp_dir, 'untitled_project'))


def test_callbacks_in_fit_kwargs(tmp_dir):
    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir)
    tuner.search(x=TRAIN_INPUTS,
                 y=TRAIN_TARGETS,
                 epochs=2,
                 validation_data=(VAL_INPUTS, VAL_TARGETS),
                 callbacks=[keras.callbacks.EarlyStopping(),
                            keras.callbacks.TensorBoard(tmp_dir)])
    assert len(tuner.oracle.trials) == 2


def test_hypermodel_with_dynamic_space(tmp_dir):
    hypermodel = ExampleHyperModel()
    tuner = kerastuner.tuners.RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir)

    assert tuner.hypermodel == hypermodel

    tuner.search_space_summary()

    tuner.search(x=TRAIN_INPUTS,
                 y=TRAIN_TARGETS,
                 epochs=2,
                 validation_data=(VAL_INPUTS, VAL_TARGETS))

    tuner.results_summary()

    assert len(tuner.oracle.trials) == 2


def test_override_compile(tmp_dir):
    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_mse',
        max_trials=2,
        executions_per_trial=1,
        metrics=['mse', 'accuracy'],
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop',
        directory=tmp_dir)

    assert tuner.oracle.objective.name == 'val_mse'
    assert tuner.optimizer == 'rmsprop'
    assert tuner.loss == 'sparse_categorical_crossentropy'
    assert tuner.metrics == ['mse', 'accuracy']

    tuner.search_space_summary()

    tuner.search(x=TRAIN_INPUTS,
                 y=TRAIN_TARGETS,
                 epochs=2,
                 validation_data=(VAL_INPUTS, VAL_TARGETS))

    tuner.results_summary()

    model = tuner._build_model(tuner.oracle.hyperparameters)
    tuner._compile_model(model)
    assert model.optimizer.__class__.__name__ == 'RMSprop'
    assert model.loss == 'sparse_categorical_crossentropy'
    assert len(model.metrics) == 2
    assert model.metrics[0]._fn.__name__ == 'mean_squared_error'
    assert model.metrics[1]._fn.__name__ == 'sparse_categorical_accuracy'


def test_static_space(tmp_dir):

    def build_model_static(hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.get('num_layers')):
            x = keras.layers.Dense(
                units=hp.get('units_' + str(i)),
                activation='relu')(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.get('learning_rate')),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model

    hp = kerastuner.HyperParameters()
    hp.Int('num_layers', 1, 3, 1, default=2)
    hp.Int('units_0', 4, 6, 1, default=5)
    hp.Int('units_1', 4, 6, 1, default=5)
    hp.Choice('learning_rate', [0.01, 0.001])
    tuner = kerastuner.tuners.RandomSearch(
        build_model_static,
        objective='val_accuracy',
        max_trials=4,
        directory=tmp_dir,
        hyperparameters=hp,
        allow_new_entries=False)

    assert tuner.oracle.hyperparameters == hp
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS))
    assert len(tuner.oracle.trials) == 4


def test_static_space_errors(tmp_dir):

    def build_model_static(hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.get('num_layers')):
            x = keras.layers.Dense(
                units=hp.get('units_' + str(i)),
                activation='relu')(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float('learning_rate', 1e-5, 1e-2)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model

    hp = kerastuner.HyperParameters()
    hp.Int('num_layers', 1, 3, 1, default=2)
    hp.Int('units_0', 4, 6, 1, default=5)
    hp.Int('units_1', 4, 6, 1, default=5)

    with pytest.raises(RuntimeError, match='`allow_new_entries` is `False`'):
        tuner = kerastuner.tuners.RandomSearch(
            build_model_static,
            objective='val_accuracy',
            max_trials=2,
            directory=tmp_dir,
            hyperparameters=hp,
            allow_new_entries=False)
        tuner.search(
            x=TRAIN_INPUTS,
            y=TRAIN_TARGETS,
            epochs=2,
            validation_data=(VAL_INPUTS, VAL_TARGETS))

    def build_model_static_invalid(hp):
        inputs = keras.Input(shape=(INPUT_DIM,))
        x = inputs
        for i in range(hp.get('num_layers')):
            x = keras.layers.Dense(
                units=hp.get('units_' + str(i)),
                activation='relu')(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float('learning_rate', 0.001, 0.008, 0.001)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model

    with pytest.raises(RuntimeError,
                       match='`allow_new_entries` is `False`'):
        tuner = kerastuner.tuners.RandomSearch(
            build_model_static_invalid,
            objective='val_accuracy',
            max_trials=2,
            directory=tmp_dir,
            hyperparameters=hp,
            allow_new_entries=False)
        tuner.search(
            x=TRAIN_INPUTS,
            y=TRAIN_TARGETS,
            epochs=2,
            validation_data=(VAL_INPUTS, VAL_TARGETS))


def test_restricted_space_using_defaults(tmp_dir):
    hp = kerastuner.HyperParameters()
    hp.Int('num_layers', 1, 5, 1, default=2)
    hp.Choice('learning_rate', [0.01, 0.001, 0.0001])

    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=4,
        directory=tmp_dir,
        hyperparameters=hp,
        allow_new_entries=True,
        tune_new_entries=False)

    assert len(tuner.oracle.hyperparameters.space) == 2
    new_lr = [p for p in tuner.oracle.hyperparameters.space
              if p.name == 'learning_rate'][0]
    assert new_lr.values == [0.01, 0.001, 0.0001]

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=1,
        validation_data=(VAL_INPUTS, VAL_TARGETS))

    assert len(tuner.oracle.trials) == 4
    assert len(tuner.oracle.hyperparameters.space) == 2  # Nothing added
    for trial in tuner.oracle.trials.values():
        assert len(trial.hyperparameters.space) == 2  # Nothing added


def test_restricted_space_with_custom_defaults(tmp_dir):
    hp = kerastuner.HyperParameters()
    hp.Int('num_layers', 1, 3, 1, default=2)
    hp.Choice('learning_rate', [0.01, 0.001, 0.0001])
    hp.Fixed('units_0', 4)
    hp.Fixed('units_1', 3)

    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=4,
        directory=tmp_dir,
        hyperparameters=hp,
        allow_new_entries=True,
        tune_new_entries=False)

    assert len(tuner.oracle.hyperparameters.space) == 4
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=1,
        validation_data=(VAL_INPUTS, VAL_TARGETS))

    assert len(tuner.oracle.trials) == 4


def test_reparameterized_space(tmp_dir):
    hp = kerastuner.HyperParameters()
    hp.Int('num_layers', 1, 3, 1, default=2)
    hp.Choice('learning_rate', [0.01, 0.001, 0.0001])

    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        seed=1337,
        objective='val_accuracy',
        max_trials=4,
        directory=tmp_dir,
        hyperparameters=hp,
        allow_new_entries=True,
        tune_new_entries=True)

    # Initial build model adds to the space.
    assert len(tuner.oracle.hyperparameters.space) == 4
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=1,
        validation_data=(VAL_INPUTS, VAL_TARGETS))

    assert len(tuner.oracle.trials) == 4
    assert len(tuner.oracle.hyperparameters.space) == 4


def test_get_best_models(tmp_dir):
    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=4,
        directory=tmp_dir)

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS))

    models = tuner.get_best_models(2)
    assert len(models) == 2
    assert isinstance(models[0], keras.Model)
    assert isinstance(models[1], keras.Model)


def test_saving_and_reloading(tmp_dir):

    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=4,
        executions_per_trial=2,
        directory=tmp_dir)

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS))

    new_tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=4,
        executions_per_trial=2,
        directory=tmp_dir)
    new_tuner.reload()

    assert len(new_tuner.oracle.trials) == 4

    new_tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS))


def test_subclass_model(tmp_dir):
    tuner = kerastuner.tuners.RandomSearch(
        build_subclass_model,
        objective='val_accuracy',
        max_trials=2,
        directory=tmp_dir)

    tuner.search_space_summary()

    tuner.search(x=TRAIN_INPUTS,
                 y=TRAIN_TARGETS,
                 epochs=2,
                 validation_data=(VAL_INPUTS, VAL_TARGETS))

    tuner.results_summary()
    assert len(tuner.oracle.trials) == 2


def test_update_trial(tmp_dir):
    class MyOracle(kerastuner.Oracle):

        def _populate_space(self, _):
            values = {p.name: p.random_sample()
                      for p in self.hyperparameters.space}
            return {'values': values, 'status': 'RUNNING'}

        def update_trial(self, trial_id, metrics, step=0):
            if step == 3:
                trial = self.trials[trial_id]
                trial.status = "STOPPED"
                return trial.status
            return super(MyOracle, self).update_trial(
                trial_id, metrics, step)

    my_oracle = MyOracle(
        objective='val_accuracy',
        max_trials=2)
    tuner = kerastuner.Tuner(
        oracle=my_oracle,
        hypermodel=build_model,
        directory=tmp_dir)
    tuner.search(x=TRAIN_INPUTS,
                 y=TRAIN_TARGETS,
                 epochs=5,
                 validation_data=(VAL_INPUTS, VAL_TARGETS))

    assert len(my_oracle.trials) == 2

    for trial in my_oracle.trials.values():
        # Test that early stopping worked.
        assert len(trial.metrics.get_history('val_accuracy')) == 3


def test_objective_formats():
    obj = kerastuner.engine.oracle._format_objective('accuracy')
    assert obj == kerastuner.Objective('accuracy', 'max')

    obj = kerastuner.engine.oracle._format_objective(
        kerastuner.Objective('score', 'min'))
    assert obj == kerastuner.Objective('score', 'min')

    obj = kerastuner.engine.oracle._format_objective([
        kerastuner.Objective('score', 'max'),
        kerastuner.Objective('loss', 'min')])
    assert obj == [kerastuner.Objective('score', 'max'),
                   kerastuner.Objective('loss', 'min')]

    obj = kerastuner.engine.oracle._format_objective([
        'accuracy', 'loss'])
    assert obj == [kerastuner.Objective('accuracy', 'max'),
                   kerastuner.Objective('loss', 'min')]
