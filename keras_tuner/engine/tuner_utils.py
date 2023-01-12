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
"""Utilities for Tuner class."""


import collections
import math
import statistics
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hparams_api
from tensorflow import keras

from keras_tuner import errors
from keras_tuner import utils
from keras_tuner.distribute import utils as ds_utils
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import objective as obj_module


class TunerStats:
    """Track tuner statistics."""

    def __init__(self):
        self.num_generated_models = 0  # overall number of instances generated
        self.num_invalid_models = 0  # how many models didn't work
        self.num_oversized_models = 0  # num models with params> max_params

    def summary(self, extended=False):
        print("Tuning stats")
        for setting, value in self.get_config():
            print(f"{setting}:", value)

    def get_config(self):
        return {
            "num_generated_models": self.num_generated_models,
            "num_invalid_models": self.num_invalid_models,
            "num_oversized_models": self.num_oversized_models,
        }

    @classmethod
    def from_config(cls, config):
        stats = cls()
        stats.num_generated_models = config["num_generated_models"]
        stats.num_invalid_models = config["num_invalid_models"]
        stats.num_oversized_models = config["num_oversized_models"]
        return stats


def get_max_epochs_and_steps(fit_args, fit_kwargs):
    if fit_args:
        x = tf.nest.flatten(fit_args)[0]
    else:
        x = tf.nest.flatten(fit_kwargs.get("x"))[0]
    batch_size = fit_kwargs.get("batch_size", 32)
    if hasattr(x, "__len__"):
        max_steps = math.ceil(float(len(x)) / batch_size)
    else:
        max_steps = fit_kwargs.get("steps")
    max_epochs = fit_kwargs.get("epochs", 1)
    return max_epochs, max_steps


class TunerCallback(keras.callbacks.Callback):
    def __init__(self, tuner, trial):
        super().__init__()
        self.tuner = tuner
        self.trial = trial

    def on_epoch_begin(self, epoch, logs=None):
        self.tuner.on_epoch_begin(self.trial, self.model, epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.tuner.on_batch_begin(self.trial, self.model, batch, logs)

    def on_batch_end(self, batch, logs=None):
        self.tuner.on_batch_end(self.trial, self.model, batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.tuner.on_epoch_end(self.trial, self.model, epoch, logs=logs)


# TODO: Add more extensive display.
class Display:
    def __init__(self, oracle, verbose=1):
        self.verbose = verbose
        self.oracle = oracle
        self.col_width = 18

        # Start time for the overall search
        self.search_start = None

        # Start time of the latest trial
        self.trial_start = None

    def on_trial_begin(self, trial):
        if self.verbose >= 1:

            self.trial_number = int(trial.trial_id) + 1
            print()
            print(f"Search: Running Trial #{self.trial_number}")
            print()

            self.trial_start = datetime.now()
            if self.search_start is None:
                self.search_start = self.trial_start

            self.show_hyperparameter_table(trial)
            print()

    def on_trial_end(self, trial):
        if self.verbose < 1:
            return
        utils.try_clear()

        time_taken_str = self.format_duration(datetime.now() - self.trial_start)
        print(f"Trial {self.trial_number} Complete [{time_taken_str}]")

        if trial.score is not None:
            print(f"{self.oracle.objective.name}: {trial.score}")

        print()
        best_trials = self.oracle.get_best_trials()
        best_score = best_trials[0].score if len(best_trials) > 0 else None
        print(f"Best {self.oracle.objective.name} So Far: {best_score}")

        time_elapsed_str = self.format_duration(datetime.now() - self.search_start)
        print(f"Total elapsed time: {time_elapsed_str}")

    def show_hyperparameter_table(self, trial):
        template = "{{0:{0}}}|{{1:{0}}}|{{2}}".format(self.col_width)
        best_trials = self.oracle.get_best_trials()
        best_trial = best_trials[0] if len(best_trials) > 0 else None
        if trial.hyperparameters.values:
            print(template.format("Value", "Best Value So Far", "Hyperparameter"))
            for hp, value in trial.hyperparameters.values.items():
                best_value = (
                    best_trial.hyperparameters.values.get(hp) if best_trial else "?"
                )
                print(
                    template.format(
                        self.format_value(value),
                        self.format_value(best_value),
                        hp,
                    )
                )
        else:
            print("default configuration")

    def format_value(self, val):
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return f"{val:.5g}"
        val_str = str(val)
        if len(val_str) > self.col_width:
            val_str = f"{val_str[:self.col_width - 3]}..."
        return val_str

    def format_duration(self, d):
        s = round(d.total_seconds())
        d = s // 86400
        s %= 86400
        h = s // 3600
        s %= 3600
        m = s // 60
        s %= 60

        if d > 0:
            return f"{d:d}d {h:02d}h {m:02d}m {s:02d}s"
        return f"{h:02d}h {m:02d}m {s:02d}s"


class SaveBestEpoch(keras.callbacks.Callback):
    """A Keras callback to save the model weights at the best epoch.

    Args:
        objective: An `Objective` instance.
        filepath: String. The file path to save the model weights.
    """

    def __init__(self, objective, filepath):
        super().__init__()
        self.objective = objective
        self.filepath = filepath
        if self.objective.direction == "max":
            self.best_value = float("-inf")
        else:
            self.best_value = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        if not self.objective.has_value(logs):
            # Save on every epoch if metric value is not in the logs. Either no
            # objective is specified, or objective is computed and returned
            # after `fit()`.
            self._save_model()
            return
        current_value = self.objective.get_value(logs)
        if self.objective.better_than(current_value, self.best_value):
            self.best_value = current_value
            self._save_model()

    def _save_model(self):
        # Create temporary saved model files on non-chief workers.
        write_filepath = ds_utils.write_filepath(
            self.filepath, self.model.distribute_strategy
        )
        self.model.save_weights(write_filepath)
        # Remove temporary saved model files on non-chief workers.
        ds_utils.remove_temp_dir_with_filepath(
            write_filepath, self.model.distribute_strategy
        )


def average_metrics_dicts(metrics_dicts):
    """Averages the metrics dictionaries to one metrics dictionary."""
    metrics = collections.defaultdict(list)
    for metrics_dict in metrics_dicts:
        for metric_name, metric_value in metrics_dict.items():
            metrics[metric_name].append(metric_value)
    averaged_metrics = {
        metric_name: np.mean(metric_values)
        for metric_name, metric_values in metrics.items()
    }

    return averaged_metrics


def _get_best_value_and_best_epoch_from_history(history, objective):
    # A dictionary to record the metric values through epochs.
    # Usage: epoch_metric[epoch_number][metric_name] == metric_value
    epoch_metrics = collections.defaultdict(dict)
    for metric_name, epoch_values in history.history.items():
        for epoch, value in enumerate(epoch_values):
            epoch_metrics[epoch][metric_name] = value
    best_epoch = 0
    for epoch, metrics in epoch_metrics.items():
        objective_value = objective.get_value(metrics)
        # Support multi-objective.
        if objective.name not in metrics:
            metrics[objective.name] = objective_value
        best_value = epoch_metrics[best_epoch][objective.name]
        if objective.better_than(objective_value, best_value):
            best_epoch = epoch
    return epoch_metrics[best_epoch], best_epoch


def convert_to_metrics_dict(results, objective):
    """Convert any supported results type to a metrics dictionary."""
    # List of multiple exectuion results to be averaged.
    # Check this case first to deal each case individually to check for errors.
    if isinstance(results, list):
        return average_metrics_dicts(
            [convert_to_metrics_dict(elem, objective) for elem in results]
        )

    # Single value.
    if isinstance(results, (int, float, np.floating)):
        return {objective.name: float(results)}

    # A dictionary.
    if isinstance(results, dict):
        return results

    # A History.
    if isinstance(results, keras.callbacks.History):
        best_value, _ = _get_best_value_and_best_epoch_from_history(
            results, objective
        )
        return best_value


def validate_trial_results(results, objective, func_name):
    if isinstance(results, list):
        for elem in results:
            validate_trial_results(elem, objective, func_name)
        return

    # Single value.
    if isinstance(results, (int, float, np.floating)):
        return

    # None
    if results is None:
        raise errors.FatalTypeError(
            f"The return value of {func_name} is None. "
            "Did you forget to return the metrics? "
        )

    # objective left unspecified,
    # and objective value is not a single float.
    if isinstance(objective, obj_module.DefaultObjective) and not (
        isinstance(results, dict) and objective.name in results
    ):
        raise errors.FatalTypeError(
            f"Expected the return value of {func_name} to be "
            "a single float when `objective` is left unspecified. "
            f"Recevied return value: {results} of type {type(results)}."
        )

    # A dictionary.
    if isinstance(results, dict):
        if objective.name not in results:
            raise errors.FatalValueError(
                f"Expected the returned dictionary from {func_name} to have "
                f"the specified objective, {objective.name}, "
                "as one of the keys. "
                f"Received: {results}."
            )
        return

    # A History.
    if isinstance(results, keras.callbacks.History):
        return

    # Other unsupported types.
    raise errors.FatalTypeError(
        f"Expected the return value of {func_name} to be "
        "one of float, dict, keras.callbacks.History, "
        "or a list of one of these types. "
        f"Recevied return value: {results} of type {type(results)}."
    )


def get_best_step(results, objective):
    # Average the best epochs if multiple executions.
    if isinstance(results, list):
        return int(
            statistics.mean([get_best_step(elem, objective) for elem in results])
        )

    # A History.
    if isinstance(results, keras.callbacks.History):
        _, best_epoch = _get_best_value_and_best_epoch_from_history(
            results, objective
        )
        return best_epoch

    return 0


def convert_hyperparams_to_hparams(hyperparams):
    """Converts KerasTuner HyperParameters to TensorBoard HParams."""
    hparams = {}
    for hp in hyperparams.space:
        hparams_value = {}
        try:
            hparams_value = hyperparams.get(hp.name)
        except ValueError:
            continue

        hparams_domain = {}
        if isinstance(hp, hp_module.Choice):
            hparams_domain = hparams_api.Discrete(hp.values)
        elif isinstance(hp, hp_module.Int):
            if hp.step is not None and hp.step != 1:
                # Note: `hp.max_value` is inclusive, unlike the end index
                # of Python `range()`, which is exclusive
                values = list(range(hp.min_value, hp.max_value + 1, hp.step))
                hparams_domain = hparams_api.Discrete(values)
            else:
                hparams_domain = hparams_api.IntInterval(hp.min_value, hp.max_value)
        elif isinstance(hp, hp_module.Float):
            if hp.step is not None:
                # Note: `hp.max_value` is inclusive, unlike the end index
                # of Numpy's arange(), which is exclusive
                values = np.arange(
                    hp.min_value, hp.max_value + 1e-7, step=hp.step
                ).tolist()
                hparams_domain = hparams_api.Discrete(values)
            else:
                hparams_domain = hparams_api.RealInterval(hp.min_value, hp.max_value)
        elif isinstance(hp, hp_module.Boolean):
            hparams_domain = hparams_api.Discrete([True, False])
        elif isinstance(hp, hp_module.Fixed):
            hparams_domain = hparams_api.Discrete([hp.value])
        else:
            raise ValueError(f"`HyperParameter` type not recognized: {hp}")

        hparams_key = hparams_api.HParam(hp.name, hparams_domain)
        hparams[hparams_key] = hparams_value

    return hparams
