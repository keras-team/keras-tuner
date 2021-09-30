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

import copy
import os

from tensorboard.plugins.hparams import api as hparams_api

from keras_tuner.engine import tuner
from keras_tuner.engine import tuner_utils


class SingleExecutionTuner(tuner.Tuner):
    """A `Tuner` class running each model once and reporting every epoch.

    Args:
        oracle: An Oracle instance.
        hypermodel: A `HyperModel` instance (or callable that takes
            hyperparameters and returns a `Model` instance).
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(self, oracle, hypermodel, **kwargs):
        if "executions_per_trial" in kwargs:
            raise ValueError(
                "executions_per_trial is not supported " "in SingleExecutionTuner."
            )
        super().__init__(oracle, hypermodel, **kwargs)

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """Evaluates a set of hyperparameter values.

        This method is called multiple times during `search` to build and
        evaluate the models with different hyperparameters.

        The method is responsible for reporting metrics related to the `Trial`
        to the `Oracle` via `self.oracle.update_trial`.

        Args:
            trial: A `Trial` instance that contains the information needed to
                run this trial. `Hyperparameters` can be accessed via
                `trial.hyperparameters`.
            *fit_args: Positional arguments passed by `search`.
            **fit_kwargs: Keyword arguments passed by `search`.
        """
        # Handle any callbacks passed to `fit`.
        copied_fit_kwargs = copy.copy(fit_kwargs)
        callbacks = fit_kwargs.pop("callbacks", [])
        callbacks = self._deepcopy_callbacks(callbacks)
        self._configure_tensorboard_dir(callbacks, trial)
        # `TunerCallback` calls:
        # - `Tuner.on_epoch_begin`
        # - `Tuner.on_batch_begin`
        # - `Tuner.on_batch_end`
        # - `Tuner.on_epoch_end`
        # These methods report results to the `Oracle` and save the trained Model. If
        # you are subclassing `Tuner` to write a custom training loop, you should
        # make calls to these methods within `run_trial`.
        callbacks.append(tuner_utils.TunerCallback(self, trial))
        copied_fit_kwargs["callbacks"] = callbacks

        self._build_and_fit_model(trial, fit_args, copied_fit_kwargs)

    def save_model(self, trial_id, model, step=0):
        epoch = step
        self._checkpoint_model(model, trial_id, epoch)
        # TODO: save the top epoch checkpoints instead of last ones.
        epoch_to_delete = epoch - self._save_n_checkpoints
        best_epoch = 0
        if epoch > 0:
            # TODO: `get_best_models` would load the `best_step` checkpoint after
            # training. It would break if oracle picks a different `best_step` than
            # `metrics.get_best_step` since it might be deleted due to it was
            # not the `best_epoch` during the training.
            best_epoch = self.oracle.get_trial(trial_id).metrics.get_best_step(
                self.oracle.objective.name
            )
        if epoch > self._save_n_checkpoints and epoch_to_delete != best_epoch:
            self._delete_checkpoint(trial_id, epoch_to_delete)

    def on_epoch_end(self, trial, model, epoch, logs=None):
        """Called at the end of an epoch.

        Args:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            epoch: The current epoch number.
            logs: Dict. Metrics for this epoch. This should include
              the value of the objective for this epoch.
        """
        self.save_model(trial.trial_id, model, step=epoch)
        # Report intermediate metrics to the `Oracle`.
        status = self.oracle.update_trial(trial.trial_id, metrics=logs, step=epoch)
        trial.status = status
        if trial.status == "STOPPED":
            model.stop_training = True

    def _configure_tensorboard_dir(self, callbacks, trial):
        for callback in callbacks:
            if callback.__class__.__name__ == "TensorBoard":
                # Patch TensorBoard log_dir and add HParams KerasCallback
                logdir = self._get_tensorboard_dir(callback.log_dir, trial.trial_id)
                callback.log_dir = logdir
                hparams = tuner_utils.convert_hyperparams_to_hparams(
                    trial.hyperparameters
                )
                callbacks.append(
                    hparams_api.KerasCallback(
                        writer=logdir, hparams=hparams, trial_id=trial.trial_id
                    )
                )

    def _get_tensorboard_dir(self, logdir, trial_id):
        return os.path.join(str(logdir), str(trial_id))
