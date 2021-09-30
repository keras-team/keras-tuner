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
"Keras Tuner class."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hparams_api
from tensorflow import keras

from keras_tuner.engine import base_tuner
from keras_tuner.engine import hypermodel as hm_module
from keras_tuner.engine import tuner_utils


class Tuner(base_tuner.BaseTuner):
    """Tuner class for Keras models.

    This is the base `Tuner` class for all tuners for Keras models. It manages
    the building, training, evaluation and saving of the Keras models. New
    tuners can be created by subclassing the class.

    Args:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class (or callable that takes
            hyperparameters and returns a Model instance).
        max_model_size: Integer, maximum number of scalars in the parameters of
            a model. Models larger than this are rejected.
        optimizer: Optional `Optimizer` instance.  May be used to override the
            `optimizer` argument in the `compile` step for the models. If the
            hypermodel does not compile the models it generates, then this
            argument must be specified.
        loss: Optional loss. May be used to override the `loss` argument in the
            `compile` step for the models. If the hypermodel does not compile
            the models it generates, then this argument must be specified.
        metrics: Optional metrics. May be used to override the `metrics`
            argument in the `compile` step for the models. If the hypermodel
            does not compile the models it generates, then this argument must
            be specified.
        distribution_strategy: Optional instance of `tf.distribute.Strategy`.
            If specified, each trial will run under this scope. For example,
            `tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])` will run
            each trial on two GPUs. Currently only single-worker strategies are
            supported.
        directory: A string, the relative path to the working directory.
        project_name: A string, the name to use as prefix for files saved by
            this `Tuner`.
        logger: Optional instance of `kerastuner.Logger` class for
            streaming logs for monitoring.
        tuner_id: Optional string, used as the ID of this `Tuner`.
        overwrite: Boolean, defaults to `False`. If `False`, reloads an
            existing project of the same name if one is found. Otherwise,
            overwrites the project.
        executions_per_trial: Integer, the number of executions (training a
            model from scratch, starting from a new initialization) to run per
            trial (model configuration). Model metrics may vary greatly
            depending on random initialization, hence it is often a good idea
            to run several executions per trial in order to evaluate the
            performance of a given set of hyperparameter values.

    Attributes:
        remaining_trials: Number of trials remaining, `None` if `max_trials` is
            not set. This is useful when resuming a previously stopped search.
    """

    def __init__(
        self,
        oracle,
        hypermodel,
        max_model_size=None,
        optimizer=None,
        loss=None,
        metrics=None,
        distribution_strategy=None,
        directory=None,
        project_name=None,
        logger=None,
        tuner_id=None,
        overwrite=False,
        executions_per_trial=1,
    ):

        # Subclasses of `KerasHyperModel` are not automatically wrapped.
        if not isinstance(hypermodel, hm_module.KerasHyperModel):
            hypermodel = hm_module.KerasHyperModel(
                hypermodel,
                max_model_size=max_model_size,
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                distribution_strategy=distribution_strategy,
            )

        super(Tuner, self).__init__(
            oracle=oracle,
            hypermodel=hypermodel,
            directory=directory,
            project_name=project_name,
            logger=logger,
            overwrite=overwrite,
        )

        if isinstance(oracle.objective, list) and len(oracle.objective) > 1:
            raise ValueError(
                "Multi-objective is not supported, found: {}".format(
                    oracle.objective
                )
            )

        self.distribution_strategy = distribution_strategy

        # Support multi-worker distribution strategies w/ distributed tuning.
        # Only the chief worker in each cluster should report results.
        if self.distribution_strategy is not None and hasattr(
            self.distribution_strategy.extended, "_in_multi_worker_mode"
        ):
            self.oracle.multi_worker = (
                self.distribution_strategy.extended._in_multi_worker_mode()
            )
            self.oracle.should_report = (
                self.distribution_strategy.extended.should_checkpoint
            )

        # Save only the last N checkpoints.
        self._save_n_checkpoints = 10

        self.tuner_id = tuner_id or self.tuner_id

        self.executions_per_trial = executions_per_trial
        # This is the `step` that will be reported to the Oracle at the end
        # of the Trial. Since intermediate results are not used, this is set
        # to 0.
        self._reported_step = 0

    def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
        """For AutoKeras to override.

        DO NOT REMOVE this function. AutoKeras overrides the function to tune
        tf.data preprocessing pipelines, preprocess the dataset to obtain
        the input shape before building the model, adapt preprocessing layers,
        and tune other fit_args and fit_kwargs.

        Args:
            trial: A `Trial` instance that contains the information needed to
                run this trial. `Hyperparameters` can be accessed via
                `trial.hyperparameters`.
            fit_args: Positional arguments passed by `search`.
            fit_kwargs: Keyword arguments passed by `search`.

        Returns:
            The fit history.
        """
        model = self.hypermodel.build(trial.hyperparameters)
        return model.fit(*fit_args, **fit_kwargs)

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self._get_checkpoint_fname(trial.trial_id, self._reported_step),
            monitor=self.oracle.objective.name,
            mode=self.oracle.objective.direction,
            save_best_only=True,
            save_weights_only=True,
        )
        original_callbacks = fit_kwargs.pop("callbacks", [])

        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(fit_kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)
            copied_fit_kwargs["callbacks"] = callbacks

            history = self._build_and_fit_model(trial, fit_args, copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == "min":
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step
        )

    def load_model(self, trial):
        model = self.hypermodel.build(trial.hyperparameters)
        # Reload best checkpoint. The Oracle scores the Trial and also
        # indicates at what epoch the best value of the objective was
        # obtained.
        best_epoch = trial.best_step
        with hm_module.maybe_distribute(self.distribution_strategy):
            model.load_weights(
                self._get_checkpoint_fname(trial.trial_id, best_epoch)
            )
        return model

    def on_batch_begin(self, trial, model, batch, logs):
        """Called at the beginning of a batch.

        Args:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            batch: The current batch number within the current epoch.
            logs: Additional metrics.
        """
        pass

    def on_batch_end(self, trial, model, batch, logs=None):
        """Called at the end of a batch.

        Args:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            batch: The current batch number within the current epoch.
            logs: Additional metrics.
        """
        pass

    def on_epoch_begin(self, trial, model, epoch, logs=None):
        """Called at the beginning of an epoch.

        Args:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            epoch: The current epoch number.
            logs: Additional metrics.
        """
        pass

    def on_epoch_end(self, trial, model, epoch, logs=None):
        # Intermediate results are not passed to the Oracle, and
        # checkpointing is handled via a `ModelCheckpoint` callback.
        pass

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the tuner's objective.

        The models are loaded with the weights corresponding to
        their best checkpoint (at the end of the best epoch of best trial).

        This method is for querying the models trained during the search.
        For best performance, it is recommended to retrain your Model on the
        full dataset using the best hyperparameters found during `search`,
        which can be obtained using `tuner.get_best_hyperparameters()`.

        Args:
            num_models: Optional number of best models to return.
                Defaults to 1.

        Returns:
            List of trained model instances sorted from the best to the worst.
        """
        # Method only exists in this class for the docstring override.
        return super(Tuner, self).get_best_models(num_models)

    def _deepcopy_callbacks(self, callbacks):
        try:
            callbacks = copy.deepcopy(callbacks)
        except:
            raise ValueError(
                "All callbacks used during a search "
                "should be deep-copyable (since they are "
                "reused across trials). "
                "It is not possible to do `copy.deepcopy(%s)`" % (callbacks,)
            )
        return callbacks

    def _configure_tensorboard_dir(self, callbacks, trial, execution=0):
        for callback in callbacks:
            if callback.__class__.__name__ == "TensorBoard":
                # Patch TensorBoard log_dir and add HParams KerasCallback
                logdir = self._get_tensorboard_dir(
                    callback.log_dir, trial.trial_id, execution
                )
                callback.log_dir = logdir
                hparams = tuner_utils.convert_hyperparams_to_hparams(
                    trial.hyperparameters
                )
                callbacks.append(
                    hparams_api.KerasCallback(
                        writer=logdir, hparams=hparams, trial_id=trial.trial_id
                    )
                )

    def _get_tensorboard_dir(self, logdir, trial_id, execution):
        return os.path.join(str(logdir), str(trial_id), "execution" + str(execution))

    def _get_checkpoint_dir(self, trial_id, epoch):
        return os.path.join(
            self.get_trial_dir(trial_id), "checkpoints", "epoch_" + str(epoch)
        )

    def _get_checkpoint_fname(self, trial_id, epoch):
        return os.path.join(
            # Each checkpoint is saved in its own directory.
            self._get_checkpoint_dir(trial_id, epoch),
            "checkpoint",
        )

    def _checkpoint_model(self, model, trial_id, epoch):
        fname = self._get_checkpoint_fname(trial_id, epoch)
        # Save in TF format.
        model.save_weights(fname)
        return fname

    def _delete_checkpoint(self, trial_id, epoch):
        tf.io.gfile.rmtree(self._get_checkpoint_dir(trial_id, epoch))
