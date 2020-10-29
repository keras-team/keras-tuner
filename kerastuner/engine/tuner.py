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

import copy
import os

from tensorboard.plugins.hparams import api as hparams_api
import tensorflow as tf

from . import base_tuner
from . import hypermodel as hm_module
from . import tuner_utils


class Tuner(base_tuner.BaseTuner):
    """Tuner class for Keras models.

    May be subclassed to create new tuners.

    # Arguments:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        max_model_size: Int. Maximum number of scalars
            in the parameters of a model. Models larger
            than this are rejected.
        optimizer: Optional. Optimizer instance.
            May be used to override the `optimizer`
            argument in the `compile` step for the
            models. If the hypermodel
            does not compile the models it generates,
            then this argument must be specified.
        loss: Optional. May be used to override the `loss`
            argument in the `compile` step for the
            models. If the hypermodel
            does not compile the models it generates,
            then this argument must be specified.
        metrics: Optional. May be used to override the
            `metrics` argument in the `compile` step
            for the models. If the hypermodel
            does not compile the models it generates,
            then this argument must be specified.
        distribution_strategy: Optional. A TensorFlow
            `tf.distribute` DistributionStrategy instance. If
            specified, each trial will run under this scope. For
            example, `tf.distribute.MirroredStrategy(['/gpu:0, /'gpu:1])`
            will run each trial on two GPUs. Currently only
            single-worker strategies are supported.
        directory: String. Path to the working directory (relative).
        project_name: Name to use as prefix for files saved
            by this Tuner.
        logger: Optional. Instance of Logger class, used for streaming data
            to Cloud Service for monitoring.
        tuner_id: Optional. If set, use this value as the id of this Tuner.
        overwrite: Bool, default `False`. If `False`, reloads an existing project
            of the same name if one is found. Otherwise, overwrites the project.
    """

    def __init__(self,
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
                 overwrite=False):

        # Subclasses of `KerasHyperModel` are not automatically wrapped.
        if not isinstance(hypermodel, hm_module.KerasHyperModel):
            hypermodel = hm_module.KerasHyperModel(
                hypermodel,
                max_model_size=max_model_size,
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                distribution_strategy=distribution_strategy)

        super(Tuner, self).__init__(oracle=oracle,
                                    hypermodel=hypermodel,
                                    directory=directory,
                                    project_name=project_name,
                                    logger=logger,
                                    overwrite=overwrite)

        self.distribution_strategy = distribution_strategy

        # Support multi-worker distribution strategies w/ distributed tuning.
        # Only the chief worker in each cluster should report results.
        if self.distribution_strategy is not None:
            self.oracle.multi_worker = (
                self.distribution_strategy.extended._in_multi_worker_mode())
            self.oracle.should_report = (
                self.distribution_strategy.extended.should_checkpoint)

        # Save only the last N checkpoints.
        self._save_n_checkpoints = 10

        self.tuner_id = tuner_id or self.tuner_id

    def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
        """For AutoKeras to override.

        DO NOT REMOVE this function. AutoKeras overrides the function to tune
        tf.data preprocessing pipelines, preprocess the dataset to obtain
        the input shape before building the model, adapt preprocessing layers,
        and tune other fit_args and fit_kwargs.

        # Arguments:
            trial: A `Trial` instance that contains the information
              needed to run this trial. `Hyperparameters` can be accessed
              via `trial.hyperparameters`.
            fit_args: Positional arguments passed by `search`.
            fit_kwargs: Keyword arguments passed by `search`.

        # Returns:
            The fit history.
        """
        model = self.hypermodel.build(trial.hyperparameters)
        return model.fit(*fit_args, **fit_kwargs)

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """Evaluates a set of hyperparameter values.

        This method is called during `search` to evaluate a set of
        hyperparameters.

        # Arguments:
            trial: A `Trial` instance that contains the information
              needed to run this trial. `Hyperparameters` can be accessed
              via `trial.hyperparameters`.
            *fit_args: Positional arguments passed by `search`.
            *fit_kwargs: Keyword arguments passed by `search`.
        """
        # Handle any callbacks passed to `fit`.
        copied_fit_kwargs = copy.copy(fit_kwargs)
        callbacks = fit_kwargs.pop('callbacks', [])
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
        copied_fit_kwargs['callbacks'] = callbacks

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
            best_epoch = self.oracle.get_trial(
                trial_id).metrics.get_best_step(self.oracle.objective.name)
        if epoch > self._save_n_checkpoints and epoch_to_delete != best_epoch:
            self._delete_checkpoint(
                trial_id, epoch_to_delete)

    def load_model(self, trial):
        model = self.hypermodel.build(trial.hyperparameters)
        # Reload best checkpoint. The Oracle scores the Trial and also
        # indicates at what epoch the best value of the objective was
        # obtained.
        best_epoch = trial.best_step
        with hm_module.maybe_distribute(self.distribution_strategy):
            model.load_weights(self._get_checkpoint_fname(
                trial.trial_id, best_epoch))
        return model

    def on_epoch_begin(self, trial, model, epoch, logs=None):
        """A hook called at the start of every epoch.

        # Arguments:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            epoch: The current epoch number.
            logs: Additional metrics.
        """
        pass

    def on_batch_begin(self, trial, model, batch, logs):
        """A hook called at the start of every batch.

        # Arguments:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            batch: The current batch number within the
              curent epoch.
            logs: Additional metrics.
        """
        pass

    def on_batch_end(self, trial, model, batch, logs=None):
        """A hook called at the end of every batch.

        # Arguments:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            batch: The current batch number within the
              curent epoch.
            logs: Additional metrics.
        """
        pass

    def on_epoch_end(self, trial, model, epoch, logs=None):
        """A hook called at the end of every epoch.

        # Arguments:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            epoch: The current epoch number.
            logs: Dict. Metrics for this epoch. This should include
              the value of the objective for this epoch.
        """
        self.save_model(trial.trial_id, model, step=epoch)
        # Report intermediate metrics to the `Oracle`.
        status = self.oracle.update_trial(
            trial.trial_id, metrics=logs, step=epoch)
        trial.status = status
        if trial.status == "STOPPED":
            model.stop_training = True

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the tuner's objective.

        The models are loaded with the weights corresponding to
        their best checkpoint (at the end of the best epoch of best trial).

        This method is only a convenience shortcut. For best performance, It is
        recommended to retrain your Model on the full dataset using the best
        hyperparameters found during `search`.

        Args:
            num_models (int, optional): Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.

        Returns:
            List of trained model instances.
        """
        # Method only exists in this class for the docstring override.
        return super(Tuner, self).get_best_models(num_models)

    def _deepcopy_callbacks(self, callbacks):
        try:
            callbacks = copy.deepcopy(callbacks)
        except:
            raise ValueError(
                'All callbacks used during a search '
                'should be deep-copyable (since they are '
                'reused across trials). '
                'It is not possible to do `copy.deepcopy(%s)`' %
                (callbacks,))
        return callbacks

    def _configure_tensorboard_dir(self, callbacks, trial):
        for callback in callbacks:
            if callback.__class__.__name__ == 'TensorBoard':
                # Patch TensorBoard log_dir and add HParams KerasCallback
                logdir = self._get_tensorboard_dir(callback.log_dir, trial.trial_id)
                callback.log_dir = logdir
                hparams = tuner_utils.convert_hyperparams_to_hparams(
                    trial.hyperparameters)
                callbacks.append(
                    hparams_api.KerasCallback(
                        writer=logdir,
                        hparams=hparams,
                        trial_id=trial.trial_id))

    def _get_tensorboard_dir(self, logdir, trial_id):
        return os.path.join(str(logdir), str(trial_id))

    def _get_checkpoint_dir(self, trial_id, epoch):
        checkpoint_dir = os.path.join(
            self.get_trial_dir(trial_id),
            'checkpoints',
            'epoch_' + str(epoch))
        tf.io.gfile.makedirs(checkpoint_dir)
        return checkpoint_dir

    def _get_checkpoint_fname(self, trial_id, epoch):
        checkpoint_fname = os.path.join(
            # Each checkpoint is saved in its own directory.
            self._get_checkpoint_dir(trial_id, epoch),
            'checkpoint')
        if (isinstance(self.distribution_strategy, tf.distribute.TPUStrategy) and
                not self.project_dir.startswith('gs://')):
            # TPU strategy only support saving h5 format on local path
            return checkpoint_fname + '.h5'
        return checkpoint_fname

    def _checkpoint_model(self, model, trial_id, epoch):
        fname = self._get_checkpoint_fname(trial_id, epoch)
        # Save in TF format.
        model.save_weights(fname)
        return fname

    def _delete_checkpoint(self, trial_id, epoch):
        tf.io.gfile.rmtree(self._get_checkpoint_dir(trial_id, epoch))
