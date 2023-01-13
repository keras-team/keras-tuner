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
"The Tuner class."


import contextlib
import copy
import gc
import os

import numpy as np
from tensorboard.plugins.hparams import api as hparams_api
from tensorflow import keras

from keras_tuner import errors
from keras_tuner.engine import base_tuner
from keras_tuner.engine import tuner_utils


class Tuner(base_tuner.BaseTuner):
    """Tuner class for Keras models.

    This is the base `Tuner` class for all tuners for Keras models. It manages
    the building, training, evaluation and saving of the Keras models. New
    tuners can be created by subclassing the class.

    All Keras related logics are in `Tuner.run_trial()` and its subroutines.
    When subclassing `Tuner`, if not calling `super().run_trial()`, it can tune
    anything.

    Args:
        oracle: Instance of `Oracle` class.
        hypermodel: Instance of `HyperModel` class (or callable that takes
            hyperparameters and returns a `Model` instance). It is optional
            when `Tuner.run_trial()` is overriden and does not use
            `self.hypermodel`.
        max_model_size: Integer, maximum number of scalars in the parameters of
            a model. Models larger than this are rejected.
        optimizer: Optional optimizer. It is used to override the `optimizer`
            argument in the `compile` step for the models. If the hypermodel
            does not compile the models it generates, then this argument must be
            specified.
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
        hypermodel=None,
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
        if hypermodel is None and self.__class__.run_trial is Tuner.run_trial:
            raise ValueError(
                "Received `hypermodel=None`. We only allow not specifying "
                "`hypermodel` if the user defines the search space in "
                "`Tuner.run_trial()` by subclassing a `Tuner` class without "
                "using a `HyperModel` instance."
            )

        super().__init__(
            oracle=oracle,
            hypermodel=hypermodel,
            directory=directory,
            project_name=project_name,
            logger=logger,
            overwrite=overwrite,
        )

        self.max_model_size = max_model_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.distribution_strategy = distribution_strategy
        self.executions_per_trial = executions_per_trial

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

        self.tuner_id = tuner_id or self.tuner_id

    def _build_hypermodel(self, hp):
        with maybe_distribute(self.distribution_strategy):
            model = self.hypermodel.build(hp)
            self._override_compile_args(model)
            return model

    def _try_build(self, hp):
        # clean-up TF graph from previously stored (defunct) graph
        keras.backend.clear_session()
        gc.collect()

        model = self._build_hypermodel(hp)
        # Stop if `build()` does not return a valid model.
        if not isinstance(model, keras.models.Model):
            raise errors.FatalTypeError(
                "Expected the model-building function, or HyperModel.build() to "
                "return a valid Keras Model instance. "
                f"Received: {model} of type {type(model)}."
            )
        # Check model size.
        size = maybe_compute_model_size(model)
        if self.max_model_size and size > self.max_model_size:
            raise errors.FailedTrialError(
                f"Oversized model: {size} parameters. Skip model."
            )
        return model

    def _override_compile_args(self, model):
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
                    optimizer = (
                        self.optimizer
                        if isinstance(self.optimizer, str)
                        else keras.optimizers.deserialize(
                            keras.optimizers.serialize(self.optimizer)
                        )
                    )
                    compile_kwargs["optimizer"] = optimizer
                if self.metrics:
                    compile_kwargs["metrics"] = self.metrics
                model.compile(**compile_kwargs)

    def _build_and_fit_model(self, trial, *args, **kwargs):
        """For AutoKeras to override.

        DO NOT REMOVE this function. AutoKeras overrides the function to tune
        tf.data preprocessing pipelines, preprocess the dataset to obtain
        the input shape before building the model, adapt preprocessing layers,
        and tune other fit_args and fit_kwargs.

        Args:
            trial: A `Trial` instance that contains the information needed to
                run this trial. `Hyperparameters` can be accessed via
                `trial.hyperparameters`.
            *args: Positional arguments passed by `search`.
            **kwargs: Keyword arguments passed by `search`.

        Returns:
            The fit history.
        """
        hp = trial.hyperparameters
        model = self._try_build(hp)
        results = self.hypermodel.fit(hp, model, *args, **kwargs)
        tuner_utils.validate_trial_results(
            results, self.oracle.objective, "HyperModel.fit()"
        )
        return results

    def run_trial(self, trial, *args, **kwargs):
        """Evaluates a set of hyperparameter values.

        This method is called multiple times during `search` to build and
        evaluate the models with different hyperparameters and return the
        objective value.

        Example:

        You can use it with `self.hypermodel` to build and fit the model.

        ```python
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            model = self.hypermodel.build(hp)
            return self.hypermodel.fit(hp, model, *args, **kwargs)
        ```

        You can also use it as a black-box optimizer for anything.

        ```python
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            x = hp.Float("x", -2.0, 2.0)
            y = x * x + 2 * x + 1
            return y
        ```

        Args:
            trial: A `Trial` instance that contains the information needed to
                run this trial. Hyperparameters can be accessed via
                `trial.hyperparameters`.
            *args: Positional arguments passed by `search`.
            **kwargs: Keyword arguments passed by `search`.

        Returns:
            A `History` object, which is the return value of `model.fit()`, a
            dictionary, a float, or a list of one of these types.

            If return a dictionary, it should be a dictionary of the metrics to
            track. The keys are the metric names, which contains the
            `objective` name. The values should be the metric values.

            If return a float, it should be the `objective` value.

            If evaluating the model for multiple times, you may return a list
            of results of any of the types above. The final objective value is
            the average of the results in the list.
        """
        # Not using `ModelCheckpoint` to support MultiObjective.
        # It can only track one of the metrics to save the best model.
        model_checkpoint = tuner_utils.SaveBestEpoch(
            objective=self.oracle.objective,
            filepath=self._get_checkpoint_fname(trial.trial_id),
        )
        original_callbacks = kwargs.pop("callbacks", [])

        # Run the training process multiple times.
        histories = []
        for execution in range(self.executions_per_trial):
            copied_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)
            copied_kwargs["callbacks"] = callbacks
            obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)

            histories.append(obj_value)
        return histories

    def load_model(self, trial):
        model = self._try_build(trial.hyperparameters)
        # Reload best checkpoint.
        # Only load weights to avoid loading `custom_objects`.
        with maybe_distribute(self.distribution_strategy):
            model.load_weights(self._get_checkpoint_fname(trial.trial_id))
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
        """Called at the end of an epoch.

        Args:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            epoch: The current epoch number.
            logs: Dict. Metrics for this epoch. This should include
              the value of the objective for this epoch.
        """
        # Intermediate results are not passed to the Oracle, and
        # checkpointing is handled via a `SaveBestEpoch` callback.
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
        return super().get_best_models(num_models)

    def _deepcopy_callbacks(self, callbacks):
        try:
            callbacks = copy.deepcopy(callbacks)
        except:
            raise errors.FatalValueError(
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
        return os.path.join(str(logdir), str(trial_id), f"execution{str(execution)}")

    def _get_checkpoint_fname(self, trial_id):
        return os.path.join(
            # Each checkpoint is saved in its own directory.
            self.get_trial_dir(trial_id),
            "checkpoint",
        )


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
