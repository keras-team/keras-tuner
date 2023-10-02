# Release v1.4.4

## Bug fixes
* Could not do `from keras_tuner.engine.hyperparameters import serialize`. It is now fixed.
* Could not do `from keras_tuner.engine.hyperparameters import deserialize`. It is now fixed.
* Could not do `from keras_tuner.engine.tuner import maybe_distribute`. It is now fixed.

# Release v1.4.3

## Bug fixes
* Could not do `from keras_tuner.engine.tuner import Tuner`. It is now fixed.
* When TensorFlow version is low, it would error out with keras models have no
  attributed called `get_build_config`. It is now fixed.

# Release v1.4.2

## Bug fixes
* Could not do `from keras_tuner.engine import trial`. It is now fixed.

# Release v1.4.1

## Bug fixes
* Could not do `from keras_tuner.engine import base_tuner`. It is now fixed.

# Release v1.4.0

## Breaking changes
* All private APIs are hidden under `keras_tuner.src.*`. For example, if you use
  `keras_tuner.some_private_api`, it will now be
  `keras_tuner.src.some_private_api`.

## New features
* Support Keras Core with multi-backend.

# Release v1.3.5

## Breaking changes
* Removed TensorFlow from the required dependencies of KerasTuner. The user need
  to install TensorFlow either separately with KerasTuner or with
  `pip install keras_tuner[tensorflow]`. This change is because some people may
  want to use KerasTuner with `tensorflow-cpu` instead of `tensorflow`.

## Bug fixes
* KerasTuner used to require protobuf version to be under 3.20. The limit is
  removed. Now, it support both protobuf 3 and 4.

# Release v1.3.4

## Bug fixes
* If you have a protobuf version > 3.20, it would through an error when import
  KerasTuner. It is now fixed.

# Release v1.3.3

## Bug fixes
* KerasTuner would install protobuf 3.19 with `protobuf<=3.20`. We want to
  install `3.20.3`, so we changed it to `protobuf<=3.20.3`. It is now fixed.

# Release v1.3.2

## Bug fixes
* It use to install protobuf 4.22.1 if install with TensorFlow 2.12, which is
  not compatible with KerasTuner. We limited the version to <=3.20. Now it is
  fixed.

# Release v1.3.1

## Bug fixes
* The `Tuner.results_summary()` did not print error messages for failed trials
  and did not display `Objective` information correctly. It is now fixed.
* The `BayesianOptimization` would break when not specifying the
  `num_initial_points` and overriding `.run_trial()`. It is now fixed.
* TensorFlow 2.12 would break because the different protobuf version. It is now
  fixed.

# Release v1.3.0

## Breaking changes
* Removed `Logger` and `CloudLogger` and the related arguments in
  `BaseTuner.__init__(logger=...)`.
* Removed `keras_tuner.oracles.BayesianOptimization`,
  `keras_tuner.oracles.Hyperband`, `keras_tuner.oracles.RandomSearch`, which
  were actually `Oracle`s instead of `Tuner`s. Please
  use`keras_tuner.oracles.BayesianOptimizationOracle`,
  `keras_tuner.oracles.HyperbandOracle`,
  `keras_tuner.oracles.RandomSearchOracle` instead.
* Removed `keras_tuner.Sklearn`. Please use `keras_tuner.SklearnTuner` instead.

## New features
* `keras_tuner.oracles.GridSearchOracle` is now available as a standalone
  `Oracle` to be used with custom tuners.

# Release v1.2.1

## Bug fixes
* The resume feature (`overwrite=False`) would crash in 1.2.0. This is now fixed.

# Release v1.2.0

## Breaking changes
* If you implemented your own `Tuner`, the old use case of reporting results
  with `Oracle.update_trial()` in `Tuner.run_trial()` is deprecated. Please
  return the metrics in `Tuner.run_trial()` instead.
* If you implemented your own `Oracle` and overrided `Oracle.end_trial()`, you
  need to change the signature of the function from
  `Oracle.end_trial(trial.trial_id, trial.status)` to `Oracle.end_trial(trial)`.
* The default value of the `step` argument in `keras_tuner.HyperParameters.Int()` is
  changed to `None`, which was `1` before. No change in default behavior.
* The default value of the `sampling` argument in
  `keras_tuner.HyperParameters.Int()` is changed to `"linear"`, which was `None`
  before. No change in default behavior.
* The default value of the `sampling` argument in
  `keras_tuner.HyperParameters.Float()` is changed to `"linear"`, which was
  `None` before. No change in default behavior.
* If you explicitly rely on protobuf values, the new protobuf bug fix may affect
  you.
* Changed the mechanism of how a random sample is drawn for a hyperparameter. They
  now all start from a random value between 0 and 1, and convert the value
  to a random sample.

## New features
* A new tuner is added, `keras_tuner.GridSearch`, which can exhaust all the
  possible hyperparameter combinations.
* Better fault tolerance during the search. Added two new arguments to `Tuner`
  and `Oracle` initializers, `max_retries_per_trial` and
  `max_consecutive_failed_trials`.
* You can now mark a `Trial` as failed by
  `raise keras_tuner.FailedTrialError("error message.")` in `HyperModel.build()`,
  `HyperModel.fit()`, or your model build function.
* Provides better error messages for invalid configs for `Int` and `Float` type
  hyperparameters.
* A decorator `@keras_tuner.synchronized` is added to decorate the methods in
  `Oracle` and its subclasses to synchronize the concurrent calls to ensure
  thread safety in parallel tuning.

## Bug fixes
* Protobuf was not converting Boolean type hyperparameter correctly. This is now
  fixed.
* Hyperband was not loading the weights correctly for half-trained models. This
  is now fixed.
* `KeyError` may occur if using `hp.conditional_scope()`, or the `parent`
  argument for hyperparameters. This is now fixed.
* `num_initial_points` of the `BayesianOptimization` should defaults to `3 *
  dimension`, but it defaults to 2. This is now fixed.
* It would through an error when using a concrete Keras optimizer object to
  override the `HyperModel` compile arg. This is now fixed.
* Workers might crash due to `Oracle` reloading when running in parallel. This is
  now fixed.
