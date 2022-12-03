# Release v1.2.0

## Breaking changes
* If you implemented your own `Tuner`, the old use case of reporting results
  with `Oracle.update_trial()` in `Tuner.run_trial()` is deprecated. Please
  return the metrics in `Tuner.run_trial()` instead.
* If you implemented your own `Oracle` and overrided `Oracle.end_trial()`, you
  need to change the signature of the function to
  `Oracle.end_trial(trial_id, status, message)`.
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
* Provides better error messages for invalid configs for `Int` and `Float` type
  hyperparameters.

## Bug fixes
* Protobuf was not converting Boolean type hyperparameter correctly. This is now
  fixed.
* Hyperband was not loading the weights correctly for half-trained models. This
  is now fixed.
