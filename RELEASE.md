# Release v1.2.0

## Breaking changes
* Default value of the `step` argument in `keras_tuner.HyperParameters.Int()` is
  changed to `None`, which was `1` before. No change in default behavior.
* Default value of the `sampling` argument in
  `keras_tuner.HyperParameters.Int()` is changed to `"linear"`, which was `None`
  before. No change in default behavior.
* Default value of the `sampling` argument in
  `keras_tuner.HyperParameters.Float()` is changed to `"linear"`, which was
  `None` before. No change in default behavior.
* If you explicity rely on protobuf values, the new protobuf bug fix may affect
  you.
* Changed the mechanism how a random sample is drawn for a hyperparameter. They
  are now all start from a random value between 0 and 1, and convert the value
  to a random sample.
## New features
* A new tuner is added, `keras_tuner.GridSearch`.
* Provides better error messages for invalid configs for `Int` and `Float` type
  hyperparameters.
## Bug fixes
* Protobuf was not converting boolean type hyperparameter correctly. Now it is
  fixed.
