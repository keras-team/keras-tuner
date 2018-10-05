# Changelog

Major KerasTuner changes by version

## v0.6

- Upon restarting the tuner -- already existing instances are loaded to prevent retraining them
- Added avg_accuracy metric computation in callback to allows checkpointing on accuracy for multi-output
- Replaced model saving at the end of the training with model checkpointing at epoch end: model are now saved when they improve.
- Added reporting of the search space size for each hyperparameter
- Added the ability to group hyperparameters in groups to make it easier for post processing
- Added a summary() function that provide a breakdown of the hyperparameter search space.
- Added a tool to display results summary in terminal `tools/display-results-summary.py`

### Extras
- Switched from keras to tf.keras
- Switched xxhash to farmhash

### Major bugs fixes
- Callback overflows
- OOM issues while training on GPU
- Multi-gpu is now working as intended
- Model generation stats are now correctly displayed
- Dryrun mode now works as intended

## v0.5

### Major features

- Added the ability to use generator via the `search_generator` function
- Added the ability to supply a personalized callback_generator function that is invoked at every execution and receive execution_info as input. Allows to use callbacks which are different from one execution to another which is needed to support complex models such a triplet loss.
- Added a `test(int)` function which allows to test the model function before getting started

### Extras

- Add a configurable model over_size detection to avoid OOM errors when TF attempt to train model with too many parameters
- Moved the cloud configuration to backend() function with notification configuration
- Added additional statistics reporting: 

  - overall: epoch budget, epoch budget remaining, eta, hypertuner used
  - per instance: eta, epoch remaining 

- Various bug fixes including GPU memory release which caused OOM crash

## v0.4

- Cloudstreaming to keraslyzer service implemented.

## v0.3

- Added callback that takes care to periodically records current execution. Mostly meant to make easy to monitor long training.

## v0.2

- Ultraband tuner implemented and becoming default tuner - inspired by ultraband.
- Breaking API: moved to epoch_budget to define how long the tuner should run. This makes it easier for user to budget optimization.
