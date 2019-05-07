# Changelog

Major KerasTuner changes by version

## V0.8: Framework complete

### Major features

- Tuner API refactored and simplified
- User API refactored and simplified
- Tunable Model API initial version and initial application with `tunable_resnet`
- Experiments recording refactored to remove duplicate, improve bucketing and
log additional informations.
- Display components refactored to offer a more consistent an clearer experience.
- Tensorflow 2 compatibility added
- Model exporting added -- allows to export in Keras formats and Tensorflow
- Model resuming added.
- Hello world tutorial completed.
- Add classification metrics reporting.

### Extra

- access to system info and display have been moved to an abstraction layer
- Major code refactoring to finalize API, cleanup the code and encapsulate all
components correctly. This include a rewrite of the following components:
`abstraction`, `distribution`, `collections`, `callbacks`, `engine`, `tuner`
- unit-test refactored and extended with over 100 more tests.


## V0.7: Ecosystem integration

The focus of these releases is to make kerastuner works well with other and make it easy to operate

### Major features

- `kerastuner-status` utility allows to monitor training in commandline
- `kerastuner-summary` utility allows to display results overview in commandline
- Tuner report status in json file every 5 seconds to make it easy to track progress with tool and remotely
- System information including CPU usage, GPU usage, memory usage, disk space is now reported
- Model max parameters is now directly infered based from GPU available memory and batch_size

### Extras

- Moved statistics reporting to callback and revamped it to look better and use new display/table system
- Reduced the amount of boilerplate code needed to write a tuner by shifting burden to the scheduler and callback
- Output adapts when run in colab/jupyter notebooks to use HTML for better readability
- New cross-platform display subsystem that produce nice and colorful output
- Kerastuner warm  user if tensorflow is not using GPU and GPU are available

### Noteworthy bugfixes

- setup.py now works in developper mode (issue #41)
- utilities now use setuptools proprerly to offers cross-platform executable

## v0.6: Engine stablization

The focus of this set of release is to have a robust engine that works seamlessly and we can build upon

### Major features

- Upon restarting the tuner -- already existing instances are loaded to prevent retraining them
- Added avg_accuracy metric computation in callback to allows checkpointing on accuracy for multi-output
- Replaced model saving at the end of the training with model checkpointing at epoch end: model are now saved when they improve.
- Added reporting of the search space size for each hyperparameter
- Added the ability to group hyperparameters in groups to make it easier for post processing
- Added a summary() function that provide a breakdown of the hyperparameter search space.
- Added a tool to display results summary in terminal `tools/display-results-summary.py`

### Extras

- simplified demo
- Switched from keras to tf.keras
- Switched xxhash to hashlib due to dependencies issues (rare)
- Added traceback via verbose for model debug

### Noteworthy bugfixes

- Default metrics not set properly
- gs:// schema unsupported crashed the tuner
- Num epochs is properly tracked
- History is now serializable
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
