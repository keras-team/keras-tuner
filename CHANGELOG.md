# Changelog

Major KerasTuner changes by version


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
