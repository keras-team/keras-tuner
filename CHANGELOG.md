# Changelog

Major KerasTuner changes by version

## v0.5

- Added the ability to use generator via the `search_generator` function
- Various bug fixes

## v0.4

- Cloudstreaming to keraslyzer service implemented.

## v0.3

- Added callback that takes care to periodically records current execution. Mostly meant to make easy to monitor long training.

## v0.2

- Ultraband tuner implemented and becoming default tuner - inspired by ultraband.
- Breaking API: moved to epoch_budget to define how long the tuner should run. This makes it easier for user to budget optimization.
