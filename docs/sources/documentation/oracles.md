# Oracles

Each `Oracle` class implements a particular hyperparameter tuning algorithm. An `Oracle` is passed as an argument to a `Tuner`. The `Oracle` tells the `Tuner` which hyperparameters should be tried next. 

Most `Oracle` classes can be combined with any user-defined `Tuner` subclass. `Hyperband` requires the `Tuner` class to implement additional `Oracle`-specific functionality (see `Hyperband` documentation). If you do not need to subclass `Tuner` (the most common case), we also provide a number of convenience classes that package a `Tuner` and an `Oracle` together (e.g `kerastuner.RandomSearch`, `kerastuner.BayesianOptimization`, and `kerastuner.Hyperband`).


<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/tuners/bayesian.py#L13)</span>
### BayesianOptimizationOracle

```python
kerastuner.tuners.bayesian.BayesianOptimizationOracle(objective, max_trials, num_initial_points=2, alpha=0.0001, beta=2.6, seed=None, hyperparameters=None, allow_new_entries=True, tune_new_entries=True)
```

Bayesian optimization oracle.

It uses Bayesian optimization with a underlying Gaussian process model.
The acquisition function used is upper confidence bound (UCB), which can
be found in the following link:
https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf

__Arguments__

- __objective__: String or `kerastuner.Objective`. If a string,
  the direction of the optimization (min or max) will be
  inferred.
- __max_trials__: Int. Total number of trials
    (model configurations) to test at most.
    Note that the oracle may interrupt the search
    before `max_trial` models have been tested if the search space has been
    exhausted.
- __num_initial_points__: Int. The number of randomly generated samples as initial
    training data for Bayesian optimization.
- __alpha__: Float. Value added to the diagonal of the kernel matrix
    during fitting. It represents the expected amount of noise
    in the observed performances in Bayesian optimization.
- __beta__: Float. The balancing factor of exploration and exploitation.
    The larger it is, the more explorative it is.
- __seed__: Int. Random seed.
- __hyperparameters__: HyperParameters class instance.
    Can be used to override (or register in advance)
    hyperparamters in the search space.
- __tune_new_entries__: Whether hyperparameter entries
    that are requested by the hypermodel
    but that were not specified in `hyperparameters`
    should be added to the search space, or not.
    If not, then the default value for these parameters
    will be used.
- __allow_new_entries__: Whether the hypermodel is allowed
    to request hyperparameter entries not listed in
    `hyperparameters`.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/tuners/hyperband.py#L21)</span>
### HyperbandOracle

```python
kerastuner.tuners.hyperband.HyperbandOracle(objective, max_epochs, factor=3, hyperband_iterations=1, seed=None, hyperparameters=None, allow_new_entries=True, tune_new_entries=True)
```

Oracle class for Hyperband.

Note that to use this Oracle with your own subclassed Tuner, your Tuner
class must be able to handle in `Tuner.run_trial` three special hyperparameters
that will be set by this Tuner:

- "tuner/trial_id": String, optionally set. The trial_id of the Trial to load
from when starting this trial.
- "tuner/initial_epoch": Int, always set. The initial epoch the Trial should be
started from.
- "tuner/epochs": Int, always set. The cumulative number of epochs this Trial
should be trained.

These hyperparameters will be set during the "successive halving" portion
of the Hyperband algorithm.

Example `run_trial`:

```
def run_trial(self, trial, *args, **kwargs):
    hp = trial.hyperparameters
    if "tuner/trial_id" in hp:
        past_trial = self.oracle.get_trial(hp['tuner/trial_id'])
        model = self.load_model(past_trial)
    else:
        model = self.hypermodel.build(hp)

    initial_epoch = hp['tuner/initial_epoch']
    last_epoch = hp['tuner/epochs']

    for epoch in range(initial_epoch, last_epoch):
        self.on_epoch_begin(...)
        for step in range(...):
            # Run model training step here.
        self.on_epoch_end(...)
```

__Arguments:__

- __objective__: String or `kerastuner.Objective`. If a string,
  the direction of the optimization (min or max) will be
  inferred.
- __max_epochs__: Int. The maximum number of epochs to train one model. It is
  recommended to set this to a value slightly higher than the expected epochs
  to convergence for your largest Model, and to use early stopping during
  training (for example, via `tf.keras.callbacks.EarlyStopping`).
- __factor__: Int. Reduction factor for the number of epochs
    and number of models for each bracket.
- __hyperband_iterations__: Int >= 1. The number of times to iterate over the full
  Hyperband algorithm. One iteration will run approximately
  `max_epochs * (math.log(max_epochs, factor) ** 2)` cumulative epochs
  across all trials. It is recommended to set this to as high a value
  as is within your resource budget.
- __seed__: Int. Random seed.
- __hyperparameters__: HyperParameters class instance.
    Can be used to override (or register in advance)
    hyperparamters in the search space.
- __tune_new_entries__: Whether hyperparameter entries
    that are requested by the hypermodel
    but that were not specified in `hyperparameters`
    should be added to the search space, or not.
    If not, then the default value for these parameters
    will be used.
- __allow_new_entries__: Whether the hypermodel is allowed
    to request hyperparameter entries not listed in
    `hyperparameters`.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/tuners/randomsearch.py#L28)</span>
### RandomSearchOracle

```python
kerastuner.tuners.randomsearch.RandomSearchOracle(objective, max_trials, seed=None, hyperparameters=None, allow_new_entries=True, tune_new_entries=True)
```

Random search oracle.

__Arguments:__

- __objective__: String or `kerastuner.Objective`. If a string,
  the direction of the optimization (min or max) will be
  inferred.
- __max_trials__: Int. Total number of trials
    (model configurations) to test at most.
    Note that the oracle may interrupt the search
    before `max_trial` models have been tested.
- __seed__: Int. Random seed.
- __hyperparameters__: HyperParameters class instance.
    Can be used to override (or register in advance)
    hyperparamters in the search space.
- __tune_new_entries__: Whether hyperparameter entries
    that are requested by the hypermodel
    but that were not specified in `hyperparameters`
    should be added to the search space, or not.
    If not, then the default value for these parameters
    will be used.
- __allow_new_entries__: Whether the hypermodel is allowed
    to request hyperparameter entries not listed in
    `hyperparameters`.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/engine/oracle.py#L36)</span>
## Oracle class

```python
kerastuner.engine.oracle.Oracle(objective, max_trials=None, hyperparameters=None, allow_new_entries=True, tune_new_entries=True)
```

Implements a hyperparameter optimization algorithm.

Attributes:
objective: String. Name of model metric to minimize
or maximize, e.g. "val_accuracy".
max_trials: The maximum number of hyperparameter
combinations to try.
hyperparameters: HyperParameters class instance.
Can be used to override (or register in advance)
hyperparamters in the search space.
tune_new_entries: Whether hyperparameter entries
that are requested by the hypermodel
but that were not specified in `hyperparameters`
should be added to the search space, or not.
If not, then the default value for these parameters
will be used.
allow_new_entries: Whether the hypermodel is allowed
to request hyperparameter entries not listed in
`hyperparameters`.


---
## Oracle methods

### _populate_space


```python
_populate_space(trial_id)
```


Fill the hyperparameter space with values for a trial.

This method should be overrridden in subclasses and called in
`create_trial` in order to populate the hyperparameter space with
values.

Args:
`trial_id`: The id for this Trial.

Returns:
A dictionary with keys "values" and "status", where "values" is
a mapping of parameter names to suggested values, and "status"
is the TrialStatus that should be returned for this trial (one
of "RUNNING", "IDLE", or "STOPPED").

---
### _score_trial


```python
_score_trial(trial)
```


Score a completed `Trial`.

This method can be overridden in subclasses to provide a score for
a set of hyperparameter values. This method is called from `end_trial`
on completed `Trial`s.

Args:
trial: A completed `Trial` object.

---
### create_trial


```python
create_trial(tuner_id)
```


Create a new `Trial` to be run by the `Tuner`.

A `Trial` corresponds to a unique set of hyperparameters to be run
by `Tuner.run_trial`.

Args:
tuner_id: A ID that identifies the `Tuner` requesting a
`Trial`. `Tuners` that should run the same trial (for instance,
when running a multi-worker model) should have the same ID.

Returns:
A `Trial` object containing a set of hyperparameter values to run
in a `Tuner`.

---
### end_trial


```python
end_trial(trial_id, status='COMPLETED')
```


Record the measured objective for a set of parameter values.

Args:
trial_id: String. Unique id for this trial.
status: String, one of "COMPLETED", "INVALID". A status of
"INVALID" means a trial has crashed or been deemed
infeasible.

---
### get_best_trials


```python
get_best_trials(num_trials=1)
```


Returns the best `Trial`s.
---
### get_state


```python
get_state()
```

---
### set_state


```python
set_state(state)
```

---
### update_trial


```python
update_trial(trial_id, metrics, step=0)
```


Used by a worker to report the status of a trial.

Args:
trial_id: A previously seen trial id.
metrics: Dict of float. The current value of this
trial's metrics.
step: (Optional) Float. Used to report intermediate results. The
current value in a timeseries representing the state of the
trial. This is the value that `metrics` will be associated with.

Returns:
Trial object. Trial.status will be set to "STOPPED" if the Trial
should be stopped early.

