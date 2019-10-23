# Tuners

Tuners are here to do the hyperparameter search. You can create custom Tuners by subclassing `kerastuner.engine.tuner.Tuner`.


<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/tuners/bayesian.py#L273)</span>
### BayesianOptimization

```python
kerastuner.tuners.bayesian.BayesianOptimization(hypermodel, objective, max_trials, num_initial_points=2, seed=None, hyperparameters=None, tune_new_entries=True, allow_new_entries=True)
```

BayesianOptimization tuning with Gaussian process.

__Arguments:__

- __hypermodel__: Instance of HyperModel class
    (or callable that takes hyperparameters
    and returns a Model instance).
- __objective__: String. Name of model metric to minimize
    or maximize, e.g. "val_accuracy".
- __max_trials__: Int. Total number of trials
    (model configurations) to test at most.
    Note that the oracle may interrupt the search
    before `max_trial` models have been tested if the search space has
    been exhausted.
- __num_initial_points__: Int. The number of randomly generated samples as initial
    training data for Bayesian optimization.
- __alpha__: Float or array-like. Value added to the diagonal of
    the kernel matrix during fitting.
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
- __**kwargs__: Keyword arguments relevant to all `Tuner` subclasses.
    Please see the docstring for `Tuner`.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/tuners/hyperband.py#L312)</span>
### Hyperband

```python
kerastuner.tuners.hyperband.Hyperband(hypermodel, objective, max_epochs, factor=3, hyperband_iterations=1, seed=None, hyperparameters=None, tune_new_entries=True, allow_new_entries=True)
```

Variation of HyperBand algorithm.

Reference:
Li, Lisha, and Kevin Jamieson.
["Hyperband: A Novel Bandit-Based
Approach to Hyperparameter Optimization."
Journal of Machine Learning Research 18 (2018): 1-52](
http://jmlr.org/papers/v18/16-558.html).


__Arguments__

- __hypermodel__: Instance of HyperModel class
    (or callable that takes hyperparameters
    and returns a Model instance).
- __objective__: String. Name of model metric to minimize
    or maximize, e.g. "val_accuracy".
- __max_epochs__: Int. The maximum number of epochs to train one model. It is
  recommended to set this to a value slightly higher than the expected time
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
- __**kwargs__: Keyword arguments relevant to all `Tuner` subclasses.
    Please see the docstring for `Tuner`.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/tuners/randomsearch.py#L125)</span>
### RandomSearch

```python
kerastuner.tuners.randomsearch.RandomSearch(hypermodel, objective, max_trials, seed=None, hyperparameters=None, tune_new_entries=True, allow_new_entries=True)
```

Random search tuner.

__Arguments:__

- __hypermodel__: Instance of HyperModel class
    (or callable that takes hyperparameters
    and returns a Model instance).
- __objective__: String. Name of model metric to minimize
    or maximize, e.g. "val_accuracy".
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
- __**kwargs__: Keyword arguments relevant to all `Tuner` subclasses.
    Please see the docstring for `Tuner`.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/tuners/sklearn.py#L25)</span>
### Sklearn

```python
kerastuner.tuners.sklearn.Sklearn(oracle, hypermodel, scoring=None, metrics=None, cv=KFold(n_splits=5, random_state=1, shuffle=True))
```

Tuner for Scikit-learn Models.

Performs cross-validated hyperparameter search for Scikit-learn
models.

__Arguments:__

- __oracle__: An instance of the `kerastuner.Oracle` class. Note that for
  this `Tuner`, the `objective` for the `Oracle` should always be set
  to `Objective('score', direction='max')`. Also, `Oracle`s that exploit
  Neural-Network-specific training (e.g. `Hyperband`) should not be
  used with this `Tuner`.
- __hypermodel__: Instance of `HyperModel` class (or callable that takes a
  `Hyperparameters` object and returns a Model instance).
- __scoring__: An sklearn `scoring` function. For more information, see
  `sklearn.metrics.make_scorer`. If not provided, the Model's default
  scoring will be used via `model.score`. Note that if you are searching
  across different Model families, the default scoring for these Models
  will often be different. In this case you should supply `scoring` here
  in order to make sure your Models are being scored on the same metric.
- __metrics__: Additional `sklearn.metrics` functions to monitor during search.
  Note that these metrics do not affect the search process.
- __cv__: An `sklearn.model_selection` Splitter class. Used to
  determine how samples are split up into groups for cross-validation.
- __**kwargs__: Keyword arguments relevant to all `Tuner` subclasses. Please
  see the docstring for `Tuner`.

Example:

```python
import kerastuner as kt
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

def build_model(hp):
  model_type = hp.Choice('model_type', ['random_forest', 'ridge'])
  if model_type == 'random_forest':
    model = ensemble.RandomForestClassifier(
        n_estimators=hp.Int('n_estimators', 10, 50, step=10),
        max_depth=hp.Int('max_depth', 3, 10))
  else:
    model = linear_model.RidgeClassifier(
        alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))
  return model

tuner = kt.tuners.Sklearn(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('score', 'max'),
        max_trials=10),
    hypermodel=build_model,
    scoring=metrics.make_scorer(metrics.accuracy_score),
    cv=model_selection.StratifiedKFold(5),
    directory='.',
    project_name='my_project')

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2)

tuner.search(X_train, y_train)

best_model = tuner.get_best_models(num_models=1)[0]
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/engine/tuner.py#L35)</span>
## Tuner class

```python
kerastuner.engine.tuner.Tuner(oracle, hypermodel, max_model_size=None, optimizer=None, loss=None, metrics=None, distribution_strategy=None, directory=None, project_name=None, logger=None, tuner_id=None, overwrite=False)
```

Tuner class for Keras models.

May be subclassed to create new tuners.

__Arguments:__

- __oracle__: Instance of Oracle class.
- __hypermodel__: Instance of HyperModel class
    (or callable that takes hyperparameters
    and returns a Model instance).
- __max_model_size__: Int. Maximum size of weights
    (in floating point coefficients) for a valid
    models. Models larger than this are rejected.
- __optimizer__: Optional. Optimizer instance.
    May be used to override the `optimizer`
    argument in the `compile` step for the
    models. If the hypermodel
    does not compile the models it generates,
    then this argument must be specified.
- __loss__: Optional. May be used to override the `loss`
    argument in the `compile` step for the
    models. If the hypermodel
    does not compile the models it generates,
    then this argument must be specified.
- __metrics__: Optional. May be used to override the
    `metrics` argument in the `compile` step
    for the models. If the hypermodel
    does not compile the models it generates,
    then this argument must be specified.
- __distribution_strategy__: Optional. A TensorFlow
    `tf.distribute` DistributionStrategy instance. If
    specified, each trial will run under this scope. For
    example, `tf.distribute.MirroredStrategy(['/gpu:0, /'gpu:1])`
    will run each trial on two GPUs. Currently only
    single-worker strategies are supported.
- __directory__: String. Path to the working directory (relative).
- __project_name__: Name to use as prefix for files saved
    by this Tuner.
- __logger__: Optional. Instance of Logger class, used for streaming data
    to Cloud Service for monitoring.
- __overwrite__: Bool, default `False`. If `False`, reloads an existing project
    of the same name if one is found. Otherwise, overwrites the project.
    

---
## Tuner methods

### get_best_models


```python
get_best_models(num_models=1)
```


Returns the best model(s), as determined by the tuner's objective.

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

---
### get_state


```python
get_state()
```

---
### load_model


```python
load_model(trial)
```

---
### on_epoch_begin


```python
on_epoch_begin(trial, model, epoch, logs=None)
```


A hook called at the start of every epoch.

__Arguments:__

- __trial__: A `Trial` instance.
- __model__: A Keras `Model`.
- __epoch__: The current epoch number.
- __logs__: Additional metrics.
    
---
### on_batch_begin


```python
on_batch_begin(trial, model, batch, logs)
```


A hook called at the start of every batch.

__Arguments:__

- __trial__: A `Trial` instance.
- __model__: A Keras `Model`.
- __batch__: The current batch number within the
  curent epoch.
- __logs__: Additional metrics.
    
---
### on_batch_end


```python
on_batch_end(trial, model, batch, logs=None)
```


A hook called at the end of every batch.

__Arguments:__

- __trial__: A `Trial` instance.
- __model__: A Keras `Model`.
- __batch__: The current batch number within the
  curent epoch.
- __logs__: Additional metrics.
    
---
### on_epoch_end


```python
on_epoch_end(trial, model, epoch, logs=None)
```


A hook called at the end of every epoch.

__Arguments:__

- __trial__: A `Trial` instance.
- __model__: A Keras `Model`.
- __epoch__: The current epoch number.
- __logs__: Dict. Metrics for this epoch. This should include
  the value of the objective for this epoch.
    
---
### run_trial


```python
run_trial(trial)
```


Evaluates a set of hyperparameter values.

This method is called during `search` to evaluate a set of
hyperparameters.

__Arguments:__

- __trial__: A `Trial` instance that contains the information
  needed to run this trial. `Hyperparameters` can be accessed
  via `trial.hyperparameters`.
- __*fit_args__: Positional arguments passed by `search`.
- __*fit_kwargs__: Keyword arguments passed by `search`.
    
---
### save_model


```python
save_model(trial_id, model, step=0)
```

---
### search


```python
search()
```


Performs a search for best hyperparameter configuations.

__Arguments:__

- __*fit_args__: Positional arguments that should be passed to
  `run_trial`, for example the training and validation data.
- __*fit_kwargs__: Keyword arguments that should be passed to
  `run_trial`, for example the training and validation data.
    
---
### set_state


```python
set_state(state)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/engine/base_tuner.py#L35)</span>
## BaseTuner class

```python
kerastuner.engine.base_tuner.BaseTuner(oracle, hypermodel, directory=None, project_name=None, logger=None, overwrite=False)
```

Tuner base class.

May be subclassed to create new tuners, including for non-Keras models.

__Arguments:__

- __oracle__: Instance of Oracle class.
- __hypermodel__: Instance of HyperModel class
    (or callable that takes hyperparameters
    and returns a Model instance).
- __directory__: String. Path to the working directory (relative).
- __project_name__: Name to use as prefix for files saved
    by this Tuner.
- __logger__: Optional. Instance of Logger class, used for streaming data
    to Cloud Service for monitoring.
- __overwrite__: Bool, default `False`. If `False`, reloads an existing project
    of the same name if one is found. Otherwise, overwrites the project.
    

---
## BaseTuner methods

### get_best_models


```python
get_best_models(num_models=1)
```


Returns the best model(s), as determined by the tuner's objective.

This method is only a convenience shortcut. For best performance, It is
recommended to retrain your Model on the full dataset using the best
hyperparameters found during `search`.

Args:
num_models (int, optional): Number of best models to return.
Models will be returned in sorted order. Defaults to 1.

Returns:
List of trained model instances.

---
### get_state


```python
get_state()
```

---
### load_model


```python
load_model(trial)
```


Loads a Model from a given trial.

__Arguments:__

- __trial__: A `Trial` instance. For models that report intermediate
  results to the `Oracle`, generally `load_model` should load the
  best reported `step` by relying of `trial.best_step`
    
---
### run_trial


```python
run_trial(trial)
```


Evaluates a set of hyperparameter values.

This method is called during `search` to evaluate a set of
hyperparameters.

For subclass implementers: This method is responsible for
reporting metrics related to the `Trial` to the `Oracle`
via `self.oracle.update_trial`.

Simplest example:

```python
def run_trial(self, trial, x, y, val_x, val_y):
    model = self.hypermodel.build(trial.hyperparameters)
    model.fit(x, y)
    loss = model.evaluate(val_x, val_y)
    self.oracle.update_trial(
      trial.trial_id, {'loss': loss})
    self.save_model(trial.trial_id, model)
```

__Arguments:__

- __trial__: A `Trial` instance that contains the information
  needed to run this trial. Hyperparameters can be accessed
  via `trial.hyperparameters`.
- __*fit_args__: Positional arguments passed by `search`.
- __*fit_kwargs__: Keyword arguments passed by `search`.
    
---
### save_model


```python
save_model(trial_id, model, step=0)
```


Saves a Model for a given trial.

__Arguments:__

- __trial_id__: The ID of the `Trial` that corresponds to this Model.
- __model__: The trained model.
- __step__: For models that report intermediate results to the `Oracle`,
  the step that this saved file should correspond to. For example,
  for Keras models this is the number of epochs trained.
    
---
### search


```python
search()
```


Performs a search for best hyperparameter configuations.

__Arguments:__

- __*fit_args__: Positional arguments that should be passed to
  `run_trial`, for example the training and validation data.
- __*fit_kwargs__: Keyword arguments that should be passed to
  `run_trial`, for example the training and validation data.
    
---
### set_state


```python
set_state(state)
```

