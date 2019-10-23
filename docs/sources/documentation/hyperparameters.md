# HyperParameters

The `HyperParameters` class serves as a hyerparameter container. A `HyperParameters` instance contains information about both the search space and the current values of each hyperparameter. 

Hyperparameters can be defined inline with the model-building code that uses them. This saves you from having to write boilerplate code and helps to make the code more maintainable.

### Example: Building a Model using `HyperParameters`

```python
import kerastuner as kt
import tensorflow as tf

def build_model(hp):
    model = tf.keras.Sequential()
    for i in range(hp.Int('layers', 3, 10)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int('units_' + str(i), 50, 100, step=10),
            activation=hp.Choice('act_' + str(i), ['relu', 'tanh'])))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

hp = kt.HyperParameters()
model = build_model(hp)
assert 'layers' in hp
assert 'units_0' in hp

# Restrict the search space.
hp = kt.HyperParameters()
hp.Fixed('layers', 5)
model = build_model(hp)
assert hp['layers'] == 5

# Reparametrize the search space.
hp = kt.HyperParameters()
hp.Int('layers', 20, 30)
model = build_model(hp)
assert hp['layers'] >= 20
```

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/engine/hyperparameters.py#L448)</span>
## HyperParameters class

```python
kerastuner.engine.hyperparameters.HyperParameters()
```

Container for both a hyperparameter space, and current values.

__Attributes:__

- __space__: A list of HyperParameter instances.
- __values__: A dict mapping hyperparameter names to current values.
    

---
## HyperParameters methods

### Boolean


```python
Boolean(name, default=False, parent_name=None, parent_values=None)
```


Choice between True and False.

__Arguments__

- __name__: Str. Name of parameter. Must be unique.
- __default__: Default value to return for the parameter.
    If unspecified, the default value will be False.

- __parent_name__: (Optional) String. Specifies that this hyperparameter is
  conditional. The name of the this hyperparameter's parent.
- __parent_values__: (Optional) List. The values of the parent hyperparameter
  for which this hyperparameter should be considered active.

__Returns:__

The current value of this hyperparameter.

---
### Choice


```python
Choice(name, values, ordered=None, default=None, parent_name=None, parent_values=None)
```


Choice of one value among a predefined set of possible values.

__Arguments:__

- __name__: Str. Name of parameter. Must be unique.
- __values__: List of possible values. Values must be int, float,
    str, or bool. All values must be of the same type.
- __ordered__: Whether the values passed should be considered to
    have an ordering. This defaults to `True` for float/int
    values. Must be `False` for any other values.
- __default__: Default value to return for the parameter.
    If unspecified, the default value will be:
    - None if None is one of the choices in `values`
    - The first entry in `values` otherwise.

- __parent_name__: (Optional) String. Specifies that this hyperparameter is
  conditional. The name of the this hyperparameter's parent.
- __parent_values__: (Optional) List. The values of the parent hyperparameter
  for which this hyperparameter should be considered active.

__Returns:__

The current value of this hyperparameter.

---
### Fixed


```python
Fixed(name, value, parent_name=None, parent_values=None)
```


Fixed, untunable value.

__Arguments__

- __name__: Str. Name of parameter. Must be unique.
- __value__: Value to use (can be any JSON-serializable
    Python type).

- __parent_name__: (Optional) String. Specifies that this hyperparameter is
  conditional. The name of the this hyperparameter's parent.
- __parent_values__: (Optional) List. The values of the parent hyperparameter
  for which this hyperparameter should be considered active.

__Returns:__

The current value of this hyperparameter.

---
### Float


```python
Float(name, min_value, max_value, step=None, sampling=None, default=None, parent_name=None, parent_values=None)
```


Floating point range, can be evenly divided.

__Arguments:__

- __name__: Str. Name of parameter. Must be unique.
- __min_value__: Float. Lower bound of the range.
- __max_value__: Float. Upper bound of the range.
- __step__: Optional. Float, e.g. 0.1.
    smallest meaningful distance between two values.
    Whether step should be specified is Oracle dependent,
    since some Oracles can infer an optimal step automatically.
- __sampling__: Optional. One of "linear", "log",
    "reverse_log". Acts as a hint for an initial prior
    probability distribution for how this value should
    be sampled, e.g. "log" will assign equal
    probabilities to each order of magnitude range.
- __default__: Default value to return for the parameter.
    If unspecified, the default value will be
    `min_value`.

- __parent_name__: (Optional) String. Specifies that this hyperparameter is
  conditional. The name of the this hyperparameter's parent.
- __parent_values__: (Optional) List. The values of the parent hyperparameter
  for which this hyperparameter should be considered active.

__Returns:__

The current value of this hyperparameter.

---
### Int


```python
Int(name, min_value, max_value, step=1, sampling=None, default=None, parent_name=None, parent_values=None)
```


Integer range.

Note that unlinke Python's `range` function, `max_value` is *included* in
the possible values this parameter can take on.

__Arguments:__

- __name__: Str. Name of parameter. Must be unique.
- __min_value__: Int. Lower limit of range (included).
- __max_value__: Int. Upper limit of range (included).
- __step__: Int. Step of range.
- __sampling__: Optional. One of "linear", "log",
    "reverse_log". Acts as a hint for an initial prior
    probability distribution for how this value should
    be sampled, e.g. "log" will assign equal
    probabilities to each order of magnitude range.
- __default__: Default value to return for the parameter.
    If unspecified, the default value will be
    `min_value`.

- __parent_name__: (Optional) String. Specifies that this hyperparameter is
  conditional. The name of the this hyperparameter's parent.
- __parent_values__: (Optional) List. The values of the parent hyperparameter
  for which this hyperparameter should be considered active.

__Returns:__

The current value of this hyperparameter.

---
### conditional_scope


```python
conditional_scope()
```


Opens a scope to create conditional HyperParameters.

All HyperParameters created under this scope will only be active
when the parent HyperParameter specified by `parent_name` is
equal to one of the values passed in `parent_values`.

When the condition is not met, creating a HyperParameter under
this scope will register the HyperParameter, but will return
`None` rather than a concrete value.

Note that any Python code under this scope will execute
regardless of whether the condition is met.

__Arguments:__

- __parent_name__: The name of the HyperParameter to condition on.
- __parent_values__: Values of the parent HyperParameter for which
  HyperParameters under this scope should be considered valid.
    
---
### get


```python
get(name)
```


Return the current value of this HyperParameter.
