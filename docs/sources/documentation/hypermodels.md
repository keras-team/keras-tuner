<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/engine/hypermodel.py#L31)</span>
### HyperModel class:

```python
kerastuner.engine.hypermodel.HyperModel(name=None, tunable=True)
```

Defines a searchable space of Models and builds Models from this space.

__Attributes:__

- __name__: The name of this HyperModel.
- __tunable__: Whether the hyperparameters defined in this hypermodel
  should be added to search space. If `False`, either the search
  space for these parameters must be defined in advance, or the
  default values will be used.

----

### build method:


```python
HyperModel.build(hp)
```


Builds a model.

__Arguments:__

- __hp__: A `HyperParameters` instance.

__Returns:__

A model instance.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/applications/xception.py#L22)</span>
### HyperXception class:

```python
kerastuner.applications.xception.HyperXception(include_top=True, input_shape=None, input_tensor=None, classes=None, **kwargs)
```

An Xception HyperModel.

__Arguments:__


- __include_top__: whether to include the fully-connected
    layer at the top of the network.
- __input_shape__: Optional shape tuple, e.g. `(256, 256, 3)`.
      One of `input_shape` or `input_tensor` must be
      specified.
- __input_tensor__: Optional Keras tensor (i.e. output of
    `layers.Input()`) to use as image input for the model.
      One of `input_shape` or `input_tensor` must be
      specified.
- __classes__: optional number of classes to classify images
    into, only to be specified if `include_top` is True,
    and if no `weights` argument is specified.
- __**kwargs__: Additional keyword arguments that apply to all
    HyperModels. See `kerastuner.HyperModel`.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/applications/resnet.py#L25)</span>
### HyperResNet class:

```python
kerastuner.applications.resnet.HyperResNet(include_top=True, input_shape=None, input_tensor=None, classes=None, **kwargs)
```

A ResNet HyperModel.

__Arguments:__


- __include_top__: whether to include the fully-connected
    layer at the top of the network.
- __input_shape__: Optional shape tuple, e.g. `(256, 256, 3)`.
      One of `input_shape` or `input_tensor` must be
      specified.
- __input_tensor__: Optional Keras tensor (i.e. output of
    `layers.Input()`) to use as image input for the model.
      One of `input_shape` or `input_tensor` must be
      specified.
- __classes__: optional number of classes to classify images
    into, only to be specified if `include_top` is True,
    and if no `weights` argument is specified.
- __**kwargs__: Additional keyword arguments that apply to all
    HyperModels. See `kerastuner.HyperModel`.

----

