"""Hyper parameter space definition.

Container for defining, reading and checking parameters. Compatible with Google
Cloud ML Engine hypertuner and the argparse module.

Basic usage:

  >>> hparams_spec = ParameterSpace()

Add an integer parameter named 'layers' with values between 1 and 4.

  >>> hparams_spec.add_integer('layers', 1, 4)

Generate Cloud ML Engine hypertuning configuration:

  >>> hparams_spec.as_cloud_ml_engine_parameter_specs() == \
        [{'parameterName': 'layers', 'type': 'INTEGER', 'maxValue': 4, \
        'scaleType': 'UNIT_LINEAR_SCALE', 'minValue': 1}]
  True

Populate an argparse parser with parameters definitions from hparams_spec:

  >>> import argparse
  >>> parser = argparse.ArgumentParser()
  >>> hparams_spec.populate_arg_parser(parser)
  >>> hparams = parser.parse_args(args=['--layers', '3'])
  >>> hparams.layers
  3

User the parsed args for model definition:

  >>> from keras.models import Sequential
  >>> from keras.layers import Dense, InputLayer
  >>> model = Sequential()
  >>> model.add(InputLayer(input_shape=(5000,)))
  >>> for _ in range(hparams.layers):
  ...   model.add(Dense(256))

Freezing an hyperparameter removes it from the Cloud ML Engine hypertuning
configuration:

  >>> hparams_spec = ParameterSpace()
  >>> hparams_spec.add_integer('layers', 1, 4, default=1, frozen=True)
  >>> hparams_spec.as_cloud_ml_engine_parameter_specs()
  []

Freezed parameters will use the default value in the argument parser object:

  >>> parser = argparse.ArgumentParser()
  >>> hparams_spec.populate_arg_parser(parser)
  >>> hparams = parser.parse_args(args=[])
  >>> hparams.layers
  1

"""

import enum

import attr


class ParameterType(enum.Enum):
  """Implementation of ML Engine v1 ParameterType.

  See
  https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#parametertype
  """
  PARAMETER_TYPE_UNSPECIFIED = 0
  DOUBLE = 1
  INTEGER = 2
  CATEGORICAL = 3
  DISCRETE = 4


class ScaleType(enum.Enum):
  """Implementation of ML Engine v1 ScaleType.

  See
  https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#scaletype
  """
  NONE = 0
  UNIT_LINEAR_SCALE = 1
  UNIT_LOG_SCALE = 2
  UNIT_REVERSE_LOG_SCALE = 3


@attr.s(frozen=True)
class AbstractParameterSpec(object):
  """Class representing a hyperparameter.

  Attributes:
    name: A string, the name of the parameter, as used in the code
    parameter_type: ParameterType
    default: Default value for this hyperparameter
    frozen: A bool, whether the parameter value can vary from the default value
  """
  name = attr.ib(type=str)
  parameter_type = attr.ib(validator=attr.validators.in_(ParameterType))
  default = attr.ib()
  frozen = attr.ib()

  def as_cloudml_engine_parameter_spec(self):
    return {
        'parameterName': self.name,
        'type': self.parameter_type.name,
    }


@attr.s()
class RealParameter(AbstractParameterSpec):
  min_value = attr.ib()
  max_value = attr.ib()
  scale_type = attr.ib(validator=attr.validators.in_(ScaleType))

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def as_cloudml_engine_parameter_spec(self):
    ret = super().as_cloudml_engine_parameter_spec()
    ret.update({
        'minValue': self.min_value,
        'maxValue': self.max_value,
        'scaleType': self.scale_type.name,
    })
    return ret


@attr.s()
class CategoricalParameter(AbstractParameterSpec):
  categorical_values = attr.ib()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def as_cloudml_engine_parameter_spec(self):
    ret = super().as_cloudml_engine_parameter_spec()
    ret.update({
        'categoricalValues': self.categorical_values,
    })
    return ret


@attr.s()
class DiscreteParameter(AbstractParameterSpec):
  discrete_values = attr.ib()
  coalesce = attr.ib()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def as_cloudml_engine_parameter_spec(self):
    ret = super().as_cloudml_engine_parameter_spec()
    ret.update({
        'discreteValues': self.discrete_values,
    })
    return ret


class ParameterSpace(object):
  """Represents a set of hyperparameters.


  """

  def __init__(self):
    self._params = []

  def add_integer(self,
                  name,
                  min_value,
                  max_value,
                  scale_type=ScaleType.UNIT_LINEAR_SCALE,
                  default=None,
                  frozen=False):
    """Adds an integer range to the hyperparameter space."""
    parameter = RealParameter(
        name=name,
        min_value=min_value,
        max_value=max_value,
        scale_type=scale_type,
        default=default,
        frozen=frozen,
        parameter_type=ParameterType.INTEGER)
    self._params.append(parameter)

  def add_double(self,
                 name,
                 min_value,
                 max_value,
                 scale_type=ScaleType.UNIT_LINEAR_SCALE,
                 default=None,
                 frozen=False):
    """Adds a real range to the hyperparameter space."""
    parameter = RealParameter(
        name=name,
        min_value=min_value,
        max_value=max_value,
        scale_type=scale_type,
        default=default,
        frozen=frozen,
        parameter_type=ParameterType.DOUBLE)
    self._params.append(parameter)

  def add_discrete(self,
                   name,
                   discrete_values,
                   default=None,
                   coalesce=float,
                   frozen=False):
    """Adds a discrete range of values to the hyperparameter space."""
    parameter = DiscreteParameter(
        name=name,
        discrete_values=discrete_values,
        default=default,
        coalesce=coalesce,
        frozen=frozen,
        parameter_type=ParameterType.DISCRETE)
    self._params.append(parameter)

  def add_categorical(self, name, discrete_values, default=None, frozen=False):
    """Adds a categorical set of values to the hyperparameter space."""
    parameter = CategoricalParameter(
        name=name,
        categorical_values=discrete_values,
        default=default,
        frozen=frozen,
        parameter_type=ParameterType.CATEGORICAL)
    self._params.append(parameter)

  def as_list(self):
    """Returns the hyperparameters as a list."""
    return self._params

  def as_cloud_ml_engine_parameter_specs(self):
    """Returns a list of parameters usable in a Cloud ML Engine
    HyperparameterSpec.param field."""
    params = [
        param.as_cloudml_engine_parameter_spec()
        for param in self._params
        if not param.frozen
    ]
    return params

  def populate_arg_parser(self, arg_parser):
    """Populates an argparse parser with args matching the parameter space."""
    for param in self.as_list():
      name = '--' + param.name
      if param.parameter_type == ParameterType.INTEGER:
        arg_parser.add_argument(name, type=int, default=param.default)
      elif param.parameter_type == ParameterType.DOUBLE:
        arg_parser.add_argument(name, type=float, default=param.default)
      elif param.parameter_type == ParameterType.DISCRETE:
        arg_parser.add_argument(
            name,
            type=param.coalesce,
            choices=param.discrete_values,
            default=param.default)
      elif param.parameter_type == ParameterType.CATEGORICAL:
        arg_parser.add_argument(
            name, choices=param.categorical_values, default=param.default)
