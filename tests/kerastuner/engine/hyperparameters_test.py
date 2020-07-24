# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from kerastuner.engine import hyperparameters as hp_module
from kerastuner.protos import kerastuner_pb2
from tensorflow import keras


def test_base_hyperparameter():
    base_param = hp_module.HyperParameter(name='base', default=0)
    assert base_param.name == 'base'
    assert base_param.default == 0
    assert base_param.get_config() == {
        'name': 'base', 'default': 0, 'conditions': []}
    base_param = hp_module.HyperParameter.from_config(
        base_param.get_config())
    assert base_param.name == 'base'
    assert base_param.default == 0


def test_hyperparameters():
    hp = hp_module.HyperParameters()
    assert hp.values == {}
    assert hp.space == []
    hp.Choice('choice', [1, 2, 3], default=2)
    assert hp.values == {'choice': 2}
    assert len(hp.space) == 1
    assert hp.space[0].name == 'choice'
    hp.values['choice'] = 3
    assert hp.get('choice') == 3
    hp = hp.copy()
    assert hp.values == {'choice': 3}
    assert len(hp.space) == 1
    assert hp.space[0].name == 'choice'
    with pytest.raises(ValueError, match='does not exist'):
        hp.get('wrong')


def test_name_collision():
    # TODO: figure out how name collision checks
    # should work.
    pass


def test_name_scope():
    hp = hp_module.HyperParameters()
    hp.Choice('choice', [1, 2, 3], default=2)
    with hp.name_scope('scope1'):
        hp.Choice('choice', [4, 5, 6], default=5)
        with hp.name_scope('scope2'):
            hp.Choice('choice', [7, 8, 9], default=8)
        hp.Int('range', min_value=0, max_value=10, step=1, default=0)
    assert hp.values == {
        'choice': 2,
        'scope1/choice': 5,
        'scope1/scope2/choice': 8,
        'scope1/range': 0
    }


def test_parent_name():
    hp = hp_module.HyperParameters()
    hp.Choice('a', [1, 2, 3], default=2)
    b1 = hp.Int(
        'b', 0, 10, parent_name='a', parent_values=1, default=5)
    b2 = hp.Int(
        'b', 0, 100, parent_name='a', parent_values=2, default=4)
    assert b1 is None
    assert b2 == 4
    # Only active values appear in `values`.
    assert hp.values == {
        'a': 2,
        'b': 4
    }


def test_conditional_scope():
    hp = hp_module.HyperParameters()
    hp.Choice('choice', [1, 2, 3], default=2)
    with hp.conditional_scope('choice', [1, 3]):
        child1 = hp.Choice('child_choice', [4, 5, 6])
    with hp.conditional_scope('choice', 2):
        child2 = hp.Choice('child_choice', [7, 8, 9])
    # Only active values appear in `values`.
    assert hp.values == {
        'choice': 2,
        'child_choice': 7
    }
    # Assignment to a non-active conditional hyperparameter returns `None`.
    assert child1 is None
    # Assignment to an active conditional hyperparameter returns the value.
    assert child2 == 7


def test_build_with_conditional_scope():

    def build_model(hp):
        model = hp.Choice('model', ['v1', 'v2'])
        with hp.conditional_scope('model', 'v1'):
            v1_params = {'layers': hp.Int('layers', 1, 3),
                         'units': hp.Int('units', 16, 32)}
        with hp.conditional_scope('model', 'v2'):
            v2_params = {'layers': hp.Int('layers', 2, 4),
                         'units': hp.Int('units', 32, 64)}

        params = v1_params if model == 'v1' else v2_params
        inputs = keras.Input(10)
        x = inputs
        for _ in range(params['layers']):
            x = keras.layers.Dense(params['units'])(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        model.compile('sgd', 'mse')
        return model

    hp = hp_module.HyperParameters()
    build_model(hp)
    assert hp.values == {
        'model': 'v1',
        'layers': 1,
        'units': 16,
    }


def test_nested_conditional_scopes_and_name_scopes():
    hp = hp_module.HyperParameters()
    a = hp.Choice('a', [1, 2, 3], default=3)
    with hp.conditional_scope('a', [1, 3]):
        b = hp.Choice('b', [4, 5, 6], default=6)
        with hp.conditional_scope('b', 6):
            c = hp.Choice('c', [7, 8, 9])
            with hp.name_scope('d'):
                e = hp.Choice('e', [10, 11, 12])
    with hp.conditional_scope('a', 2):
        f = hp.Choice('f', [13, 14, 15])
        with hp.name_scope('g'):
            h = hp.Int('h', 0, 10)

    assert hp.values == {
        'a': 3,
        'b': 6,
        'c': 7,
        'd/e': 10,
    }
    # Assignment to an active conditional hyperparameter returns the value.
    assert a == 3
    assert b == 6
    assert c == 7
    assert e == 10
    # Assignment to a non-active conditional hyperparameter returns `None`.
    assert f is None
    assert h is None


def test_get_with_conditional_scopes():
    hp = hp_module.HyperParameters()
    hp.Choice('a', [1, 2, 3], default=2)
    assert hp.get('a') == 2
    with hp.conditional_scope('a', 2):
        hp.Fixed('b', 4)
        assert hp.get('b') == 4
        assert hp.get('a') == 2
    with hp.conditional_scope('a', 3):
        hp.Fixed('b', 5)
        assert hp.get('b') == 4

    # Value corresponding to the currently active condition is returned.
    assert hp.get('b') == 4


def test_merge_inactive_hp_with_conditional_scopes():
    hp = hp_module.HyperParameters()
    hp.Choice('a', [1, 2, 3], default=3)
    assert hp.get('a') == 3
    with hp.conditional_scope('a', 2):
        hp.Fixed('b', 4)

    hp2 = hp_module.HyperParameters()
    hp2.merge(hp)
    # only active hp should be included to values
    assert 'a' in hp2.values
    assert 'b' not in hp2.values


def test_Choice():
    choice = hp_module.Choice('choice', [1, 2, 3], default=2)
    choice = hp_module.Choice.from_config(choice.get_config())
    assert choice.default == 2
    assert choice.random_sample() in [1, 2, 3]
    assert choice.random_sample(123) == choice.random_sample(123)
    # No default
    choice = hp_module.Choice('choice', [1, 2, 3])
    assert choice.default == 1
    with pytest.raises(ValueError, match='default value should be'):
        hp_module.Choice('choice', [1, 2, 3], default=4)


@pytest.mark.parametrize(
    "values,ordered_arg,ordered_val",
    [([1, 2, 3], True, True),
     ([1, 2, 3], False, False),
     ([1, 2, 3], None, True),
     (['a', 'b', 'c'], False, False),
     (['a', 'b', 'c'], None, False)])
def test_Choice_ordered(values, ordered_arg, ordered_val):
    choice = hp_module.Choice('choice', values, ordered=ordered_arg)
    assert choice.ordered == ordered_val
    choice_new = hp_module.Choice(**choice.get_config())
    assert choice_new.ordered == ordered_val


def test_Choice_ordered_invalid():
    with pytest.raises(ValueError, match='must be `False`'):
        hp_module.Choice('a', ['a', 'b'], ordered=True)


def test_Choice_types():
    values1 = ['a', 'b', 0]
    with pytest.raises(TypeError, match='can contain only one'):
        hp_module.Choice('a', values1)
    values2 = [{'a': 1}, {'a': 2}]
    with pytest.raises(TypeError, match='can contain only `int`'):
        hp_module.Choice('a', values2)


def test_Float():
    # Test with step arg
    linear = hp_module.Float(
        'linear', min_value=0.5, max_value=9.5, step=0.1, default=9.)
    linear = hp_module.Float.from_config(linear.get_config())
    assert linear.default == 9.
    assert 0.5 <= linear.random_sample() <= 9.5
    assert isinstance(linear.random_sample(), float)
    assert linear.random_sample(123) == linear.random_sample(123)

    # Test without step arg
    linear = hp_module.Float(
        'linear', min_value=0.5, max_value=6.5, default=2.)
    linear = hp_module.Float.from_config(linear.get_config())
    assert linear.default == 2.
    assert 0.5 <= linear.random_sample() < 6.5
    assert isinstance(linear.random_sample(), float)
    assert linear.random_sample(123) == linear.random_sample(123)

    # No default
    linear = hp_module.Float(
        'linear', min_value=0.5, max_value=9.5, step=0.1)
    assert linear.default == 0.5


def test_sampling_arg():
    f = hp_module.Float('f', 1e-20, 1e10, sampling='log')
    f = hp_module.Float.from_config(f.get_config())
    assert f.sampling == 'log'

    i = hp_module.Int('i', 0, 10, sampling='linear')
    i = hp_module.Int.from_config(i.get_config())
    assert i.sampling == 'linear'

    with pytest.raises(ValueError, match='`sampling` must be one of'):
        hp_module.Int('j', 0, 10, sampling='invalid')


def test_log_sampling_random_state():
    f = hp_module.Float('f', 1e-3, 1e3, sampling='log')
    rand_sample = f.random_sample()
    assert rand_sample >= f.min_value
    assert rand_sample <= f.max_value

    val = 1e-3
    prob = hp_module.value_to_cumulative_prob(val, f)
    assert prob == 0
    new_val = hp_module.cumulative_prob_to_value(prob, f)
    assert np.isclose(val, new_val)

    val = 1
    prob = hp_module.value_to_cumulative_prob(val, f)
    assert prob == 0.5
    new_val = hp_module.cumulative_prob_to_value(prob, f)
    assert np.isclose(val, new_val)

    val = 1e3
    prob = hp_module.value_to_cumulative_prob(val, f)
    assert prob == 1
    new_val = hp_module.cumulative_prob_to_value(prob, f)
    assert np.isclose(val, new_val)


def test_reverse_log_sampling_random_state():
    f = hp_module.Float('f', 1e-3, 1e3, sampling='reverse_log')
    rand_sample = f.random_sample()
    assert rand_sample >= f.min_value
    assert rand_sample <= f.max_value

    val = 1e-3
    prob = hp_module.value_to_cumulative_prob(val, f)
    assert prob == 0
    new_val = hp_module.cumulative_prob_to_value(prob, f)
    assert np.isclose(val, new_val)

    val = 1
    prob = hp_module.value_to_cumulative_prob(val, f)
    assert prob > 0 and prob < 1
    new_val = hp_module.cumulative_prob_to_value(prob, f)
    assert np.isclose(val, new_val)


def test_Int():
    rg = hp_module.Int(
        'rg', min_value=5, max_value=9, step=1, default=6)
    rg = hp_module.Int.from_config(rg.get_config())
    assert rg.default == 6
    assert 5 <= rg.random_sample() <= 9
    assert isinstance(rg.random_sample(), int)
    assert rg.random_sample(123) == rg.random_sample(123)
    # No default
    rg = hp_module.Int(
        'rg', min_value=5, max_value=9, step=1)
    assert rg.default == 5


def test_Boolean():
    # Test default default
    boolean = hp_module.Boolean('bool')
    assert boolean.default is False
    # Test default setting
    boolean = hp_module.Boolean('bool', default=True)
    assert boolean.default is True
    # Wrong default type
    with pytest.raises(ValueError, match='must be a Python boolean'):
        hp_module.Boolean('bool', default=None)
    # Test serialization
    boolean = hp_module.Boolean('bool', default=True)
    boolean = hp_module.Boolean.from_config(boolean.get_config())
    assert boolean.default is True
    assert boolean.name == 'bool'

    # Test random_sample
    assert boolean.random_sample() in {True, False}
    assert boolean.random_sample(123) == boolean.random_sample(123)


def test_Fixed():
    fixed = hp_module.Fixed('fixed', 'value')
    fixed = hp_module.Fixed.from_config(fixed.get_config())
    assert fixed.default == 'value'
    assert fixed.random_sample() == 'value'

    fixed = hp_module.Fixed('fixed', True)
    assert fixed.default is True
    assert fixed.random_sample() is True

    fixed = hp_module.Fixed('fixed', False)
    fixed = hp_module.Fixed.from_config(fixed.get_config())
    assert fixed.default is False
    assert fixed.random_sample() is False

    fixed = hp_module.Fixed('fixed', 1)
    assert fixed.value == 1
    assert fixed.random_sample() == 1

    fixed = hp_module.Fixed('fixed', 8.2)
    assert fixed.value == 8.2
    assert fixed.random_sample() == 8.2

    with pytest.raises(ValueError, match='value must be an'):
        hp_module.Fixed('fixed', None)


def test_merge():
    hp = hp_module.HyperParameters()
    hp.Int('a', 0, 100)
    hp.Fixed('b', 2)

    hp2 = hp_module.HyperParameters()
    hp2.Fixed('a', 3)
    hp.Int('c', 10, 100, default=30)

    hp.merge(hp2)

    assert hp.get('a') == 3
    assert hp.get('b') == 2
    assert hp.get('c') == 30

    hp3 = hp_module.HyperParameters()
    hp3.Fixed('a', 5)
    hp3.Choice('d', [1, 2, 3], default=1)

    hp.merge(hp3, overwrite=False)

    assert hp.get('a') == 3
    assert hp.get('b') == 2
    assert hp.get('c') == 30
    assert hp.get('d') == 1


def test_float_proto():
    hp = hp_module.Float('a', -10, 10, sampling='linear', default=3)
    proto = hp.to_proto()
    assert proto.name == 'a'
    assert proto.min_value == -10.
    assert proto.max_value == 10.
    assert proto.sampling == kerastuner_pb2.Sampling.LINEAR
    assert proto.default == 3.
    # Zero is the default, gets converted to `None` in `from_proto`.
    assert proto.step == 0.

    new_hp = hp_module.Float.from_proto(proto)
    assert new_hp.get_config() == hp.get_config()


def test_int_proto():
    hp = hp_module.Int('a', 1, 100, sampling='log')
    proto = hp.to_proto()
    assert proto.name == 'a'
    assert proto.min_value == 1
    assert proto.max_value == 100
    assert proto.sampling == kerastuner_pb2.Sampling.LOG
    # Proto stores the implicit default.
    assert proto.default == 1
    assert proto.step == 1

    new_hp = hp_module.Int.from_proto(proto)
    assert new_hp._default == 1
    # Pop the implicit default for comparison purposes.
    new_hp._default = None
    assert new_hp.get_config() == hp.get_config()


def test_choice_proto():
    hp = hp_module.Choice('a', [2.3, 4.5, 6.3], ordered=True)
    proto = hp.to_proto()
    assert proto.name == 'a'
    assert proto.ordered
    assert np.allclose([v.float_value for v in proto.values], [2.3, 4.5, 6.3])
    # Proto stores the implicit default.
    assert np.isclose(proto.default.float_value, 2.3)

    new_hp = hp_module.Choice.from_proto(proto)
    assert new_hp.name == 'a'
    assert np.allclose(new_hp.values, hp.values)
    assert new_hp.ordered
    assert np.isclose(new_hp._default, 2.3)

    # Test int values.
    int_choice = hp_module.Choice('b', [1, 2, 3], ordered=False, default=2)
    new_int_choice = hp_module.Choice.from_proto(int_choice.to_proto())
    assert int_choice.get_config() == new_int_choice.get_config()

    # Test float values.
    float_choice = hp_module.Choice('b', [0.5, 2.5, 4.], ordered=False, default=2.5)
    new_float_choice = hp_module.Choice.from_proto(float_choice.to_proto())
    assert float_choice.get_config() == new_float_choice.get_config()


def _sort_space(hps):
    space = hps.get_config()['space']
    return sorted(space, key=lambda hp: hp['config']['name'])


def test_hyperparameters_proto():
    hps = hp_module.HyperParameters()
    hps.Int('a', 1, 10, sampling='reverse_log', default=3)
    hps.Float('b', 2, 8, sampling='linear', default=4)
    hps.Choice('c', [1, 5, 10], ordered=False, default=5)
    hps.Fixed('d', '3')
    with hps.name_scope('d'):
        hps.Choice('e', [2., 4.5, 8.5], default=2.)
        hps.Choice('f', ['1', '2'], default='1')
        with hps.conditional_scope('f', '1'):
            hps.Int('g', -10, 10, step=2, default=-2)

    new_hps = hp_module.HyperParameters.from_proto(hps.to_proto())
    assert _sort_space(hps) == _sort_space(new_hps)
    assert hps.values == new_hps.values


def test_hyperparameters_values_proto():
    values = kerastuner_pb2.HyperParameters.Values(values={
        'a': kerastuner_pb2.Value(int_value=1),
        'b': kerastuner_pb2.Value(float_value=2.0),
        'c': kerastuner_pb2.Value(string_value='3')})

    # When only values are provided, each param is created as `Fixed`.
    hps = hp_module.HyperParameters.from_proto(values)
    assert hps.values == {'a': 1, 'b': 2.0, 'c': '3'}


def test_dict_methods():
    hps = hp_module.HyperParameters()
    hps.Int('a', 0, 10, default=3)
    hps.Choice('b', [1, 2], default=2)
    with hps.conditional_scope('b', 1):
        hps.Float('c', -10, 10, default=3)
        # Don't allow access of a non-active param within its scope.
        with pytest.raises(ValueError, match='is currently inactive'):
            hps['c']
    with hps.conditional_scope('b', 2):
        hps.Float('c', -30, -20, default=-25)

    assert hps['a'] == 3
    assert hps['b'] == 2
    # Ok to access 'c' here since there is an active 'c'.
    assert hps['c'] == -25
    with pytest.raises(ValueError, match='does not exist'):
        hps['d']

    assert 'a' in hps
    assert 'b' in hps
    assert 'c' in hps
    assert 'd' not in hps


def test_prob_one_choice():
    hp = hp_module.Choice('a', [0, 1, 2])
    # Check that boundaries are valid.
    value = hp_module.cumulative_prob_to_value(1, hp)
    assert value == 2

    value = hp_module.cumulative_prob_to_value(0, hp)
    assert value == 0


def test_return_populated_value_for_new_hp():
    hp = hp_module.HyperParameters()

    hp.values['hp_name'] = 'hp_value'
    assert hp.Choice(
        'hp_name',
        ['hp_value', 'hp_value_default'],
        default='hp_value_default') == 'hp_value'


def test_return_default_value_if_not_populated():
    hp = hp_module.HyperParameters()

    assert hp.Choice(
        'hp_name',
        ['hp_value', 'hp_value_default'],
        default='hp_value_default') == 'hp_value_default'
