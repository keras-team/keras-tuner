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

import pytest

from kerastuner.engine import hyperparameters as hp_module
from tensorflow import keras


def test_base_hyperparameter():
    base_param = hp_module.HyperParameter(name='base', default=0)
    assert base_param.name == 'base'
    assert base_param.default == 0
    assert base_param.get_config() == {'name': 'base', 'default': 0}
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
    with pytest.raises(ValueError, match='Unknown parameter'):
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
        hp.Range('range', min_value=0, max_value=10, step=1, default=0)
    assert hp.values == {
        'choice': 2,
        'scope1/choice': 5,
        'scope1/scope2/choice': 8,
        'scope1/range': 0
    }


def test_conditional_scope():
    hp = hp_module.HyperParameters()
    hp.Choice('choice', [1, 2, 3], default=2)
    with hp.conditional_scope('choice', [1, 3]):
        child1 = hp.Choice('child_choice', [4, 5, 6])
    with hp.conditional_scope('choice', 2):
        child2 = hp.Choice('child_choice', [7, 8, 9])
    assert hp.values == {
        'choice': 2,
        'choice=1|3/child_choice': 4,
        'choice=2/child_choice': 7
    }
    # Assignment to a non-active conditional hyperparameter returns `None`.
    assert child1 is None
    # Assignment to an active conditional hyperparameter returns the value.
    assert child2 == 7


def test_build_with_conditional_scope():

    def build_model(hp):
        model = hp.Choice('model', ['v1', 'v2'])
        with hp.conditional_scope('model', 'v1'):
            v1_params = {'layers': hp.Range('layers', 1, 3),
                         'units': hp.Range('units', 16, 32)}
        with hp.conditional_scope('model', 'v2'):
            v2_params = {'layers': hp.Range('layers', 2, 4),
                         'units': hp.Range('units', 32, 64)}

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
    model = build_model(hp)
    assert hp.values == {
        'model': 'v1',
        'model=v1/layers': 1,
        'model=v1/units': 16,
        'model=v2/layers': 2,
        'model=v2/units': 32
    }


def test_nested_conditional_scopes_and_name_scopes():
    hp = hp_module.HyperParameters()
    a = hp.Choice('a', [1, 2, 3], default=2)
    with hp.conditional_scope('a', [1, 3]):
        b = hp.Choice('b', [4, 5, 6])
        with hp.conditional_scope('b', 6):
            c = hp.Choice('c', [7, 8, 9])
            with hp.name_scope('d'):
                e = hp.Choice('e', [10, 11, 12])
    with hp.conditional_scope('a', 2):
        f = hp.Choice('f', [13, 14, 15])

    assert hp.values == {
        'a': 2,
        'a=1|3/b': 4,
        'a=1|3/b=6/c': 7,
        'a=1|3/b=6/d/e': 10,
        'a=2/f': 13
    }
    # Assignment to an active conditional hyperparameter returns the value.
    assert a == 2
    assert f == 13
    # Assignment to a non-active conditional hyperparameter returns `None`.
    assert b is None
    assert c is None
    assert e is None


def test_Choice():
    choice = hp_module.Choice('choice', [1, 2, 3], default=2)
    choice = hp_module.Choice.from_config(choice.get_config())
    assert choice.default == 2
    assert choice.random_sample() in [1, 2, 3]
    assert choice.random_sample(123) == choice.random_sample(123)
    # No default
    choice = hp_module.Choice('choice', [1, 2, 3])
    assert choice.default == 1
    # None default
    choice = hp_module.Choice('choice', [1, None, 3])
    assert choice.default is None
    with pytest.raises(ValueError, match='default value should be'):
        choice = hp_module.Choice('choice', [1, 2, 3], default=4)


def test_Linear():
    linear = hp_module.Linear(
        'linear', min_value=0.5, max_value=9.5, resolution=0.1, default=9.)
    linear = hp_module.Linear.from_config(linear.get_config())
    assert linear.default == 9.
    assert 0.5 <= linear.random_sample() < 9.5
    assert isinstance(linear.random_sample(), float)
    assert linear.random_sample(123) == linear.random_sample(123)
    # No default
    linear = hp_module.Linear(
        'linear', min_value=0.5, max_value=9.5, resolution=0.1)
    assert linear.default == 0.5


def test_Range():
    rg = hp_module.Range(
        'rg', min_value=5, max_value=9, step=1, default=6)
    rg = hp_module.Range.from_config(rg.get_config())
    assert rg.default == 6
    assert 5 <= rg.random_sample() < 9
    assert isinstance(rg.random_sample(), int)
    assert rg.random_sample(123) == rg.random_sample(123)
    # No default
    rg = hp_module.Range(
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
