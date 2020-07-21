# Copyright 2020 The Keras Tuner Authors
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
import tensorflow as tf
from kerastuner.engine import multi_execution_tuner
from kerastuner.tuners import randomsearch
from kerastuner.tuners import finetuner


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('finetuner_test', numbered=True)


def build_sequential_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(5))
    model.add(tf.keras.layers.Dense(3))
    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

def test_search_space(tmp_dir):
    # Tests when finetuner is used, search space will include
    # 'unfreeze_factor'.
    oracle = randomsearch.RandomSearchOracle(
        objective='accuracy',
        max_trials=1)
    tuner = finetuner.FineTuner(oracle=oracle,
        hypermodel=build_sequential_model,
        directory=tmp_dir)

    x = np.ones((10, 10))
    y = np.ones((10, 3))
    tuner.search(x, y, epochs=1)
    assert {hp.name for hp in tuner.oracle.get_space().space} == {
        'unfreeze_factor'}


def build_functional_model():
    # in -> a
    #     /   \
    #   e_bn  b -> b1 -> (b)
    #     |   |  
    #     |   d -> d1 -> d2 -> (d)
    #      \ /
    #       f
    #       |
    #       g
    #      / \
    #      \ /
    #       h -> out

    inputs = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.Dense(5, name='a')(inputs)
    x0 = x

    # single bubble
    layer_b = tf.keras.layers.Dense(5, name='b')
    x = layer_b(x)
    x = tf.keras.layers.Dense(5, name='b1')(x)
    x = layer_b(x)

    # double bubble
    layer_d = tf.keras.layers.Dense(5, name='d')
    x = layer_d(x)
    x = tf.keras.layers.Dense(5, name='d1')(x)
    x = tf.keras.layers.Dense(5, name='d2')(x)
    x = layer_d(x)
    
    x0 = tf.keras.layers.BatchNormalization(name='e_bn')(x0)
    x = tf.keras.layers.Add(name='f')([x, x0])
    x, y = tf.keras.layers.Lambda(lambda x: (x, x), name='g')(x)
    x = tf.keras.layers.Add(name='h')([x, y])
    model = tf.keras.Model(inputs, x)
    return model

def test_pseudo_sequential_model():
    model = build_functional_model()    
    ps_model = finetuner.PseudoSequential(model)
    assert {x.name for x in ps_model.blocks[0]} == {'h'}
    assert {x.name for x in ps_model.blocks[1]} == {'g'}
    assert {x.name for x in ps_model.blocks[2]} == {
        'b', 'e_bn', 'b', 'b1', 'd', 'd1', 'd2', 'f'}
    assert {x.name for x in ps_model.blocks[3]} == {'a'}

def test_pseudo_sequential_model_resnet50v2():
    model = tf.keras.applications.ResNet50V2()
    ps_model = finetuner.PseudoSequential(model)

    assert {x.name for x in ps_model.blocks[0]} == {'predictions'}

    # from 'conv5_block2_out' (exclusive) to 'conv5_block3_out' (inclusive)
    names = {x.name for x in ps_model.blocks[4]}
    assert 'conv5_block3_out' in names
    assert 'conv5_block2_out' not in names
    assert len(names) == 11

    names = {x.name for x in ps_model.blocks[-1]}
    assert names == {'conv1_pad'}

def test_pseudo_sequential_model_natively_sequential():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, name='a'))
    model.add(tf.keras.layers.Dense(3, name='b'))
    model.add(tf.keras.layers.Dense(3, name='c'))
    ps_model = finetuner.PseudoSequential(model)

    assert {x.name for x in ps_model.blocks[0]} == {'c'}
    assert {x.name for x in ps_model.blocks[1]} == {'b'}
    assert {x.name for x in ps_model.blocks[2]} == {'a'}


def test_unfreeze_all_include_bn():
    model = build_functional_model()
    finetuner.unfreeze(model, unfreeze_bn=True)
    # model must be set to be trainable for fine grained
    # control of freezing / unfreezing.
    assert model.trainable
    for l in model.layers:
        assert l.trainable

def test_unfreeze_all_exclude_bn():
    model = build_functional_model()
    finetuner.unfreeze(model)
    # model must be set to be trainable for fine grained
    # control of freezing / unfreezing.
    assert model.trainable
    for l in model.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            assert l.trainable == False
        else:
            assert l.trainable

def test_unfreeze_part_include_bn():
    model = build_functional_model()
    finetuner.unfreeze(model, factor=0.3, unfreeze_bn=True)
    assert model.trainable
    trainable_table = {'a': False,
        'e_bn': True,
        'b': False,
        'b1': False,
        'd': False,
        'd1': False,
        'd2': False,
        'f': False,
        'g': True,
        'h': True}

    # skipped input layer
    for l in model.layers[1:]:
        assert trainable_table[l.name] == l.trainable
