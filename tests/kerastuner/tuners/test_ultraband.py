import json
import time

import numpy as np
import pytest
import sklearn
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from sklearn.utils.multiclass import type_of_target
from kerastuner.tuners.ultraband import UltraBand
from kerastuner.engine.metric import compute_common_classification_metrics


def basic_model_fn():
    # *can't pass as fixture as the tuner expects a function
    i = Input(shape=(1,), name="input")
    o = Dense(4, name="output")(i)
    model = Model(inputs=i, outputs=o)
    model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
    return model

@pytest.fixture
def ultraband():
    return UltraBand(
        basic_model_fn, 
        'loss', 
        dry_run=True, 
        epoch_budget=1900,
        model_name="test_geometric_sequence")

def test_geometric_sequence(ultraband):
    seq = ultraband._geometric_seq(1.0/3, 3, 27, scale_factor=1)
    print(seq)
    assert 2== 3
