import pytest
import os
from pathlib import Path

from kerastuner.collections.instancestatescollection import InstanceStatesCollection


@pytest.fixture()
def data_path():
    data_path = Path(__file__).parents[2]
    data_path = Path.joinpath(data_path, 'data', 'tuner_results')
    return data_path


@pytest.fixture()
def col(data_path):
    col = InstanceStatesCollection()
    count = col.load_from_dir(data_path)
    assert count == 6
    return col


def test_loading(data_path):
    col = InstanceStatesCollection()
    count = col.load_from_dir(data_path)
    assert count == 6


def test_loading_objective_canonicalization(col):
    instance_state = col.get_last()
    assert instance_state.objective == 'val_accuracy'


def test_sort_by_metrics(col):
    instance_states = col.sort_by_metric('loss')
    losses = []
    accuracies = []
    for instance_state in instance_states:
        l = instance_state.agg_metrics.get('loss').get_best_value()
        losses.append(l)
        a = instance_state.agg_metrics.get(
            'val_accuracy').get_best_value()
        accuracies.append(a)
    assert losses[0] < losses[1]
    assert accuracies[0] < accuracies[1]
    assert len(accuracies) == 6
