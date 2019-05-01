import pytest
import os
from pathlib import Path

from kerastuner.collections.instancescollection import InstancesCollection


@pytest.fixture()
def data_path():
    data_path = Path(__file__).parents[2]
    data_path = Path.joinpath(data_path, 'data', 'results')
    return data_path


@pytest.fixture()
def col(data_path):
    col = InstancesCollection()
    count = col.load_from_dir(data_path)
    assert count == 2
    return col


def test_loading(data_path):
    col = InstancesCollection()
    count = col.load_from_dir(data_path)
    assert count == 2


def test_loading_objective_canonicalization(col):
    instance = col.get_last()
    assert instance.objective == 'val_accuracy'


def test_sort_by_metrics(col):
    instances = col.sort_by_metric('loss')
    losses = []
    accuracies = []
    for instance in instances:
        l = instance.agg_metrics.get('loss').get_best_value()
        losses.append(l)
        a = instance.agg_metrics.get('val_acc').get_best_value()
        accuracies.append(a)
    assert losses[0] < losses[1]
    assert accuracies[0] > accuracies[1]
    assert len(accuracies) == 2
