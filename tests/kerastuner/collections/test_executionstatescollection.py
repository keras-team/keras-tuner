import json
import pytest
import os
from pathlib import Path

from kerastuner.collections.executionstatescollection import ExecutionStatesCollection
from kerastuner.states.executionstate import ExecutionState


@pytest.fixture(scope='class')
def metric_config():
    return [{
        "statistics": {
            "min": 0.5,
            "max": 0.5,
            "median": 0.5,
            "mean": 0.5,
            "variance": 0.0,
            "stddev": 0.0
        },
        "history": [0.5, 0.5, 0.5],
        "direction": "max",
        "is_objective": False,
        "best_value": 0.5,
        "name": "accuracy",
        "start_time": 1234,
        "last_value": 0.5,
        "wall_time": [3, 2, 1]}]


@pytest.fixture(scope='class')
def metric_config2():
    return [{
        "statistics": {
            "min": 0.25,
            "max": 0.25,
            "median": 0.25,
            "mean": 0.25,
            "variance": 0.0,
            "stddev": 0.0
        },
        "history": [0.25, 0.25, 0.25],
        "direction": "max",
        "is_objective": False,
        "best_value": 0.5,
        "name": "accuracy",
        "start_time": 1234,
        "last_value": 0.5,
        "wall_time": [4, 3, 2]}]


def test_load_save(metric_config, metric_config2):
    execution_state = ExecutionState(2, metric_config)
    execution_state2 = ExecutionState(3, metric_config2)

    # Override the idx - it's currently set based on time, leading to
    # collisions in tests.
    execution_state.idx = 1233
    execution_state2.idx = 1234

    execution_states = ExecutionStatesCollection()
    execution_states.add(1233, execution_state)
    execution_states.add(1234, execution_state2)

    assert len(execution_states) == 2

    conf = execution_states.to_config()
    execution_states2 = ExecutionStatesCollection.from_config(conf)

    assert len(execution_states2) == 2

    for idx in range(1233, 1235):
        state = execution_states.get(idx)
        state2 = execution_states2.get(idx)

        assert state.max_epochs == state2.max_epochs
        assert state.epochs == state2.epochs
        assert state.idx == state2.idx
        assert state.start_time == state2.start_time
        assert state.eta == state2.eta
        assert json.dumps(state.metrics.to_config()) == json.dumps(
            state2.metrics.to_config())

    assert json.dumps(execution_states.to_config()) == json.dumps(
        execution_states2.to_config())
