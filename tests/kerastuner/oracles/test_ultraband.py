from kerastuner.engine import hyperparameters as hp_module
from kerastuner.oracles.ultraband import *


def test_ultraband_oracle():
    hp_list = [hp_module.Choice('a', [1, 2], default=1),
               hp_module.Choice('b', [3, 4], default=3),
               hp_module.Choice('c', [5, 6], default=5),
               hp_module.Choice('d', [7, 8], default=7),
               hp_module.Choice('e', [9, 0], default=9)]
    oracle = UltraBand(trials=34)
    assert oracle._num_brackets == 3

    oracle.populate_space('x', [])

    for trial_id in range(oracle._model_sequence[0]):
        assert oracle.populate_space(str(trial_id), hp_list)['status'] == 'RUN'
    assert oracle.populate_space('idle0', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[0]):
        oracle.result(str(trial_id), trial_id)

    for trial_id in range(oracle._model_sequence[1]):
        assert oracle.populate_space('1_' + str(trial_id), hp_list)['status'] == 'RUN'
    assert oracle.populate_space('idle1', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[1]):
        oracle.result('1_' + str(trial_id), trial_id)

    for trial_id in range(oracle._model_sequence[2]):
        assert oracle.populate_space('2_' + str(trial_id), hp_list)['status'] == 'RUN'
        oracle.result('2_' + str(trial_id), trial_id)

    for trial_id in range(oracle._model_sequence[0]):
        assert oracle.populate_space('0_' + str(trial_id), hp_list)['status'] == 'RUN'

    assert oracle.populate_space('idle2', hp_list)['status'] == 'IDLE'

    for trial_id in range(oracle._model_sequence[0]):
        oracle.result('0_' + str(trial_id), trial_id)

    assert oracle.populate_space('last', hp_list)['status'] == 'RUN'
    oracle.result('last', 0)
    assert oracle.populate_space('exit', hp_list)['status'] == 'EXIT'
