import pytest

from keras_tuner.engine import oracle as oracle_module


class OracleStub(oracle_module.Oracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_trial_called = False

    def populate_space(self, trial_id):
        return "populate_space"

    def score_trial(self, trial_id):
        self.score_trial_called = True


def test_private_populate_space_deprecated_and_call_public():
    oracle = OracleStub(objective="val_loss")
    with pytest.deprecated_call():
        assert oracle._populate_space("100") == "populate_space"


def test_private_score_trial_deprecated_and_call_public():
    oracle = OracleStub(objective="val_loss")
    with pytest.deprecated_call():
        oracle._score_trial("trial")
    assert oracle.score_trial_called
