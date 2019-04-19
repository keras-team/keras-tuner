from kerastuner.states.tunerstatsstate import TunerStatsState

from .common import is_serializable


def test_serialialization():
    st = TunerStatsState()
    is_serializable(st)
