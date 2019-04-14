from kerastuner.states.hypertunerstatsstate import HypertunerStatsState

from .common import is_serializable


def test_serialialization():
    st = HypertunerStatsState()
    is_serializable(st)
