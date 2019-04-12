import pytest
import json

from kerastuner.state import State


def test_state_init():
    st = State()
    assert not st.instance
    assert st.executions == []
