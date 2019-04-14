from __future__ import absolute_import

from .state import State


class DummyState(State):
    "Place holder state to be used at init-time"
    def __init__(self):
        pass

    def to_dict(self):
        return {}
