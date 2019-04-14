from __future__ import absolute_import
from abc import abstractmethod


class State(object):
    "Instance state abstraction"

    def __init__(self):
        self.exportable_attributes = []

    @abstractmethod
    def to_dict(self):
        "return state as an object"
