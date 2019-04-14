from __future__ import absolute_import
from abc import abstractmethod


class State(object):
    "Instance state abstraction"

    def __init__(self):
        self.exportable_attributes = []

    def to_dict(self):
        "return state as an object"
        export = {}
        for attribute in self.exportable_attributes:
            val = getattr(self, attribute)
            export[attribute] = val
