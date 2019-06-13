# Copyright 2019 The Keras Tuner Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from abc import abstractmethod

from kerastuner.abstractions.display import fatal


class State(object):
    "Instance state abstraction"

    def __init__(self, **kwargs):
        self.user_parameters = []
        self.to_report = []
        self.kwargs = kwargs

    @abstractmethod
    def get_config(self):
        "return state as an object"

    @abstractmethod
    def summary(self, extended=False):
        "display state status summary"

    def _config_from_attrs(self, attrs):
        """Build a config dict from a list of attributes

        Args:
            attrs (list): list of attributes to build the list from

        Returns:
            dict: generated config dict
        """
        config = {}
        for attr in attrs:
            config[attr] = getattr(self, attr)
        return config

    def _register(self, name, default_value, to_report=False):
        """
        Register a user value and check its value type match what is expected

        Args:
            name (str): Arg name
            default_value: the default value if not supplied by the user
            to_report (bool, optional): Defaults to False. Report as key param?

        Returns:
            value to use
        """

        value = self.kwargs.get(name, default_value)
        if not isinstance(value, type(default_value)):
            fatal('Invalid type for %s -- expected:%s, got:%s' %
                  (name, type(default_value), type(value)))
        self.user_parameters.append(name)
        if to_report:
            self.to_report.append(name)
        return value
