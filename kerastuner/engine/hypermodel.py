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
"HyperModel base class."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class HyperModel(object):
    """Defines a searchable space of Models and builds Models from this space.

    Attributes:
        name: The name of this HyperModel.
        tunable: Whether the hyperparameters defined in this hypermodel
          should be added to search space. If `False`, either the search
          space for these parameters must be defined in advance, or the
          default values will be used.
    """

    def __init__(self, name=None, tunable=True):
        self.name = name
        self.tunable = tunable

        self._build = self.build
        self.build = self._tunable_aware_build

    def build(self, hp):
        raise NotImplementedError

    def _tunable_aware_build(self, hp):
        if not self.tunable:
            # Copy `HyperParameters` object so that new entries are not added
            # to the search space.
            hp = hp.copy()
        return self._build(hp)


class DefaultHyperModel(HyperModel):

    def __init__(self, build, name=None, tunable=True):
        super(DefaultHyperModel, self).__init__(name=name)
        self.build = build
