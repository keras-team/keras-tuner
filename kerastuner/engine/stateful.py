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
"Tuner base class."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf


class Stateful(object):

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def save(self, fname):
        state = self.get_state()
        state_json = json.dumps(state)
        with tf.io.gfile.GFile(fname, 'w') as f:
            f.write(state_json)
        return str(fname)

    def reload(self, fname):
        with tf.io.gfile.GFile(fname, 'r') as f:
            state_data = f.read()
        state = json.loads(state_data)
        self.set_state(state)
