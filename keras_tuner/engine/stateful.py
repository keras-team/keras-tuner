# Copyright 2019 The KerasTuner Authors
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


import json

import tensorflow as tf


class Stateful:
    """The base class for saving and restoring the state.

    The functionalities in this class is for the user to resume a previously
    stopped program by runing the same code again.

    Usually, the arguments in `__init__()` should not be added to the state. The
    `Stateful` objects can get the arguments again just by running the same code
    even without loading the previous states.

    The state of an object here is the results produced during running the
    program, which can save time for the user if restored.
    """

    def get_state(self):
        """Returns the current state of this object.

        This method is called during `save`.

        Returns:
            A dictionary of serializable objects as the state.
        """
        raise NotImplementedError

    def set_state(self, state):
        """Sets the current state of this object.

        This method is called during `reload`.

        Args:
            state: A dictionary of serialized objects as the state to restore.
        """
        raise NotImplementedError

    def save(self, fname):
        """Saves this object using `get_state`.

        Args:
            fname: A string, the file name to save to.

        Returns:
            String. The serialized state of the object.
        """
        state = self.get_state()
        state_json = json.dumps(state)
        with tf.io.gfile.GFile(fname, "w") as f:
            f.write(state_json)
        return str(fname)

    def reload(self, fname):
        """Reloads this object using `set_state`.

        Args:
            fname: A string, the file name to restore from.
        """
        with tf.io.gfile.GFile(fname, "r") as f:
            state_data = f.read()
        state = json.loads(state_data)
        self.set_state(state)
