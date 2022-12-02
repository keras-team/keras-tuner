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


class FailedTrialError(Exception):
    """Raise this error to mark a `Trial` as invalid.

    When this error is raised, the `Tuner` would not retry the `Trial` but
    directly mark it as `"INVALID"`.

    Example:

    ```py
    class MyHyperModel(keras_tuner.HyperModel):
        def build(self, hp):
            # Build the model
            ...
            if too_slow(model):
                # Mark the Trial as "INVALID" if the model is too slow.
                raise keras_tuner.FailedTrialError("Model is too slow.")
            return model
    ```
    """

    pass


class FatalError(Exception):
    """Error specially to breakthrough try-except for `Tuner.run_trial()`.

    This error will raised again after caught by the try-except around
    `Tuner.run_trial()`. So, it should only be used in subroutines of
    `Tuner.run_trial()`.

    It is used to terminate the KerasTuner program for errors that need users
    immediate attention.

    """

    pass


class FatalValueError(FatalError, ValueError):
    pass


class FatalTypeError(FatalError, TypeError):
    pass


class FatalRuntimeError(FatalError, RuntimeError):
    pass
