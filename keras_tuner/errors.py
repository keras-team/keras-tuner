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


from keras_tuner.api_export import keras_tuner_export


@keras_tuner_export(["keras_tuner.errors.FailedTrialError"])
class FailedTrialError(Exception):
    """Raise this error to mark a `Trial` as failed.

    When this error is raised in a `Trial`, the `Tuner` would not retry the
    `Trial` but directly mark it as `"FAILED"`.

    Example:

    ```py
    class MyHyperModel(keras_tuner.HyperModel):
        def build(self, hp):
            # Build the model
            ...
            if too_slow(model):
                # Mark the Trial as "FAILED" if the model is too slow.
                raise keras_tuner.FailedTrialError("Model is too slow.")
            return model
    ```
    """

    pass


@keras_tuner_export(["keras_tuner.errors.FatalError"])
class FatalError(Exception):
    """A fatal error during search to terminate the program.

    It is used to terminate the KerasTuner program for errors that need
    users immediate attention. When this error is raised in a `Trial`, it will
    not be caught by KerasTuner.
    """

    pass


@keras_tuner_export(["keras_tuner.errors.FatalValueError"])
class FatalValueError(FatalError, ValueError):
    """A fatal error during search to terminate the program.

    It is a subclass of `FatalError` and `ValueError`.

    It is used to terminate the KerasTuner program for errors that need
    users immediate attention. When this error is raised in a `Trial`, it will
    not be caught by KerasTuner.
    """

    pass


@keras_tuner_export(["keras_tuner.errors.FatalTypeError"])
class FatalTypeError(FatalError, TypeError):
    """A fatal error during search to terminate the program.

    It is a subclass of `FatalError` and `TypeError`.

    It is used to terminate the KerasTuner program for errors that need
    users immediate attention. When this error is raised in a `Trial`, it will
    not be caught by KerasTuner.
    """

    pass


@keras_tuner_export(["keras_tuner.errors.FatalRuntimeError"])
class FatalRuntimeError(FatalError, RuntimeError):
    """A fatal error during search to terminate the program.

    It is a subclass of `FatalError` and `RuntimeError`.

    It is used to terminate the KerasTuner program for errors that need
    users immediate attention. When this error is raised in a `Trial`, it will
    not be caught by KerasTuner.
    """

    pass
