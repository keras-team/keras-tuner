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


from ..oracles import ultraband
from ..engine import tuner as tuner_module
from ..engine import trial as trial_module


class UltraBand(tuner_module.Tuner):
    """Variation of HyperBand algorithm."""

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 **kwargs):
        oracle = ultraband.UltraBand(max_trials)
        super(UltraBand, self).__init__(
            oracle,
            hypermodel,
            objective,
            max_trials,
            **kwargs)

    def run_trial(self, hp, trial_id, *fit_args, **fit_kwargs):
        model = self._build_model(hp)
        trial = trial_module.Trial(trial_id, hp, model, self.objective,
                                   tuner=self, cloudservice=self._cloudservice)
        self.trials.append(trial)

        fit_kwargs['epochs'] = hp.values['epochs']
        for _ in range(self.executions_per_trial):
            execution = trial.run_execution(*fit_args, **fit_kwargs)
        return trial


