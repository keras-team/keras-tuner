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


import copy
from ..oracles import ultraband
from ..engine import tuner as tuner_module
from ..engine import execution as execution_module
from ..engine import tuner_utils


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

    def run_trial(self, trial, hp, fit_args, fit_kwargs):
        fit_kwargs = copy.copy(fit_kwargs)
        original_callbacks = fit_kwargs.get('callbacks', [])[:]
        for i in range(self.executions_per_trial):
            execution_id = tuner_utils.format_execution_id(
                i, self.executions_per_trial)
            # Patch fit arguments
            max_epochs, max_steps = tuner_utils.get_max_epochs_and_steps(
                fit_args, fit_kwargs)
            fit_kwargs['verbose'] = 0

            # Get model; this will reset the Keras session
            if not self.tune_new_entries:
                hp = hp.copy()
            model = self._build_model(hp)
            self._compile_model(model)

            # Start execution
            execution = execution_module.Execution(
                execution_id=execution_id,
                trial_id=trial.trial_id,
                max_epochs=max_epochs,
                max_steps=max_steps,
                base_directory=trial.directory)
            trial.executions.append(execution)
            self.on_execution_begin(trial, execution, model)

            # During model `fit`,
            # the patched callbacks call
            # `self.on_epoch_begin`, `self.on_epoch_end`,
            # `self.on_batch_begin`, `self.on_batch_end`.
            fit_kwargs['callbacks'] = self._inject_callbacks(
                original_callbacks, trial, execution)
            if 'epochs' in hp.values:
                fit_kwargs['epochs'] = hp.values['epochs']
            if 'trial_id' in hp.values:
                history_trial = self._get_trial(hp.values['trial_id'])
                if history_trial.executions[0].best_checkpoint is not None:
                    best_checkpoint = trial.executions[0].best_checkpoint + '-weights.h5'
                    model.load_weights(best_checkpoint)
            model.fit(*fit_args, **fit_kwargs)
            self.on_execution_end(trial, execution, model)

    def _get_trial(self, trial_id):
        for temp_trial in self.trials:
            if temp_trial.trial_id == trial_id:
                return temp_trial
