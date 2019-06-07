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

"Basic randomsearch tuner"
from termcolor import cprint
from kerastuner.engine import Tuner
from kerastuner.abstractions.display import subsection
from kerastuner.distributions import RandomDistributions
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils


class RandomSearch(Tuner):
    "Basic hypertuner"

    def __init__(self, model_fn, objective, **kwargs):
        """ RandomSearch hypertuner
        Args:
            model_fn (function): Function that returns the Keras model to be
            hypertuned. This function is supposed to return a different model
            at every invocation via the use of distribution.* hyperparameters
            range.

            objective (str): Name of the metric to optimize for. The referenced
            metric must be part of the the `compile()` metrics.

        Attributes:
            epoch_budget (int, optional): how many epochs to hypertune for.
            Defaults to 100.

            max_budget (int, optional): how many epochs to spend at most on
            a given model. Defaults to 10.

            min_budget (int, optional): how many epochs to spend at least on
            a given model. Defaults to 3.

            num_executions(int, optional): number of execution for each model.
            Defaults to 1.

            project (str, optional): project the tuning belong to.
            Defaults to 'default'.

            architecture (str, optional): Name of the architecture tuned.
            Default to 'default'.

            user_info(dict, optional): user supplied information that will be
            recorded alongside training data. Defaults to {}.

            label_names (list, optional): Label names for confusion matrix.
            Defaults to None, in which case the numerical labels are used.

            max_model_parameters (int, optional):maximum number of parameters
            allowed for a model. Prevent OOO issue. Defaults to 2500000.

            checkpoint (Bool, optional): Checkpoint model. Setting it to false
            disable it. Defaults to True

            dry_run(bool, optional): Run the tuner without training models.
            Defaults to False.

            debug(bool, optional): Display debug information if true.
            Defaults to False.

            display_model(bool, optional):Display model summary if true.
            Defaults to False.

            results_dir (str, optional): Tuning results dir.
            Defaults to results/. Can specify a gs:// path.

            tmp_dir (str, optional): Temporary dir. Wiped at tuning start.
            Defaults to tmp/. Can specify a gs:// path.

            export_dir (str, optional): Export model dir. Defaults to export/.
            Can specify a gs:// path.

        """

        # Do the super last -- it uses the variable setup by the tuner
        super(RandomSearch, self).__init__(model_fn, objective, 'RandomSearch',
                                           RandomDistributions, **kwargs)

    def tune(self, x, y, **kwargs):
        while self.state.remaining_budget:
            instance = self.new_instance()
            # not instances left time to wrap-up
            if not instance:
                break

            # train n executions for the given model
            for _ in range(self.state.num_executions):
                execution = instance.fit(x, y, self.state.max_epochs, **kwargs)  # nopep8 pylint: disable=unused-variable

