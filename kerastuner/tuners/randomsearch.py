from termcolor import cprint
import copy
import sys
from ..engine import HyperTuner


class RandomSearch(HyperTuner):
    "Basic hypertuner"

    def __init__(self, model_fn, **kwargs):
        """ RandomSearch hypertuner
        Args:
            epoch_budget (int): How many epochs to spend hyper-tuning. Default 3171
            max_epochs (int): How long to train model at most. default 50
            model_name (str): used to prefix results. Default: timestamp

            executions (int): number of execution for each model tested

            display_model (str): base: cpu/single gpu version, multi-gpu: multi-gpu, both: base and multi-gpu. default (Nothing)

            num_gpu (int): number of gpu to use. Default 0
            gpu_mem (int): amount of RAM per GPU. Used for batch size calculation

            local_dir (str): where to store results and models. Default results/
            gs_dir (str): Google cloud bucket to use to store results and model (optional). Default None

            dry_run (bool): do not train the model just run the pipeline. Default False
            max_fail_streak (int): number of failed model before giving up. Default 20

        """
        self.tuner_name = "RandomSearch"
        super(RandomSearch, self).__init__(model_fn, **kwargs)
        self.log.tuner_name(self.tuner_name)

    def hypertune(self, x, y, **kwargs):
        remaining_budget = self.epoch_budget
        num_instances = 0
        while remaining_budget > self.max_epochs:
            instance = self.get_random_instance()
            if not instance:
                self.log.error(
                    "[FATAL] No valid model found - check your model_fn() function is valid"
                )
                return
            num_instances += 1
            self.log.new_instance(instance, num_instances, remaining_budget)
            for cur_execution in range(self.num_executions):
                cprint(
                    "|- execution: %s/%s" % (cur_execution + 1, self.num_executions), 'cyan')
                if self.dry_run:
                    remaining_budget -= self.max_epochs
                else:
                    kwargs['epochs'] = self.max_epochs
                    history = instance.fit(x, y, **kwargs)
                    remaining_budget -= len(history.history['loss'])
                    self.record_results()
            self.statistics()
        self.log.done()
