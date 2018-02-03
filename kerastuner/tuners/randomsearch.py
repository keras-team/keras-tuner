from tqdm import tqdm
import copy
import sys
from ..engine import HyperTuner

class RandomSearch(HyperTuner):
    "Basic hypertuner"

    def __init__(self, model_fn, **kwargs):
        """ RandomSearch hypertuner
        Args:
            iterations (int): number of model to test
            executions (int): number of exection for each model tested

            num_gpu (int): number of gpu to use. Default 0
            gpu_mem (int): amount of RAM per GPU. Used for batch size calculation

            dryrun (bool): do not train the model just run the pipeline. Default False
            max_fail_streak (int): number of failed model before giving up. Default 20

        """  
        super(RandomSearch, self).__init__(model_fn, **kwargs)

    def search(self,x, y, **kwargs):
        # Overwrite Keras default verbose value
        if not kwargs.get('verbose', None): 
            kwargs['verbose'] = 0
        # Use progress bar if no verbose
        if kwargs['verbose'] == 0:
            use_progress_bar = True

        if use_progress_bar:
            pb = tqdm(total=self.num_iterations, desc='Instances', unit='instance')
        
        for _ in range(self.num_iterations):
            instance = self.get_random_instance()
            for _ in range(self.num_executions):
                #Note: the results are not used for this tuner.
                results = instance.fit(x, y, **kwargs)
            
            self.record_results()
            if use_progress_bar:
                #pb.set_postfix(stats)
                pb.update(1)