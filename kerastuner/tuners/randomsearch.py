from tqdm import tqdm
import copy
import sys
from ..engine import HyperTuner

class RandomSearch(HyperTuner):
    "Basic hypertuner"

    def __init__(self, model_fn, **kwargs):    
        super(RandomSearch, self).__init__(model_fn, **kwargs)

    def search(self,x, y, **kwargs):
        # Overwrite Keras default verbose value
        if not kwargs.get('verbose', None): 
            kwargs['verbose'] = 0
        # Use progress bar if no verbose
        if kwargs['verbose'] == 0:
            use_progress_bar = True

        if use_progress_bar:
            pb = tqdm(total=self.iterations, desc='Instances', unit='instance')
        
        for _ in range(self.iterations):
            instance = self.get_random_instance()
            if not instance:
                return
            if use_progress_bar:
                pb2 = tqdm(total=self.executions, desc='Executions', unit='model',leave=False)
            
            for _ in range(self.executions):
                execution = instance.new_execution()
                results = execution.model.fit(x, y, **kwargs)
                stats = self.record_results(execution, results)
                pb2.update(1)
            
            if use_progress_bar:
                pb.set_postfix(stats)
                pb.update(1)
        self