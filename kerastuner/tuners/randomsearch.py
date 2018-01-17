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
            pb = tqdm(total=self.iterations, desc='Searching', unit='mdl')
        
        for _ in range(self.iterations):
            instance = self.get_random_instance()
            if not instance:
                return
            results = instance.model.fit(x, y, **kwargs)
            stats = self.record_results(instance, results)
            
            if use_progress_bar:
                pb.set_postfix(stats)
                pb.update(1)