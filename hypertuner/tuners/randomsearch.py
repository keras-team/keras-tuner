from tqdm import tqdm
import copy
import sys
from ..engine import HyperTuner

class RandomSearch(HyperTuner):
    "Most basic hypertuner"

    def __init__(self, **kwargs):    
        super(RandomSearch, self).__init__(**kwargs)

    def search(self,x, y, **kwargs):
               
        # Overwrite Keras default verbose value
        if not kwargs.get('verbose', None): 
            kwargs['verbose'] = 0
        # Use progress bar if no verbose
        if kwargs['verbose'] == 0:
            use_progress_bar = True

        if use_progress_bar:
            pb = tqdm(total=self.iterations, desc='Searching', unit='mdl')
        
        max_acc = 0
        min_loss = sys.maxint
        max_val_acc = 0
        min_val_loss = sys.maxint
        for _ in range(self.iterations):
            model = self.get_random_model_instance(verbose=0)
            if not model:
                return

            results = model.fit(x, y, **kwargs)

            max_acc = max(max_acc, results.history['acc'][-1])
            min_loss = min(min_loss, results.history['loss'][-1])
            stats = {'max_acc': max_acc, 'min_loss': min_loss}
            
            if 'val_acc' in results.history:
                max_val_acc = max(max_val_acc, results.history['val_acc'][-1])
                stats['max_val_acc'] = max_val_acc
            
            if 'val_loss' in results.history:
                min_val_loss = min(min_val_loss, results.history['val_loss'][-1])
                stats['min_val_loss'] = min_val_loss
            
            if use_progress_bar:
                pb.set_postfix(stats)
                pb.update(1)