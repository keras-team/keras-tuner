""""Ultraband hypertuner

Initial algorithm: https://gist.github.com/ebursztein/8304de052a40058fd0ebaf08c949cc1d

"""

import numpy as np
from termcolor import cprint

from termcolor import cprint
import copy
import sys
import numpy as np
from math import log, ceil
from tqdm import tqdm

from ..engine import Tuner

class UltraBand(Tuner):
  "UltraBand tuner"

  def __init__(self, model_fn, **kwargs):
    """ UltraBand hypertuner initialization
    Args:
      model_name (str): used to prefix results. Default: ts

      epoch_budget (int): how many epochs to spend on optimization. default 1890
      max_epochs (int): number of epoch to train the best model on. Default 45
      min_epochs (int): minimal number of epoch to train model on. Default 3
      halving_ratio (int): ratio used to grow the distribution. Default 3

      executions (int): number of exection for each model tested

      display_model (str): base: cpu/single gpu version, multi-gpu: multi-gpu, both: base and multi-gpu. default (Nothing)

      num_gpu (int): number of gpu to use. Default 0
      gpu_mem (int): amount of RAM per GPU. Used for batch size calculation

      local_dir (str): where to store results and models. Default results/
      gs_dir (str): Google cloud bucket to use to store results and model (optional). Default None

      dry_run (bool): do not train the model just run the pipeline. Default False
      max_fail_streak (int): number of failed model before giving up. Default 20

    FIXME:
     - Deal with early stop correctly
     - allows different halving ratio for epochs and models
     - allows differnet type of distribution

    """

    self.tuner_name = 'UltraBand'
    self.halving_ratio = kwargs.get('halving_ratio', 3)

    self.epoch_budget_expensed = 0
    self.epoch_sequence = self.__geometric_seq(self.halving_ratio , self.min_epochs, self.max_epochs)
    self.model_sequence = list(reversed(self.epoch_sequence)) # FIXME: allows to use another type of sequence
    # clip epoch_sequence to ensure we don't train past max epochs
    self.epoch_sequence[-1] = self.epoch_sequence[-1] - np.sum(self.epoch_sequence[:-1])

    self.band_costs = []
    for i in range(1, len(self.epoch_sequence) + 1):
      s1 = self.epoch_sequence[:i]
      s2 = self.model_sequence[:i]
      self.band_costs.append(np.dot(self.epoch_sequence[:i], self.model_sequence[:i])) #Note: General form, sepecialize sequences have faster way
    self.loop_cost = np.sum(self.band_costs)

    self.num_loops = self.epoch_budget / float(self.loop_cost)
    self.num_bands = len(self.model_sequence)
    self.loop_left = self.num_loops

    cprint('-=[UtraBand Tuning]=-', 'magenta')
    cprint('|- Budget: %s' % self.epoch_budget, 'yellow')
    cprint('|- Num models seq %s' % self.model_sequence, 'yellow' )
    cprint('|- Num epoch seq: %s'%  self.epoch_sequence, 'yellow')
    cprint('|- Bands', 'green' )
    cprint('   |- numbers of band: %s' % len(self.model_sequence), 'green' )
    cprint('   |- cost per band: %s'% self.band_costs, 'green')
    cprint('|- Loops', 'blue')
    cprint('   |- numbers of loop: %s' % self.num_loops, 'blue')
    cprint('   |- cost per loop: %s'% self.loop_cost, 'blue')

    super(UltraBand, self).__init__(model_fn, **kwargs)



  def search(self,x, y, **kwargs):
    while self.loop_left > 0:
      cprint('Budget:%s/%s - Loop %.2f/%.2f' % (self.epoch_budget_expensed, self.epoch_budget, self.loop_left, self.num_loops), 'blue')

      #Last (fractional) loop
      if self.loop_left < 1:
        #Reduce the number of models for the last fractional loop
        self.model_sequence = self.__scale_loop(self.model_sequence, self.loop_left)
        cprint('|- Scaled models seq %s' % self.model_sequence, 'yellow' )
      for band_idx, num_models in enumerate(self.model_sequence):
        band_total_cost = 0
        cprint('Budget:%s/%s - Loop %.2f/%.2f - Bands %s/%s' % (self.epoch_budget_expensed, self.epoch_budget, self.loop_left, self.num_loops, band_idx + 1, self.num_bands), 'green' )
        num_epochs = self.epoch_sequence[band_idx]
        cost = num_models * num_epochs
        self.epoch_budget_expensed += cost
        band_total_cost += cost
        num_steps = len(self.model_sequence[band_idx + 1:])

        ## Generate models
        cprint('|- Generating %s models' % num_models, 'yellow')
        model_instances = []
        kwargs['epochs'] = num_epochs
        if not self.dry_run:
          for _ in tqdm(range(num_models), desc='Generating models', unit='model'):
            model_instances.append(self.new_instance())



        #Training here
        cprint('|- Training %s models for %s epochs' % (num_models, num_epochs), 'yellow')
        kwargs['epochs'] = num_epochs
        if self.dry_run:
          loss_values = model_instances = np.random.rand(num_models) #real training instead
        else:
          loss_values = []
          for instance in model_instances:
            results = instance.fit(x, y, **kwargs)
            loss_values.append(results.history['loss'][-1])
            self.record_results(idx=instance.idx)


        #climbing the band
        for step, num_models in enumerate(self.model_sequence[band_idx + 1:]):
          num_epochs = self.epoch_sequence[step + band_idx + 1]
          cost = num_models * num_epochs
          self.epoch_budget_expensed += cost
          band_total_cost += cost

          # selecting best model
          band_models = self.__sort_models(model_instances, loss_values) #Bogus replace 2nd term with the loss array
          band_models = band_models[:num_models] # halve the m odels

          #train
          cprint('|- Training %s models for an additional %s epochs' % (num_models, num_epochs), 'yellow')
          kwargs['epochs'] = num_epochs
          if self.dry_run:
            loss_values = model_instances = np.random.rand(num_models) #real training instead
          else:
            loss_values = []
            for instance in model_instances:
              results = instance.fit(x, y, **kwargs)
              loss_values.append(results.history['loss'][-1])
              self.record_results(idx=instance.idx)

      self.loop_left -= 1

  def __geometric_seq(self, ratio, min_num_epochs, max_num_epochs, scale_factor=1):
    seq = []
    for i in range(1, max_num_epochs + 1): #will never go over
      val = (scale_factor * ratio)**i
      if val > max_num_epochs:
        if seq[-1] != max_num_epochs:
          seq.append(max_num_epochs)
        if seq[0] != min_num_epochs:
          seq = [min_num_epochs] + seq
        return seq
      if val > min_num_epochs:
        seq.append(val)

  def __scale_loop(self, seq, ratio):
    "Scale the last loop to stay under budget"
    scaled_seq = np.round(np.asarray(seq) * ratio).astype(int) #floor: don't want to go budget
    return scaled_seq

  def __sort_models(self, models, loss_values):
    "Return a sorted list of model by loss rate"
    #FIXME remove early stops
    indices = np.argsort( loss_values ) # recall loss is decreasing so use asc
    sorted_models = []
    for idx in indices:
      sorted_models.append(models[idx])
    return sorted_models