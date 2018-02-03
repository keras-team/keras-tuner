"""Export the best model utility

Args:
  idx: model idx for direct export
  criteria: which criteria used to select model. e.g val_acc, loss ...
  criteria_direction: min or max
  criteria aggr: which statistical aggregation to use for the model:  mean, max, min, avg 
"""