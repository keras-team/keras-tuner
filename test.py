import numpy as np
from termcolor import cprint
epoch_budget = 600
max_num_epochs = 45
min_num_epochs = 3
ratio = 3

def geometric_seq(ratio, min_num_epochs, max_num_epochs, scale_factor=1):
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

# assume that max_epoch and min_epoch can be divided by ratio
# otherwise we need to either clip or overgrow for last band
epoch_sequence = geometric_seq(ratio , min_num_epochs, max_num_epochs)
print(epoch_sequence)
model_sequence = list(reversed(epoch_sequence)) # FIXME: allows to use another type of sequence
print(model_sequence)
#computing the cost of each band and the of the full loop
band_costs = []
for i in range(1, len(epoch_sequence) + 1):
  s1 = epoch_sequence[:i]
  s2 = model_sequence[:i]
  print('s1: ', s1, 's2 ', s2)
  band_costs.append(np.dot(epoch_sequence[:i], model_sequence[:i])) #Note: General form, sepecialize sequences have faster way
loop_cost = np.sum(band_costs)
num_loop = epoch_budget / float(loop_cost) 

cprint('epoch_budget: %s' % epoch_budget, 'blue')
cprint('[Bands] numbers %s:' % len(model_sequence), 'blue' )
cprint('[Bands] costs: %s'% band_costs, 'green')
cprint('[Loop] numbers: %s' % num_loop, 'blue')
cprint('[Loop] cost: %s'% loop_cost, 'blue')





#budget_per_bucket = budget/len(sequence) #not correct
# print("sequence ", sequence)
# for step in sequence:
#   # probably can use pop() with a reverse seq aka stack
#   sub_seq = sequence[sequence.index(step):]
#   print("sub_seq: ", sub_seq)
  
#   num_models = sum(sub_seq)
#   print("num models: ", num_models)
#   epochs = max_num_epochs
#   for epochs in reversed(sub_seq[:-1]):
#     epochs = epochs / ratio
    #print("steps ", step, "epochs ", epochs)
  #break

