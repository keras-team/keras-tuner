
class Execution(object):
  """Model Execution class. Each Model instance can be executed N time"""

  def __init__(self):
    #final value
    self.acc = -1
    self.loss = -1
    self.val_acc = -1
    self.val_loss = -1
    self.num_epochs = -1

  def record_results(self, results):
    "Record an epoch result"
    self.history = history
    self.acc = max(results.history['acc'])
    self.loss = min(results.history['loss'])
    if 'val_acc' in results.history:
            
      max_acc = max(max_acc, results.history['acc'][-1])
      min_loss = min(min_loss, results.history['loss'][-1])
    stats = {'max_acc': max_acc, 'min_loss': min_loss}
    
    if 'val_acc' in results.history:
        max_val_acc = max(max_val_acc, results.history['val_acc'][-1])
        stats['max_val_acc'] = max_val_acc
    
    if 'val_loss' in results.history:
        min_val_loss = min(min_val_loss, results.history['val_loss'][-1])
        stats['min_val_loss'] = min_val_loss

class Instance(object):
  """Model instance class."""

  def __init__(self, model, idx):
    self.model = model
    self.idx = idx

  def to_json():
    print "FIXME"
    return