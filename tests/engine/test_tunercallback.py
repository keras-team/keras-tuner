from kerastuner.engine.tunercallback import TunerCallback

def test_avg_accuracy_computation():
  logs = {
    "v1_accuracy": 0.8,
    "v2_accuracy": 0.4,
    "val_v1_accuracy": 0.8,
    "val_v2_accuracy": 0.2,
  }
  
  
  #TODO: refactor as as tcb object when writing more tests
  checkpoint = {
    "enable": False,
    "metric": "v1_accuracy",
    "mode": "min"
  }

  info = {"key1": "val1"}
  key_metrics = ['v1_accuracy']
  meta_data = {'meta1': 'val1'}

  tcb = TunerCallback(info, key_metrics , meta_data, checkpoint)
  logs = tcb._TunerCallback__compute_avg_accuracy(logs)
  assert "avg_accuracy" in logs
  assert logs['avg_accuracy'] == 0.6

  assert "val_avg_accuracy" in logs
  assert logs['val_avg_accuracy'] == 0.5