import keras
import time
from termcolor import cprint
from tensorflow.python.lib.io import file_io # allows to write to GCP or local
from collections import defaultdict
from os import path
import json
import numpy as np
class TunerCallback(keras.callbacks.Callback):
	
	def __init__(self, info, key_metrics, log_interval=30):
		"""
		Args:
			log_interval: interval of time in second between the execution stats are written on disk 
		"""
		self.info = info
		self.key_metrics = []
		for k in key_metrics:
			self.key_metrics.append(k[0])

		self.start_ts = int(time.time())
		self.last_write = time.time()
		self.current_epoch_history = defaultdict(list)
		self.current_epoch_key_metrics = defaultdict(list)
		self.history = defaultdict(list)
		self.history_key_metrics = defaultdict(list)
		self.log_interval = log_interval
		self.training_complete = False


	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		self.training_complete = True
		self.__log()
		return

	def on_epoch_begin(self, epoch, logs={}):
		#clearing epoch
		self.current_epoch_history = defaultdict(list)
		self.current_epoch_key_metrics = defaultdict(list)
		return

	def on_epoch_end(self, epoch, logs={}):
		for k,v in logs.items():
			self.history[k].append(v)
			if k in self.key_metrics:
				self.history_key_metrics[k].append(v)
		self.__log()
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		for k,v in logs.items():
			self.current_epoch_history[k].append(v)
			if k in self.key_metrics:
				self.current_epoch_key_metrics[k].append(v)
		if time.time() - self.last_write > self.log_interval:
			self.__log()
		return
	
	def __log(self):
		ts = time.time()
		results  = self.info

		#epoch data must be aggregated
		current_epoch_metrics = {}
		current_epoch_key_metrics = {}
		for k, values in self.current_epoch_history.items():
			if k in ['batch', 'size']:
				continue
			avg = float(np.average(values))
			current_epoch_metrics[k] = avg
			if k in self.current_epoch_key_metrics:
				current_epoch_key_metrics[k] = avg
		results['batch_size'] = self.current_epoch_history['size'][0]
		results['current_epoch_metrics'] = current_epoch_metrics
		results['current_epoch_key_metrics'] = current_epoch_key_metrics

		results['history'] = self.history
		results['ts'] = {"start": self.start_ts, "stop": int(time.time())} 
		results['num_epochs'] = len(self.history)
		results['training_complete'] = self.training_complete

		fname = '%s-%s-%s-execution-results.json' % (self.info['model_name'], self.info['idx'], self.info['execution_idx'])
		output_path = path.join(self.info['local_dir'], fname)
		with file_io.FileIO(output_path, 'w') as outfile:
			outfile.write(json.dumps(results))
		self.last_write = ts