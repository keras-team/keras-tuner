import keras
import time
from termcolor import cprint
from tensorflow.python.lib.io import file_io # allows to write to GCP or local
from collections import defaultdict
from os import path
import json
class TunerCallback(keras.callbacks.Callback):
	
	def __init__(self, info, key_metrics, log_interval=30):
		"""
		Args:
			log_interval: interval of time in second between the execution stats are written on disk 
		"""
		self.info = info
		self.key_metrics = key_metrics
		self.start_ts = time.time()
		self.last_write = time.time()
		self.current_epoch_history = defaultdict(list)
		self.current_epoch_key_metrics = defaultdict(list)
		self.history = defaultdict(list)
		self.history_key_metrics = defaultdict(list)
		self.log_interval = log_interval


	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		#clearing epoch
		self.current_epoch_history = defaultdict(list)
		self.current_epoch_key_metrics = defaultdict(list)
		for k,v in logs.items():
			self.history[k].append(v)
			if k in self.key_metrics:
				self.history_key_metrics[v]
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
		#cprint("\n%s\n" % logs, 'green')
		return
	
	def __log(self):
		ts = time.time()
		results  = self.info
		results['current_epoch_history'] = self.current_epoch_history
		results['history'] = self.history

		fname = '%s-%s-%s-execution-results.json' % (self.info['model_name'], self.info['idx'], self.info['execution_idx'])
		output_path = path.join(self.info['local_dir'], fname)
		with file_io.FileIO(output_path, 'w') as outfile:
			outfile.write(json.dumps(results))
		self.last_write = ts
			
