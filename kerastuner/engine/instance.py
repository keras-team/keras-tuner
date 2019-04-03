import time
import json
import numpy as np
from os import path
from collections import defaultdict
from tensorflow.python.lib.io import file_io  # allows to write to GCP or local
from tensorflow.keras import backend as K
import tensorflow as tf
import gc
import copy

from .execution import InstanceExecution
from . import backend


class Instance(object):
    """Model instance class."""

    def __init__(self, idx, model, hyper_parameters, meta_data, num_gpu,
                 batch_size, display_model, key_metrics, keras_function,
                 checkpoint, callback_fn, backend):

        self.ts = int(time.time())
        self.training_size = -1
        self.model = model
        self.hyper_parameters = hyper_parameters
        self.optimizer_config = model.optimizer.get_config()

        self.checkpoint = checkpoint
        self.idx = idx

        # ensure meta data dopn't have side effect
        self.meta_data = copy.deepcopy(meta_data)
        self.meta_data['instance'] = idx

        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.display_model = display_model
        self.ts = int(time.time())
        self.executions = []
        self.model_size = self.compute_model_size()
        self.validation_size = 0
        self.results = {}
        self.key_metrics = key_metrics
        self.keras_function = keras_function
        self.callback_fn = callback_fn
        self.backend = backend

    def __get_instance_info(self):
        """Return a dictionary of the model parameters

          Used both for the instance result file and the execution result file
        """
        info = {
            "key_metrics": {},  # key metrics results not metrics definition
            "ts": self.ts,
            "training_size": self.training_size,
            # FIXME: add validation split if needed
            "validation_size": self.validation_size,
            "num_executions": len(self.executions),
            "model": self.model.get_config(),
            "batch_size": self.batch_size,
            "model_size": int(self.model_size),
            "hyper_parameters": self.hyper_parameters,
            "optimizer_config": self.optimizer_config
        }
        return info

    def compute_model_size(self):
        "comput the size of a given model"
        params = [K.count_params(p) for p in set(self.model.trainable_weights)]
        return np.sum(params)

    def fit(self, x, y, resume_execution=False, **kwargs):
        """Fit an execution of the model instance
        Args:
          resume_execution (bool): Instead of creating a new execution,
          resume training the previous one. Default false.
        """

        # in theory for batch training the function is __len__
        # should be implemented - we might need to test the type
        self.training_size = len(x)

        if kwargs.get('validation_data'):
            self.validation_size = len(kwargs['validation_data'][1])

        if resume_execution and len(self.executions):
            execution = self.executions[-1]
            # FIXME: We need to reload the model as it is destroyed
            # at that point / need to be tested
            results = execution.fit(
                x, y, initial_epoch=execution.num_epochs, **kwargs)
        else:
            execution = self.__new_execution()
            results = execution.fit(x, y, **kwargs)
        # compute execution level metrics
        execution.record_results(results)
        return results

    def __new_execution(self):
        num_executions = len(self.executions)

        # ensure that info is only displayed once per iteration
        if num_executions > 0:
            display_model = None
            display_info = False
        else:
            display_info = True
            display_model = self.display_model

        instance_info = self.__get_instance_info()
        execution = InstanceExecution(
            self.model, self.idx, self.meta_data, self.num_gpu, display_model,
            display_info, instance_info, self.key_metrics, self.keras_function,
            self.checkpoint, self.callback_fn, self.backend)
        self.executions.append(execution)
        return execution

    def _clear_gpu_memory(self):
        "Clear tensorflow graph to avoid OOM issues"
        K.clear_session()
        K.get_session().close()

        gc.collect()

        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=cfg))

    def record_results(self):
        """Record training results
        Returns:
          dict: results data
        """

        results = self.__get_instance_info()
        local_dir = self.meta_data['server']['local_dir']

        # collecting executions results
        exec_metrics = defaultdict(lambda: defaultdict(list))
        executions = []  # execution data
        for execution in self.executions:

            # metrics collection
            for metric, data in execution.metrics.items():
                exec_metrics[metric]['min'].append(
                    execution.metrics[metric]['min'])
                exec_metrics[metric]['max'].append(
                    execution.metrics[metric]['max'])

            try:
                json.dumps(execution.model.loss)
                reported_loss_fns = execution.model.loss
            except:
                reported_loss_fns = "CUSTOM"

            # execution data
            execution_info = {
                "num_epochs": execution.num_epochs,
                "history": execution.history,
                "loss_fn": reported_loss_fns,
                "loss_weights": execution.model.loss_weights,
                "meta_data": execution.meta_data,
            }
            executions.append(execution_info)

            # cleanup memory
            del execution.model
            self._clear_gpu_memory()

        results['executions'] = executions
        results['meta_data'] = self.meta_data

        # aggregating statistics
        metrics = defaultdict(dict)
        for metric in exec_metrics.keys():
            for direction, data in exec_metrics[metric].items():
                metrics[metric][direction] = {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "median": float(np.median(data))
                }
        results['metrics'] = metrics

        # Usual metrics reported as top fields for their median values
        for tm in self.key_metrics:
            if tm[0] in metrics:
                results['key_metrics'][tm[0]] = metrics[tm[0]][tm[1]]['median']

        fname = '%s-%s-%s-results.json' % (self.meta_data['project'],
                                           self.meta_data['architecture'],
                                           self.meta_data['instance'])
        local_path = path.join(local_dir, fname)
        with file_io.FileIO(local_path, 'w') as outfile:
            outfile.write(json.dumps(results))

        # cloud recording if needed
        if self.backend:
            self.backend.send_results(results)

        self.results = results
        return results
