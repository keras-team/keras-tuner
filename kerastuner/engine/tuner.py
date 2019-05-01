"Meta classs for hypertuner"
from __future__ import absolute_import, division, print_function

import gc
import hashlib
import json
import os
import socket
import sys
import time
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
import traceback

# used to check if supplied model_fn is a valid model
from tensorflow.keras.models import Model  # pylint: disable=import-error
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.abstractions.display import highlight, display_table, section
from kerastuner.abstractions.display import display_setting, display_settings
from kerastuner.abstractions.display import info, warning, fatal, set_log
from kerastuner.abstractions.display import progress_bar, subsection
from kerastuner.abstractions.display import colorize, colorize_default
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.tools.summary import results_summary as _results_summary
from kerastuner import config
from kerastuner.states import TunerState
from .cloudservice import CloudService
from .instance import Instance
from kerastuner.collections import InstancesCollection


class Tuner(object):
    """Abstract hypertuner class."""

    def __init__(self, model_fn, objective, name, distributions, **kwargs):
        """ Tuner abstract class

        Args:
            model_fn (function): Function that return a Keras model
            name (str): name of the tuner
            objective (str): Which objective the tuner optimize for
            distributions (Distributions): distributions object

        Notes:
            All meta data and varialbles are stored into self.state
            defined in ../states/tunerstate.py
        """

        # hypertuner state init
        self.state = TunerState(name, objective, **kwargs)
        self.stats = self.state.stats  # shorthand access
        self.cloudservice = CloudService()

        # check model function
        if not model_fn:
            fatal("Model function can't be empty")
        try:
            mdl = model_fn()
        except:
            traceback.print_exc()
            fatal("Invalid model function")

        if not isinstance(mdl, Model):
            t = "tensorflow.keras.models.Model"
            fatal("Invalid model function: Doesn't return a %s object" % t)

        # function is valid - recording it
        self.model_fn = model_fn

        # Initializing distributions
        hparams = config._DISTRIBUTIONS.get_hyperparameters_config()
        if len(hparams) == 0:
            warning("No hyperparameters used in model function. Are you sure?")

        # set global distribution object to the one requested by tuner
        # !MUST be after _eval_model_fn()
        config._DISTRIBUTIONS = distributions(hparams)

        # instances management
        self.max_fail_streak = 5  # how many failure before giving up
        self.instances = InstancesCollection()

        # previous models
        count = self.instances.load_from_dir(self.state.host.result_dir,
                                             self.state.project,
                                             self.state.architecture)
        self.stats.instances_previously_trained = count
        info("Tuner initialized")

    def summary(self, extended=False):
        """Print tuner summary

        Args:
            extended (bool, optional): Display extended summary.
            Defaults to False.
        """
        section('Tuner summary')
        self.state.summary(extended=extended)
        config._DISTRIBUTIONS.config_summary()

    def enable_cloud(self, api_key, url=None):
        """Enable cloud service reporting

            Args:
                api_key (str): The backend API access token.
                url (str, optional): The backend base URL.

            Note:
                this is called by the user
        """
        self.cloudservice.enable(api_key, url)

    def search(self, x, y, **kwargs):
        self.state.keras_function = 'fit'
        kwargs["verbose"] = 0
        self.tune(x, y, **kwargs)
        if self.cloudservice.is_enable:
            self.cloudservice.complete()

    def search_generator(self, x, **kwargs):
        self.state.keras_function = 'fit_generator'
        kwargs["verbose"] = 0
        # fit_generator don't use y this so we use None instead
        self.tune(x, None, **kwargs)
        if self.cloudservice.is_enable:
            self.cloudservice.complete()

    def new_instance(self):
        "Return a never seen before model instance"
        fail_streak = 0
        collision_streak = 0
        over_sized_streak = 0

        while 1:
            # clean-up TF graph from previously stored (defunct) graph
            tf_utils.clear_tf_session()
            self.stats.generated_instances += 1
            fail_streak += 1
            try:
                model = self.model_fn()
            except:
                if self.state.debug:
                    traceback.print_exc()

                self.stats.invalid_instances += 1
                warning("invalid model %s/%s" % (self.stats.invalid_instances,
                                                 self.max_fail_streak))

                if self.stats.invalid_instances >= self.max_fail_streak:
                    warning("too many consecutive failed model - stopping")
                    return None
                continue

            # stop if the model_fn() return nothing
            if not model:
                warning("No model returned from model function - stopping.")
                return None

            # computing instance unique idx
            idx = self.__compute_model_id(model)
            if self.instances.exist(idx):
                collision_streak += 1
                self.stats.collisions += 1
                warning("Collision for %s -- skipping" % (idx))
                if collision_streak >= self.max_fail_streak:
                    return None
                continue

            # check size
            nump = tf_utils.compute_model_size(model)
            if nump > self.state.max_model_parameters:
                over_sized_streak += 1
                self.stats.over_sized_models += 1
                warning("Oversized model: %s parameters-- skipping" % (nump))
                if over_sized_streak >= self.max_fail_streak:
                    warning("too many consecutive failed model - stopping")
                    return None
                continue

            # creating instance
            hparams = config._DISTRIBUTIONS.get_hyperparameters()
            instance = Instance(idx, model, hparams, self.state,
                                self.cloudservice)
            break

        # recording instance
        self.instances.add(idx, instance)
        return instance

    def get_best_model(self, **kwargs):
        instances, executions, models = self.get_best_models(
            num_models=1, **kwargs)
        return instances[0], executions[0], models[0]

    def get_model(self, instance, execution, compile=False):
        # TODO - this needs to go somewhere common. It's used here and in monitorcallback
        base_prefix = "%s-%s-%s" % (self.state.project,
                                    self.state.architecture, instance.state.idx)

        results_file = os.path.join(
            self.state.host.result_dir, base_prefix + "-results.json")

        base_prefix = "%s-%s" % (base_prefix, execution.state.idx)

        config_file = os.path.join(
            self.state.host.result_dir, base_prefix + "-config.json")
        h5_file = os.path.join(self.state.host.result_dir,
                               base_prefix + "-weights.h5")

        model = tf_utils.reload_model(config_file, h5_file,
                                      results_file, compile=compile)
        return model

    def get_best_models(self, num_models=1, compile=False):
        objective = self.state.agg_metrics.get_objective()

        instances = self.instances.get_best_instances(
            objective,
            N=num_models)

        models = []
        executions = []
        for instance in instances:
            best_execution = instance.executions.get_best_executions(
                objective, N=1)
            best_execution = best_execution[0]
            model = self.get_model(instance, best_execution, compile=compile)
            models.append(model)
            executions.append(best_execution)

        return instances, executions, models

    def save_best_model(self, **kwargs):
        return self.save_best_models(num_models=1, **kwargs)

    def save_best_models(self, output_type="keras", num_models=1):
        """ Exports the best model based on the specified metric, to the
            results directory.

            Args:
                output_type (str, optional): Defaults to "keras". What format
                    of model to export:

                    "keras" - Save as separate config (JSON) and weights (HDF5)
                        files.
                    "keras_bundle" - Saved in Keras's native format (HDF5), via
                        save_model()
                    "tf" - Saved in tensorflow's SavedModel format. See:
                        https://www.tensorflow.org/alpha/guide/saved_model
                    "tf_frozen" - A SavedModel, where the weights are stored
                        in the model file itself, rather than a variables
                        directory. See:
                        https://www.tensorflow.org/guide/extend/model_files
                    "tf_optimized" - A frozen SavedModel, which has
                        additionally been transformed via tensorflow's graph
                        transform library to remove training-specific nodes
                        and operations.  See:
                        https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms
                    "tf_lite" - A TF Lite model.
        """

        instances, executions, models = self.get_best_models(
            num_models=num_models, compile=False)

        for idx, (model, instance, execution) in enumerate(zip(models, instances, executions)):
            export_prefix = "%s-%s-%s-%s" % (
                self.state.project,
                self.state.architecture,
                instance.state.idx,
                execution.state.idx)

            export_path = os.path.join(
                self.state.host.export_dir, export_prefix)
            tmp_path = os.path.join(self.state.host.tmp_dir, export_prefix)
            info("Exporting top model (%d/%d) - %s" %
                 (idx + 1, len(models), export_path))
            tf_utils.save_model(model, export_path, tmp_path=tmp_path,
                                output_type=output_type)

    def __compute_model_id(self, model):
        "compute model hash"
        s = str(model.get_config())
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]

    def results_summary(self, num_models=10, sort_metric=None):
        _results_summary(input_dir=self.state.host.result_dir,
                         project=self.state.project,
                         architecture=self.state.architecture,
                         num_models=10, sort_metric=sort_metric)

    @abstractmethod
    def tune(self, x, y, **kwargs):
        "method called by the hypertuner to train an instance"
