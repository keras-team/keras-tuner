from tqdm import tqdm

import hashlib
import os
import queue
from abc import abstractmethod
import traceback

from kerastuner.abstractions.display import fatal
from kerastuner.tools.summary import results_summary as _results_summary
from kerastuner.states import TunerState, InstanceState
from kerastuner.engine.cloudservice import CloudService
from kerastuner.abstractions.io import reload_model
from sklearn.model_selection import train_test_split
from kerastuner import config
from kerastuner.abstractions.display import info, subsection, warning, section
from kerastuner.abstractions.display import display_settings
from kerastuner.abstractions.io import get_weights_filename
from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.collections import InstanceStatesCollection
from kerastuner.distributions import RandomDistributions
from kerastuner.engine.instance import Instance
from .ultraband_config import UltraBandConfig


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

        if not isinstance(mdl, tf.keras.Model):
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
        self.instance_states = InstanceStatesCollection()

        # previous models
        print("Loading from %s" % self.state.host.results_dir)
        count = self.instance_states.load_from_dir(self.state.host.results_dir,
                                                   self.state.project,
                                                   self.state.architecture)
        self.stats.instance_states_previously_trained = count
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

    def search(self, x, y=None, **kwargs):
        kwargs["verbose"] = 0

        validation_data = None

        if "validation_split" in kwargs and "validation_data" in kwargs:
            raise ValueError(
                "Specify validation_data= or validation_split=, but not both.")

        if "validation_split" in kwargs and "validation_data" not in kwargs:
            val_split = kwargs["validation_split"]
            x, x_validation = train_test_split(x, test_size=val_split)
            if y is not None:
                y, y_validation = train_test_split(y, test_size=val_split)
            else:
                y, y_validation = None, None
            validation_data = (x_validation, y_validation)
            del kwargs["validation_split"]
        elif "validation_data" in kwargs:
            validation_data = kwargs["validation_data"]
            del kwargs["validation_data"]

        self.tune(x, y, validation_data=validation_data, **kwargs)
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
                warning("invalid model %s/%s" %
                        (self.stats.invalid_instances, self.max_fail_streak))

                if self.stats.invalid_instances >= self.max_fail_streak:
                    warning("too many consecutive failed models - stopping")
                    return None
                continue

            # stop if the model_fn() return nothing
            if not model:
                warning("No model returned from model function - stopping.")
                return None

            # computing instance unique idx
            idx = self.__compute_model_id(model)
            if self.instance_states.exist(idx):
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
        self.instance_states.add(idx, instance.state)
        return instance

    def reload_instance(self, idx, execution="last", metrics=[]):
        tf_utils.clear_tf_session()

        instance_state = self.instance_states.get(idx)
        if not instance_state:
            raise ValueError("Attempted to reload unknown instance '%s'." %
                             idx)

        # Find the specified execution.
        executions = instance_state.execution_states_collection
        execution_state = None
        if execution == "last":
            execution_state = executions.get_last()
        elif execution == "best":
            execution_state = executions.sort_by_metric(
                self.state.objective).to_list()[0]
        elif execution:
            execution_state = executions.get(idx)

        weights_filename = get_weights_filename(self.state, instance_state,
                                                execution_state)

        model = InstanceState.model_from_configs(
            instance_state.model_config,
            instance_state.loss_config,
            instance_state.optimizer_config,
            instance_state.metrics_config,
            weights_filename=weights_filename)

        instance = Instance(idx=instance_state.idx,
                            model=model,
                            hparams=instance_state.hyper_parameters,
                            tuner_state=self.state,
                            cloudservice=self.cloudservice,
                            instance_state=instance_state)
        return instance

    def get_best_models(self, num_models=1, compile=False):
        """Returns the best models, as determined by the tuner's objective.

        Args:
            num_models (int, optional): Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.
            compile (bool, optional): If True, infer the loss and optimizer,
                and compile the returned models. Defaults to False.

        Returns:
            tuple: Tuple containing a list of InstanceStates, a list of
                ExecutionStates, and a list of Models, where the Nth Instance
                and Nth Execution correspond to the Nth Model.
        """
        sorted_instance_states = self.instance_states.sort_by_objective()
        if len(sorted_instance_states) > num_models:
            sorted_instance_states = sorted_instance_states[:num_models]

        execution_states = []
        for instance_state in sorted_instance_states:
            sorted_execution_list = (
                instance_state.execution_states_collection.sort_by_metric(
                    instance_state.objective))
            best_execution_state = sorted_execution_list[0]
            execution_states.append(best_execution_state)

        models = []
        for instance_state, execution_state in zip(sorted_instance_states,
                                                   execution_states):
            model = reload_model(self.state,
                                 instance_state,
                                 execution_state,
                                 compile=True)
            models.append(model)

        return sorted_instance_states, execution_states, models

    def save_best_model(self, export_type="keras"):
        """Shortcut for save_best_models for the case of only keeping the best
        model."""
        return self.save_best_models(export_type=export_type, num_models=1)

    def save_best_models(self, export_type="keras", num_models=1):
        """ Exports the best model based on the specified metric, to the
            results directory.

            Args:
                output_type (str, optional): Defaults to "keras". What format
                    of model to export:

                    # Tensorflow 1.x/2.x
                    "keras" - Save as separate config (JSON) and weights (HDF5)
                        files.
                    "keras_bundle" - Saved in Keras's native format (HDF5), via
                        save_model()

                    # Currently only supported in Tensorflow 1.x
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

        instance_states, execution_states, models = self.get_best_models(
            num_models=num_models, compile=False)

        zipped = zip(models, instance_states, execution_states)
        for idx, (model, instance_state, execution_state) in enumerate(zipped):
            export_prefix = "%s-%s-%s-%s" % (
                self.state.project, self.state.architecture,
                instance_state.idx, execution_state.idx)

            export_path = os.path.join(self.state.host.export_dir,
                                       export_prefix)

            tmp_path = os.path.join(self.state.host.tmp_dir, export_prefix)
            info("Exporting top model (%d/%d) - %s" %
                 (idx + 1, len(models), export_path))
            tf_utils.save_model(model,
                                export_path,
                                tmp_path=tmp_path,
                                export_type=export_type)

    def __compute_model_id(self, model):
        "compute model hash"
        s = str(model.get_config())
        # Optimizer and loss are not currently part of the model config,
        # but could conceivably be part of the model_fn/tuning process.
        if model.optimizer:
            s += "optimizer:" + str(model.optimizer.get_config())
        s += "loss:" + str(model.loss)
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]

    def results_summary(self, num_models=10, sort_metric=None):
        """Display tuning results summary.

        Args:
            num_models (int, optional): Number of model to display.
            Defaults to 10.
            sort_metric (str, optional): Sorting metric, when not specified
            sort models by objective value. Defaults to None.
        """
        if self.state.dry_run:
            info("Dry-Run - no results to report.")
            return

        # FIXME API documentation
        _results_summary(input_dir=self.state.host.results_dir,
                         project=self.state.project,
                         architecture=self.state.architecture,
                         num_models=num_models,
                         sort_metric=sort_metric)

    @abstractmethod
    def tune(self, x, y, **kwargs):
        "method called by the hypertuner to train an instance"

    @abstractmethod
    def retrain(self, idx, x, y, execution=None, metrics=[], **kwargs):
        "method called by the hypertuner to resume training an instance"
        instance = self.reload_instance(idx,
                                        execution=execution,
                                        metrics=metrics)
        instance.fit(x, y, self.state.max_epochs, **kwargs)


class UltraBand(Tuner):
    def __init__(self, model_fn, objective, **kwargs):
        """ RandomSearch hypertuner
        Args:
            model_fn (function): Function that returns the Keras model to be
            hypertuned. This function is supposed to return a different model
            at every invocation via the use of distribution.* hyperparameters
            range.

            objective (str): Name of the metric to optimize for. The referenced
            metric must be part of the the `compile()` metrics.

        Attributes:
            epoch_budget (int, optional): how many epochs to hypertune for.
            Defaults to 100.

            max_budget (int, optional): how many epochs to spend at most on
            a given model. Defaults to 10.

            min_budget (int, optional): how many epochs to spend at least on
            a given model. Defaults to 3.

            num_executions(int, optional): number of execution for each model.
            Defaults to 1.

            project (str, optional): project the tuning belong to.
            Defaults to 'default'.

            architecture (str, optional): Name of the architecture tuned.
            Default to 'default'.

            user_info(dict, optional): user supplied information that will be
            recorded alongside training data. Defaults to {}.

            label_names (list, optional): Label names for confusion matrix.
            Defaults to None, in which case the numerical labels are used.

            max_model_parameters (int, optional):maximum number of parameters
            allowed for a model. Prevent OOO issue. Defaults to 2500000.

            checkpoint (Bool, optional): Checkpoint model. Setting it to false
            disable it. Defaults to True

            dry_run(bool, optional): Run the tuner without training models.
            Defaults to False.

            debug(bool, optional): Display debug information if true.
            Defaults to False.

            display_model(bool, optional):Display model summary if true.
            Defaults to False.

            results_dir (str, optional): Tuning results dir.
            Defaults to results/. Can specify a gs:// path.

            tmp_dir (str, optional): Temporary dir. Wiped at tuning start.
            Defaults to tmp/. Can specify a gs:// path.

            export_dir (str, optional): Export model dir. Defaults to export/.
            Can specify a gs:// path.

        FIXME:
         - Deal with early stop correctly
         - allows different halving ratio for epochs and models
         - allows differnet type of distribution

        """

        super(UltraBand, self).__init__(model_fn, objective, "UltraBand",
                                        RandomDistributions, **kwargs)

        self.config = UltraBandConfig(kwargs.get('ratio',
                                                 3), self.state.min_epochs,
                                      self.state.max_epochs,
                                      self.state.epoch_budget)

        self.epoch_budget_expensed = 0

        settings = {
            "Epoch Budget": self.state.epoch_budget,
            "Num Models Sequence": self.config.model_sequence,
            "Num Epochs Sequence": self.config.epoch_sequence,
            "Num Brackets": self.config.num_brackets,
            "Number of Iterations": self.config.num_batches,
            "Total Cost per Band": self.config.total_epochs_per_band
        }

        section('UltraBand Tuning')
        subsection('Settings')
        display_settings(settings)

    def __load_instance(self, instance_state):
        if self.state.dry_run:
            return None

        # Determine the weights file (if any) to load, and rebuild the model.
        weights_file = None

        if instance_state.execution_states_collection:
            esc = instance_state.execution_states_collection
            execution_state = esc.get_last()
            weights_file = get_weights_filename(self.state, instance_state,
                                                execution_state)
            if not tf.io.gfile.exists(weights_file):
                warning("Could not open weights file: '%s'" % weights_file)
                weights_file = None

        model = instance_state.recreate_model(weights_filename=weights_file)

        return Instance(instance_state.idx,
                        model,
                        instance_state.hyper_parameters,
                        self.state,
                        self.cloudservice,
                        instance_state=instance_state)

    def __train_instance(self, instance, x, y, **fit_kwargs):
        tf_utils.clear_tf_session()

        # Reload the Instance
        instance = self.__load_instance(instance)

        # Fit the model
        if not self.state.dry_run:
            instance.fit(x, y, **fit_kwargs)

    def __train_bracket(self, instance_collection, num_epochs, x, y,
                        **fit_kwargs):
        "Train all the models that are in a given bracket."
        num_instances = len(instance_collection)

        info('Training %d models for %d epochs.' % (num_instances, num_epochs))
        for idx, instance in enumerate(instance_collection.to_list()):
            info('  Training: %d/%d' % (idx, num_instances))
            self.__train_instance(instance,
                                  x,
                                  y,
                                  epochs=num_epochs,
                                  **fit_kwargs)

    def __filter_early_stops(self, instance_collection, epoch_target):
        filtered_instances = []
        for instance in instance_collection:
            last_execution = instance.execution_states_collection.get_last()
            if not last_execution.metrics or not last_execution.metrics.exist(
                    "loss"):
                info("Skipping instance %s - no metrics." % instance.idx)
                continue
            metric = last_execution.metrics.get("loss")
            epoch_history_len = len(metric.history)
            if epoch_history_len < epoch_target:
                info("Skipping instance %s - history is only %d epochs long - "
                     "expected %d - assuming early stop." %
                     (instance.idx, epoch_history_len, epoch_target))
                continue

            filtered_instances.append(instance)
        return filtered_instances

    def bracket(self, instance_collection, num_to_keep, num_epochs,
                total_num_epochs, x, y, **fit_kwargs):
        output_collection = InstanceStatesCollection()
        if self.state.dry_run:
            for i in range(num_to_keep):
                output_collection.add(i, None)
            return output_collection

        self.__train_bracket(instance_collection, num_epochs, x, y,
                             **fit_kwargs)
        instances = instance_collection.sort_by_objective()
        instances = self.__filter_early_stops(instances, total_num_epochs)

        if len(instances) > num_to_keep:
            instances = instances[:num_to_keep]
            info("Keeping %d instances out of %d" %
                 (len(instances), len(instance_collection)))

        output_collection = InstanceStatesCollection()
        for instance in instances:
            output_collection.add(instance.idx, instance)
        return output_collection

    def search(self, x, y, **kwargs):
        assert 'epochs' not in kwargs, \
            "Number of epochs is controlled by the tuner."
        remaining_batches = self.config.num_batches

        while remaining_batches > 0:
            info('Budget: %s/%s - Loop %.2f/%.2f' %
                 (self.epoch_budget_expensed, self.state.epoch_budget,
                  remaining_batches, self.config.num_batches))

            # Last (fractional) loop
            if remaining_batches < 1.0:
                # Reduce the number of models for the last fractional loop
                model_sequence = self.config.partial_batch_epoch_sequence
                if model_sequence is None:
                    break
                info('Partial Batch Model Sequence %s' % model_sequence)
            else:
                model_sequence = self.config.model_sequence

            # Generate N models, and perform the initial training.
            subsection('Generating %s models' % model_sequence[0])
            candidates = InstanceStatesCollection()
            num_models = model_sequence[0]

            for idx in tqdm(range(num_models),
                            desc='Generating models',
                            unit='model'):

                if self.state.dry_run:
                    candidates.add(idx, None)
                else:
                    instance = self.new_instance()
                    if instance is not None:
                        candidates.add(instance.state.idx, instance.state)

            if not candidates:
                info("No models were generated.")
                break

            subsection("Training models.")

            for bracket_idx, num_models in enumerate(model_sequence):
                num_epochs = self.config.delta_epoch_sequence[bracket_idx]
                total_num_epochs = self.config.epoch_sequence[bracket_idx]

                num_to_keep = 0
                if bracket_idx < len(model_sequence) - 1:
                    num_to_keep = model_sequence[bracket_idx + 1]
                    info("Running a bracket to reduce from %d to %d models "
                         "in %d epochs" %
                         (num_models, num_to_keep, num_epochs))
                else:
                    num_to_keep = model_sequence[bracket_idx]
                    info("Running final bracket - %d models for %d epochs" %
                         (num_to_keep, num_epochs))

                info('Budget: %s/%s - Loop %.2f/%.2f - Brackets %s/%s' %
                     (self.epoch_budget_expensed, self.state.epoch_budget,
                      remaining_batches, self.config.num_batches,
                      bracket_idx + 1, self.config.num_brackets))

                self.epoch_budget_expensed += num_models * num_epochs

                candidates = self.bracket(candidates, num_to_keep, num_epochs,
                                          total_num_epochs, x, y, **kwargs)

            remaining_batches -= 1

        info('Final Budget Used: %s/%s' %
             (self.epoch_budget_expensed, self.state.epoch_budget))

    def run_trail(self, hp):
        pass

    def populate_hyperparameter_values(self, hp):
        pass


class Oracle(object):

    def populate_hyperparameters(self, space):
        raise NotImplementedError

    def update_space(self, additional_hps):
        raise NotImplementedError

    def result(self, values, scalar_result):
        raise NotImplementedError

    def train(self, model, fit_args):
        model.fit(fit_args)


class UltraBandOracle(Oracle):

    def __init__(self, trials):
        super().__init__()
        self.trials = trials
        self.queue = queue.Queue()
        self.during_batch = False
        self._first_time = True
        self._perf_record = {}
        self._current_round = 0
        self.spaces = []
        self._model_sequence = []

    def result(self, values, scalar_result):
        self._perf_record[values] = scalar_result

    def update_space(self, additional_hps):
        pass

    def populate_hyperparameters(self, space):
        if self._first_time:
            self._generate_candidates(space)
            self._first_time = False

        if not self.queue.empty():
            self._copy_values(space, self.queue.get())
            return

        self._current_round += 1
        self._select_candidates()

    def _copy_values(self, space, values):
        pass

    def _generate_candidates(self, space):
        self.spaces = []

    def _select_candidates(self):
        for values, _ in sorted(self._perf_record.items(),
                                key=lambda key, value: value)[self._model_sequence[self._current_round]]:
            self.queue.put(values)

    def train(self, model, fit_args):
        fit_args['epochs'] = None
        model.fit(**fit_args)
