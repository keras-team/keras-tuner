from __future__ import absolute_import

import os
from time import time
from collections import defaultdict

from kerastuner.abstractions.display import fatal, set_log, section, subsection
from kerastuner.abstractions.display import display_settings, colorize
from kerastuner.abstractions.display import print_table

from .state import State
from .dummystate import DummyState
from .checkpointstate import CheckpointState
from .tunerstatsstate import TunerStatsState
from .hoststate import HostState


class TunerState(State):
    """
    Keep track tuner state

    Args:
        name (str): tuner name.

        objective (Objective): Which objective the tuner is optimizing for

        epoch_budget (int, optional): how many epochs to hypertune for.
        Defaults to 100.

        max_budget (int, optional): how many epochs to spend at most on
        a given model. Defaults to 10.

        min_budget (int, optional): how many epochs to spend at least on
        a given model. Defaults to 3.

        project (str, optional): project the tuning belong to.
        Defaults to 'default'.

        architecture (str, optional): Name of the architecture tuned.
        Default to timestamp.

        user_info(dict, optional): user supplied information that will be
        recorded alongside training data. Defaults to {}.

        num_executions(int, optional):number of execution per model.
        Defaults to 1.

        max_model_parameters (int, optional):maximum number of parameters
        allowed for a model. Prevent OOO issue. Defaults to 2500000.

        dry_run(bool, optional): Run the tuner without training models.
        Defaults to False.

        debug(bool, optional): Display debug information if true.
        Defaults to False.

        display_model(bool, optional):Display model summary if true.
        Defaults to False.

    Attributes:
        start_time (int): when tuning started
        remaining_budget (int): how many epoch are left
        hyper_parameters (dict): hyper parameters used
    """

    def __init__(self, name, objective, **kwargs):
        super(TunerState, self).__init__(**kwargs)

        self.name = name
        self.start_time = int(time())

        # budget
        self.epoch_budget = self._register('epoch_budget', 100, True)
        self.max_epochs = self._register('max_epochs', 10, True)
        self.min_epochs = self._register('min_epochs', 3, True)
        self.remaining_budget = self.epoch_budget

        # user info
        self.project = self._register('project', 'default')
        self.architecture = self._register('architecture', str(time()))
        self.user_info = self._register('user_info', {})

        # execution
        self.num_executions = self._register('num_executions', 1, True)
        self.hyper_parameters = None  # set in Tuner._check_and_store_model_fn
        self.max_model_parameters = self._register('max_model_parameters',
                                                   25000000, True)

        # debug
        self.dry_run = self._register('dry_run', False)
        self.debug = self._register('debug', False)
        self.display_model = self._register('display_model', '')

        # sub-states
        self.host = HostState(**kwargs)
        self.stats = TunerStatsState()
        self.checkpoint = CheckpointState(**kwargs)

        # logfile
        log_name = "%s_%s_%d.log" % (self.project, self.architecture,
                                     self.start_time)
        log_file = os.path.join(self.host.result_dir, log_name)
        set_log(log_file)

    def summary(self, extended=False):
        """Display a summary of the tuner state

        Args:
            extended (bool, optional):Display an extended summay.
            Defaults to False.
        """
        section('Tuner config')
        subsection('Main parameters')
        summary = {}
        if not extended and not self.debug:
            # collect parameters marked as to report
            for attr in self.to_report:
                summary[attr] = getattr(self, attr)
        else:
            summary = self.to_config()
        display_settings(summary)

        self.checkpoint.summary(extended=extended)
        self.host.summary(extended=extended)

        subsection("Hyper-parmeters search space")
        # Compute the size of the hyperparam space by generating a model
        total_size = 1
        data_by_group = defaultdict(dict)
        group_size = defaultdict(lambda: 1)
        for data in self.hyper_parameters.values():
            data_by_group[data['group']][data['name']] = data['space_size']
            group_size[data['group']] *= data['space_size']
            total_size *= data['space_size']

        # Generate the table.
        rows = [['param', 'space size']]
        for idx, grp in enumerate(sorted(data_by_group.keys())):
            if idx % 2:
                color = 'blue'
            else:
                color = 'default'

            rows.append([colorize(grp, color), ''])
            for param, size in data_by_group[grp].items():
                rows.append([colorize("|-%s" % param, color),
                             colorize(size, color)])

        rows.append(['', ''])
        rows.append([colorize('total', 'magenta'),
                     colorize(total_size, 'magenta')])
        print_table(rows)

    def to_config(self):
        res = {}

        # collect user params
        for name in self.user_parameters:
            res[name] = getattr(self, name)

        # collect programtically defined params
        attrs = ['name', 'start_time', 'remaining_budget', 'hyper_parameters']
        for attr in attrs:
            res[attr] = getattr(self, attr)

        # collect sub components
        res['stats'] = self.stats.to_config()
        res['checkpoint'] = self.checkpoint.to_config()
        res['host'] = self.host.to_config()
        return res
