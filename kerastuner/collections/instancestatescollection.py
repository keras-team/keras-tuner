import json

from kerastuner.abstractions.display import info, progress_bar, warning
from kerastuner.abstractions.io import glob, read_file
from kerastuner.states.instancestate import InstanceState
from pathlib import Path
from .collections import Collection


class InstanceStatesCollection(Collection):
    "Manage a collection of InstanceStates"

    def __init__(self):
        super(InstanceStatesCollection, self).__init__()

    def to_config(self):
        return self.to_dict()

    def sort_by_objective(self):
        "Returns a list of `InstanceState`s sorted by the objective."
        instance_state = self.get_last()
        if not instance_state:
            warning('No previous instance found')
            return []
        return self.sort_by_metric(instance_state.objective)

    def sort_by_metric(self, metric_name):
        "Returns a list of `InstanceState`s sorted by a given metric."

        # checking if metric exist and getting its direction
        instance_state = self.get_last()
        # !don't use _objects -> use get() instead due to canonicalization
        metric = instance_state.agg_metrics.get(metric_name)
        if not metric:
            warning('Metric %s not found' % metric_name)
            return []

        # getting metric values
        values = {}
        for instance_state in self._objects.values():
            metric = instance_state.agg_metrics.get(metric.name)
            value = metric.get_best_value()
            # seems wrong but make it easy to sort by value and return instance
            values[value] = instance_state

        # sorting
        if metric.direction == 'min':
            sorted_values = sorted(values.keys())
        else:
            sorted_values = sorted(values.keys(), reverse=True)

        sorted_instance_states = []
        for val in sorted_values:
            sorted_instance_states.append(values[val])

        return sorted_instance_states

    def load_from_dir(self, path, project='default', architecture=None,
                      verbose=1):
        """Load instance collection from disk or bucket

        Args:
            path (str): Local path or bucket path where instance results
            are stored

            project (str, optional): Tuning project name. Defaults to default.

            architecture (str, optional): Tuning architecture name.
            Defaults to None.

            verbose (int, optional): Verbose output? Default to 1.

        Returns:
            int: number of instances loaded
        """
        count = 0

        glob_path = str(Path(path) / "*-results.json")
        filenames = glob(glob_path)

        for fname in progress_bar(filenames, unit='instance',
                                  desc='Loading tuning results'):

            config = json.loads(read_file(str(fname)))

            # check fields existence
            if 'tuner' not in config:
                continue
            if 'architecture' not in config['tuner']:                
                continue
            if 'project' not in config['tuner']:
                continue

            # check instance belongs to the right project / architecture
            if (project != config['tuner']['project']):
                print("Rejected %s != %s" % (project, config['tuner']['project']))
                continue

            # Allowing architecture to be None allows to reload models from
            # various architecture for retrain, summary and export purpose
            if (architecture and architecture != config['tuner']['architecture']):  # nopep8
                print("Rejected arch %s != %s" % (architecture, config['tuner']['architecture']))
                continue

            idx = config['instance']['idx']
            instance_state = InstanceState.from_config(config['instance'])
            self._objects[idx] = instance_state
            self._last_insert_idx = idx
            count += 1

        if verbose:
            info("%s previous instances reloaded" % count)

        return count
