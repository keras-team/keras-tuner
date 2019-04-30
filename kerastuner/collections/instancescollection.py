import json

from .collections import Collection
from kerastuner.states.instancestate import InstanceState
from kerastuner.abstractions.display import progress_bar, info, warning
from kerastuner.abstractions.io import read_file, glob


class InstancesCollection(Collection):
    "Manage a collection of instances"

    def __init__(self):
        super(InstancesCollection, self).__init__()

    def to_config(self):
        return self.to_dict()

    def sort_by_objective(self):
        "Returns instances list sorted by objective"
        instance = self.get_last()
        if not instance:
            warning('No previous instance found')
            return []
        return self.sort_by_metric(instance.objective)

    def sort_by_metric(self, metric_name):
        "Returns instances list sorted by a given metric"

        # checking if metric exist and getting its direction
        instance = self.get_last()
        #!don't use _objects directly use get() instead who do canonicalization
        metric = instance.agg_metrics.get(metric_name)
        if not metric:
            warning('Metric %s not found' % metric_name)
            return []

        # getting metric values
        values = {}
        for instance in self._objects.values():
            value = instance.agg_metrics.get(metric.name).get_best_value()
            # seems wrong but make it easy to sort by value and return instance
            values[value] = instance

        # sorting
        if metric.direction == 'min':
            sorted_values = sorted(values.keys())
        else:
            sorted_values = sorted(values.keys(), reverse=True)

        sorted_instances = []
        for val in sorted_values:
            sorted_instances.append(values[val])

        return sorted_instances

    def load_from_dir(self, path, project='default', architecture='default'):
        """Load instance collection from disk or bucket

        Args:
            path (str): local path or bucket path where instance are stored
            project (str, optional): tuning project name. Defaults to default.
            architecture (str, optional): tuning architecture name.
            Defaults to None.

        Returns:
            int: number of instances loaded
        """
        count = 0
        filenames = glob("%s*-results.json" % path)

        for fname in progress_bar(filenames, unit='instance', desc='Loading'):

            config = json.loads(read_file(str(fname)))

            # check fields existence
            if 'tuner' not in config:
                continue
            if 'architecture' not in config['tuner']:
                continue
            if 'project' not in config['tuner']:
                continue

            # check instance belongs to the right project / architecture
            if (architecture == config['tuner']['architecture'] and
                project == config['tuner']['project']):  # nopep8
                    idx = config['instance']['idx']
                    instance = InstanceState.from_config(config['instance'])
                    self._objects[idx] = instance
                    self._last_insert_idx = idx
                    count += 1

        info("%s previous instances reloaded" % count)
        return count
