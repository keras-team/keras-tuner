import json

from .collections import Collection
from kerastuner.abstractions.display import progress_bar, info
from kerastuner.abstractions.io import read_file, glob


class InstancesCollection(Collection):
    "Manage a collection of instances"

    def __init__(self):
        super(InstancesCollection, self).__init__()

    def to_config(self):
        return self.to_dict()

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

        for fname in progress_bar(filenames, unit='instance',
                                  desc='Loading instances'):

            data = json.loads(read_file(str(fname)))

            # check fields existance
            if 'tuner' not in data:
                continue
            if 'architecture' not in data['tuner']:
                continue
            if 'project' not in data['tuner']:
                continue

            # check instance belongs to the right project / architecture
            if (architecture == data['tuner']['architecture'] and
                project == data['tuner']['project']):  # nopep8
                    self._objects[data['instance']['idx']] = data
                    count += 1
        info("%s previous instances reloaded" % count)
        return count
