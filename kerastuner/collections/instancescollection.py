import json

from .collections import Collection
from kerastuner.abstractions.display import get_progress_bar, info
from kerastuner.abstractions.io import read_file, glob


class InstancesCollection(Collection):
    "Manage a collection of instances"

    def __init__(self):
        super(InstancesCollection, self).__init__()

    def to_config(self):
        return self.to_dict()

    def load_from_dir(self, path, project=None, architecture=None):
        """Load instance collection from disk or bucket

        Args:
            path (str): local path or bucket path where instance are stored
            project (str, optional): tuning project name. Defaults to None.
            architecture (str, optional): tuning architecture name.
            Defaults to None.

        Returns:
            int: number of instances loaded
        """
        count = 0
        filenames = glob("%s*-results.json" % path)

        for fname in get_progress_bar(filenames, unit='instance',
                                      desc='Loading instances'):

            data = json.loads(read_file(str(fname)))

            if 'tuner' not in 'data':
                continue
            # Narrow down to matching project and architecture
            if (not architecture or
                    (data['tuner']['architecture'] == architecture)):
                if (data['tuner']['project'] == project or not project):
                    self._objects[data['instance']['idx']] = data
                    count += 1
        info("%s previous instances reloaded" % count)
        return count
