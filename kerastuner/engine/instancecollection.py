import json

from kerastuner.abstractions.display import get_progress_bar
from kerastuner.abstractions.io import read_file, glob


class InstanceCollection(object):
    """ Manage a collection of instance

    Args:
        path (str): where instances results are stored
        project (str): id
    """

    def __init__(self):
        self._instances = {}  # collection of instance
        self._last_instance_idx = None

    def add(self, idx, instance):
        """Add instance to the collection

        Args:
            idx (str): Instance idx
            instance (Instance): Instance object
        """
        self._instances[idx] = instance
        self._last_instance_idx = idx

    def get(self, idx):
        "Return the instance associated with an idx"
        if idx in self._instances:
            return self._instances[idx]
        else:
            return None

    def get_last(self):
        return self._instances[self._last_instance_idx]

    def load_from_dir(self, path, project=None, architecture=None):
        """Load instance collection from disk or bucket

        Args:
            path (str): local path or bucket path where instance are stored
            project (str, optional): Tuning project name. Defaults to None.
            architecture (str, optional): Tuning architecture name.
            Defaults to None.

        Returns:
            [type]: [description]
        """
        count = 0
        filenames = glob("%s*-results.json" % path)

        for fname in get_progress_bar(filenames, unit='instance',
                                      desc='Loading instances'):

            data = json.loads(read_file(str(fname)))
            # Narrow down to matching project and architecture
            if (not architecture or
                    (data['tuner']['architecture'] == architecture)):
                if (data['tuner']['project'] == project or not project):
                    self._instances[data['instance']['idx']] = data
                    count += 1
        return count

