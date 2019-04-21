import json

from kerastuner.abstractions.display import get_progress_bar, info, warning
from kerastuner.abstractions.io import read_file, glob


class Collection(object):
    """ Manage a collection of instance

    Attributes:
        _objects (dict): collection of objects
        _last_insert_idx (str): id of the last inserted object
    """

    def __init__(self):
        self._objects = {}  # collection of instance
        self._last_insert_idx = None

    def add(self, idx, obj):
        """Add object to the collection

        Args:
            idx (str): object index
            obj (Object): Object to add
        """
        self._objects[idx] = obj
        self._last_insert_idx = idx

    def get(self, idx):
        """Return the object associated with a given id

        Args:
            idx (str): Object id

        Returns:
            Object: object associated if found or None
        """
        if idx in self._objects:
            return self._objects[idx]
        else:
            warning("%s not found" % idx)
            return None

    def get_last(self):
        return self._objects[self._last_insert_idx]

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
