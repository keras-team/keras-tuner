from abc import abstractmethod

from kerastuner.abstractions.display import warning


class Collection(object):
    """ Manage a collection of objects

    Attributes:
        _objects (dict): collection of objects
        _last_insert_idx (str): id of the last inserted object
    """

    def __init__(self):
        self._objects = {}  # collection of instance
        self._last_insert_idx = None

    def __len__(self):
        return len(self._objects)

    def add(self, idx, obj):
        """Add object to the collection

        Args:
            idx (str): object index
            obj (Object): Object to add
        """
        if idx in self._objects:
            warning('overriding object %s - use update() instead?' % idx)
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

    def exist(self, idx):
        "is a given idx part of the collection"
        if idx in self._objects:
            return True
        else:
            return False

    def update(self, idx, obj):
        "update a given object"
        self._objects[idx] = obj

    def get_last(self):
        "Returns the last inserted object"
        if self._last_insert_idx:
            return self._objects[self._last_insert_idx]
        else:
            return None

    def to_dict(self):
        "Returns collection as a dict"
        return self._objects

    @abstractmethod
    def to_config(self):
        "return a serializable config"

    def to_list(self, sorted_by=None, reverse=False):
        """Returns collection as a list

        Args:
            sorted_by (function): Function which generates takes two parameters
                (idx, object) and produces the sort key. If `sorted_by` is
                None, the objects are sorted by object ID.
            reverse (bool, optional): Reverse order. Defaults to False.

        Returns:
            list: list of objects sorted by the specified sort key.
        """
        names = sorted(self._objects.keys(), reverse=reverse)
        return [self._objects[name] for name in names]
