import sys
from kerastuner.abstractions.display import fatal


class Metric(object):
    "Training metric object"

    def __init__(self, name, direction, keras_metric=None, is_stateful=True,
                 display=True):
        """ Initialize a metric object

        Args:
            name (str): metric name
            direction (str): metric direction. One of {'min', 'max'}

            keras_metric: Defaults to None. Underlying keras metric, if None
            use name as metric, otherwise use the passed object

            is_stateful (bool, optional): Defaults to True. metric stateful?
            display (bool, optional): Defaults to True. Display in summary?
        """
        self.name = name
        if direction not in ['min', 'max']:
            fatal('invalid direction. must be in {min, max}')
        self.direction = direction
        self.is_stateful = is_stateful
        self.display = display
        self.history = []

        if not keras_metric:
            self.keras_metric = name
        else:
            # FIXME: typecheck that is is a valid metric
            self.keras_metric = keras_metric

    def update(self, value):
        """ Update metric

        Args:
            value (float): new metric value
        Returns
            Bool: True if the metric improved, false otherwise
        """
        # ensure standard python type for serialization purpose
        value = float(value)
        best_value = self._get_best_value()
        self.history.append(value)

        # if no best_value then current is best
        if not best_value:
            return True

        # testing best value vs new taking into account direction
        if self.direction == 'min' and value < best_value:
            return True
        elif self.direction == 'max' and value > best_value:
            return True

        # not the best
        return False

    def get_last_value(self):
        "Return metric current value"
        if self.history:
            return self.history[-1]
        else:
            return None

    def get_best_value(self):
        """Get metric best value

        Returns:
            dict: {mean, variance, max, min}
        """
        return self._get_best_value()

    def get_history(self):
        """return the value history

        Returns:
            list(float): values per epoch
        """
        return self.history

    def to_dict(self):
        """Get a serializable dict version of the metric"""
        return {
            "name": self.name,
            "best_value": self._get_best_value(),
            "last_value": self.get_last_value(),
            "direction": self.direction,
            "is_stateful": self.is_stateful,
            "display": self.display,
            "history": self.history
        }

    def _get_best_value(self):
        """
        Return the current best value
        Returns:
            float: best value
        """
        if self.direction == 'min' and len(self.history):
            return min(self.history)
        elif self.direction == 'max' and len(self.history):
            return max(self.history)
        else:
            return None
