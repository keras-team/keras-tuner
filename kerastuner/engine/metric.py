import sys
import numpy as np

from kerastuner.abstractions.display import fatal


class Metric(object):
    "Training metric object"

    def __init__(self, name, direction):
        """ Initialize a metric object

        Args:
            name (str): metric name
            direction (str): metric direction. One of {'min', 'max'}

        Attributes:
            history (list): metric history
            is_objective (bool): is this metric the main tuning objectived.
            Defaults to False.
        """
        self.name = name
        if direction not in ['min', 'max']:
            fatal('invalid direction. must be in {min, max}')
        self.direction = direction
        self.history = []
        self.is_objective = False

    def update(self, value):
        """ Update metric

        Args:
            value (float): new metric value
        Returns
            Bool: True if the metric improved, false otherwise
        """
        # ensure standard python type for serialization purpose
        value = float(value)
        best_value = self.get_best_value()
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

    def get_statistics(self):
        "Return metric statistics"
        return {
            "min": float(np.min(self.history)),
            "max": float(np.max(self.history)),
            "mean": float(np.mean(self.history)),
            "median": float(np.median(self.history)),
            "variance": float(np.var(self.history)),
            "stddev": float(np.std(self.history))
        }

    def get_last_value(self):
        "Return metric current value"
        if self.history:
            return self.history[-1]
        else:
            return None

    def get_best_value(self):
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

    def get_history(self):
        """return the value history

        Returns:
            list(float): values per epoch
        """
        return self.history

    def to_config(self):
        """Get a serializable dict version of the metric"""
        return {
            "name": self.name,
            "best_value": self.get_best_value(),
            "last_value": self.get_last_value(),
            "direction": self.direction,
            "history": self.history,
            "is_objective": self.is_objective
        }

    @staticmethod
    def from_config(config):
        "Reload metric from config"
        metric = Metric(config['name'], config['direction'])
        metric.history = config['history']
        metric.is_objective = config['is_objective']
        return metric
