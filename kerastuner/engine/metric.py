import sys
import numpy as np
from time import time
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils.multiclass import type_of_target
from kerastuner.abstractions.display import warning, fatal

# TODO - keep this?


def canonicalize_metric_name(name):
    _METRIC_ALIASES = {
        "acc": "accuracy"
    }
    VAL_PREFIX = "val_"

    # Drop the val_, if applicable, temporarily
    is_validation = False
    if name.startswith(VAL_PREFIX):
        name = name[len(VAL_PREFIX):]
        is_validation = True

    name = _METRIC_ALIASES.get(name, name)

    if is_validation:
        return "val_" + name
    else:
        return name


class Metric(object):
    "Training metric object"

    def __init__(self, name, direction):
        """ Initialize a metric object

        Args:
            name (str): metric name
            direction (str): metric direction. One of {'min', 'max'}

        Attributes:
            history (list): metric epoch history
            is_objective (bool): is this metric the main tuning objectived.
            Defaults to False.
            start_time (float): when the metric was created
            wall_time (list): time it took to reach a given epoch from start
            time. Data recorded as float which are delta from start_time.

        """
        self.name = name
        if direction not in ['min', 'max']:
            fatal('invalid direction. must be in {min, max}')
        self.direction = direction
        self.history = []
        self.is_objective = False

        self.start_time = time()
        self.wall_time = []

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
        self.wall_time.append(time() - self.start_time)

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
        if not len(self.history):
            return {}
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
            "statistics": self.get_statistics(),
            "is_objective": self.is_objective,
            "start_time": self.start_time,
            "wall_time": self.wall_time
        }

    @staticmethod
    def from_config(config):
        "Reload metric from config"
        metric = Metric(config['name'], config['direction'])
        metric.history = config['history']
        metric.is_objective = config['is_objective']
        metric.start_time = config['start_time']
        metric.wall_time = config['wall_time']
        return metric


def _compute_continuous_labels(model, x, y):
    predictions = model.predict(x)
    predicted_labels = np.round(predictions)
    actual_labels = y
    return actual_labels, predicted_labels, predictions


def _compute_categorical_labels(model, x, y):
    predictions = model.predict(x)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(y, axis=1)
    return actual_labels, predicted_labels, predictions


def _compute_continuous_classification_metrics(model, validation_data,
                                               label_names=None):
    x, y = validation_data
    actual_labels, predicted_labels, predictions = _compute_continuous_labels(
        model, x, y)
    matrix = confusion_matrix(actual_labels, predicted_labels)
    metrics = classification_report(actual_labels, predicted_labels,
                                    output_dict=True, target_names=label_names)

    data = {
        "actual_labels": actual_labels,
        "predicted_labels": predicted_labels,
        "predicted_probabilities": predictions
    }
    results = {
        "target_type": type_of_target(y),
        "confusion_matrix": matrix.tolist(),
        "classification_metrics": metrics
    }

    return results, data


def _compute_categorical_classification_metrics(model, validation_data,
                                                label_names=None):
    x, y = validation_data
    actual_labels, predicted_labels, predictions = _compute_categorical_labels(
        model, x, y)
    matrix = confusion_matrix(actual_labels, predicted_labels)
    metrics = classification_report(actual_labels, predicted_labels,
                                    output_dict=True, target_names=label_names)

    data = {
        "actual_labels": actual_labels,
        "predicted_labels": predicted_labels,
        "predicted_probabilities": predictions
    }
    results = {
        "target_type": type_of_target(y),
        "confusion_matrix": matrix.tolist(),
        "classification_metrics": metrics
    }
    return results, data


def _compute_common_classification_metrics(model, validation_data,
                                           label_names=None):
    output_type = type_of_target(validation_data[1])
    if output_type == "continuous":
        return _compute_categorical_classification_metrics(model,
                                                           validation_data,
                                                           label_names)
    elif output_type == "categorical":
        return _compute_continuous_classification_metrics(model,
                                                          validation_data,
                                                          label_names)
    else:
        warning("No metrics configured for output type: ", output_type)
        return {"target_type": output_type}, {}


def compute_epoch_end_classification_metrics(model, validation_data,
                                             label_names=None):
    results, _ = _compute_common_classification_metrics(
        model, validation_data, label_names)
    return results


def compute_training_end_classification_metrics(model, validation_data,
                                                label_names=None):
    results, data = _compute_common_classification_metrics(
        model, validation_data, label_names)

    target_type = data["target_type"]

    # data = {
    #     "actual_labels": actual_labels,
    #     "predicted_labels": predicted_labels,
    #     "predicted_probabilities": predictions
    # }
    if target_type == "binary":
        fpr, tpr, thresholds = roc_curve(
            data["actual_labels", "prediction_probabilities"])

        results["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist()
        }
        results["roc_auc_score"] = roc_auc_score(actual_labels, predictions)

    return results
