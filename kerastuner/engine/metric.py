import sys
import numpy as np
from time import time
import sklearn
from kerastuner.abstractions.display import warning, fatal
import traceback


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
    def from_config(config, with_values=True):
        """Reload metric from config

        Args:
            config (dict): Configuration dictionary, as returned by to_config.
            with_values (bool, optional): If True, metric values are copied from
                the configuration. If False, they are omitted. Defaults to True.

        Returns:
            Metric: The Metric object defined by the config.
        """
        metric = Metric(config['name'], config['direction'])
        if with_values:
            metric.history = config['history']
            metric.start_time = config['start_time']
            metric.wall_time = config['wall_time']
        metric.is_objective = config['is_objective']
        return metric


def _is_supported(y):
    output_type = sklearn.utils.multiclass.type_of_target(y)
    valid_types = [
        "multilabel-indicator", "binary", "continuous",
        "multiclass-multioutput"
    ]
    return output_type in valid_types


def _convert_labels(y):
    argmax_types = [
        "multilabel-indicator", "multiclass-multioutput",
        "continuous-multioutput"
    ]

    output_type = sklearn.utils.multiclass.type_of_target(y)
    if output_type in argmax_types:
        return np.argmax(y, axis=1)

    if output_type == "binary" or output_type == "continuous":
        return np.round(y)

    return y


def compute_common_classification_metrics(model,
                                          validation_data,
                                          label_names=None):
    """Computes classification metrics on the validation set.

        Args:
            model (Model): Model used to compute the predictions for the
                validation data.
            validation_data (tuple): tuple of feature data and labels.
            label_names (str, optional): Label names to be used in the
                confusion matrix/classification report.

        Returns:
            dict of classification metrics, dict of data.
    """
    x, y_true = validation_data

    start = time()

    # We need to infer the validation set, and we also need to compute a
    # a per-item inference time. So, while computing the predictions, we set the
    # batch size to 1 so we can get the proper per-item
    y_pred = model.predict(x, batch_size=1)

    # https://dawn.cs.stanford.edu/benchmark/
    one_example_latency = time() - start
    # Seconds -> Milliseconds
    one_example_latency *= 1000
    # Average latency over the input set
    one_example_latency /= len(x)
    one_example_latency = round(one_example_latency, 4)

    predicted_labels = _convert_labels(y_pred)
    actual_labels = _convert_labels(y_true)

    output_type = sklearn.utils.multiclass.type_of_target(actual_labels)

    matrix = None

    try:
        matrix = sklearn.metrics.confusion_matrix(actual_labels, predicted_labels)
        matrix = matrix.tolist()
    except:
        traceback.print_exc()

    metrics = None

    try:
        metrics = sklearn.metrics.classification_report(actual_labels,
                                        predicted_labels,
                                        output_dict=True,
                                        target_names=label_names)
    except:
        e = traceback.format_exc()
        raise ValueError("Could not get classification_report: %s" % e)


    data = {
        "actual_labels": actual_labels,
        "predicted_labels": predicted_labels,
        "predicted_probabilities": y_pred
    }

    metrics["one_example_latency_millis"] = one_example_latency
    results = {
        "target_type": output_type,
        "confusion_matrix": matrix,
        "classification_metrics": metrics,
    }

    target_type = results["target_type"]

    if target_type == "binary":
        try:
            actual_labels = data["actual_labels"]
            predictions = data["predicted_probabilities"]
            predictions = predictions[..., 0]

            fpr, tpr, thresholds = sklearn.metrics.roc_curve(actual_labels, predictions)

            results["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()
            }
            results["roc_auc_score"] = sklearn.metrics.roc_auc_score(actual_labels, predictions)
        except:
            e = traceback.format_exc()
            raise ValueError("Could not get roc_curve/roc_auc_score: %s" % e)


    return results
