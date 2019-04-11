from kerastuner.abstractions.display import warning, fatal
from .named_metrics import Loss, ValLoss
from .named_metrics import Accuracy, ValAccuracy
from .named_metrics import CategoricalAccuracy, ValCategoricalAccuracy
from .named_metrics import SparseCategoricalAccuracy
from .named_metrics import ValSparseCategoricalAccuracy
from .named_metrics import BinaryAccuracy, ValBinaryAccuracy
from .named_metrics import TopKCategoricalAccuracy, ValTopKCategoricalAccuracy
from .named_metrics import SparseTopKCategoricalAccuracy
from .named_metrics import ValSparseTopKCategoricalAccuracy


class MetricCollections(object):

    def __init__(self):
        self.metrics = {}
        self.available_metrics = {
            "loss": Loss,
            "val_loss": ValLoss,
            "accuracy": Accuracy,
            "acc": Accuracy,
            "val_acc": ValAccuracy,
            "val_accuracy": ValAccuracy,
            "binary_accuracy": BinaryAccuracy,
            "val_binary_accuracy": ValBinaryAccuracy,
            "categorical_accuracy": CategoricalAccuracy,
            "val_categorical_accuracy": ValCategoricalAccuracy,
            "sparse_categorical_accuracy": SparseCategoricalAccuracy,
            "val_sparse_categorical_accuracy": ValSparseCategoricalAccuracy,
            "top_k_categorical_accuracy": TopKCategoricalAccuracy,
            "val_top_k_categorical_accuracy": ValTopKCategoricalAccuracy,
            "sparse_top_k_categorical_accuracy": SparseTopKCategoricalAccuracy,
            "val_sparse_top_k_categorical_accuracy": ValSparseTopKCategoricalAccuracy  # nopep8
        }

    def add(self, metric):
        """ Add a metric to the collection

        Args:
            metric (Metric or str): metric object or metric name
        """

        if isinstance(metric, str):
            metric_name = metric
            if metric_name not in self.available_metrics:
                fatal('Unknown metric %s - use a custom one?' % metric_name)
            metric = self.available_metrics[metric_name]()
        else:
            metric_name = metric.name

        if metric_name in self.metrics:
            fatal('Duplicate metric:%s' % metric_name)
        self.metrics[metric_name] = metric

    def get_metric(self, metric_name):
        """
        Get a metric by name

        Args:
            metric_name (str): metric name
        """
        if metric_name not in self.metrics:
            warning("Metric:%s doesn't exist" % metric_name)
            return None
        return self.metrics[metric_name]

    def get_metrics(self):
        """
        Return a collection of metrics
        """
        metric_names = sorted(self.metrics.keys())
        return [self.metrics[name] for name in metric_names]

    def update(self, metric_name, value):
        """
        Update a given metric

        Args:
            metric_name (str): name of the metric to update
            value (float or int): updated value
        """
        metric = self.get_metric(metric_name)
        if metric:
            return metric.update(value)
        return False

    def get_dict(self):
        """
        Serializable dict representation of the metric collection
        Returns:
            list: collection of metric dict
        """
        metrics = self.get_metrics()
        md = []
        for metric in metrics:
            md.append(metric.to_dict())
        return md
