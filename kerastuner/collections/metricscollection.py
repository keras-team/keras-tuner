from .collections import Collection
from kerastuner.engine.metric import Metric
from kerastuner.abstractions.display import warning, fatal, display_table

_METRIC_DIRECTION = {
    'accuracy': 'max',
    'binary_accuracy': 'max',
    'categorical_accuracy': 'max',
    'categorical_crossentropy': 'min',
    'loss': 'min',
    'sparse_categorical_accuracy': 'max',
    'sparse_top_k_categorical_accuracy': 'max',
    'test': 'min',
    'top_k_categorical_accuracy': 'max',
}

_METRIC_ALIAS = {
    "acc": "accuracy"
}


class MetricsCollection(Collection):

    def __init__(self):
        super(MetricsCollection, self).__init__()
        self._objective_name = None  # track which metric is the objective

    def add(self, metric):
        """ Add a metric to the collection

        Args:
            metric (Metric or str): Metric object or metric name
        """
        # our own metric object -> direct add
        if isinstance(metric, Metric):
            # our own metric, do nothing
            metric_name = metric.name
        else:
            if isinstance(metric, str):
                # metric by name
                metric_name = metric
            else:
                # keras metric
                metric_name = metric.name

            metric_name = self._replace_alias(metric_name)
            # canonalize metric name (val_metric vs metric)
            no_val_name = metric_name.replace('val_', '')
            if no_val_name in _METRIC_DIRECTION:
                direction = _METRIC_DIRECTION[no_val_name]
            else:
                fatal('Unknown metric %s' % metric_name)

            # create a metric object
            metric = Metric(metric_name, direction)

        if metric_name in self._objects:
            fatal('Duplicate metric:%s' % metric_name)
        self._objects[metric_name] = metric
        self._last_insert_idx = metric_name

    def update(self, metric_name, value):
        """
        Update a given metric

        Args:
            metric_name (str): Name of the metric to update.
            value (float or int): Updated value.

        Returns:
            bool: True if the metric improved, False otherwise.
        """
        metric_name = self._replace_alias(metric_name)
        metric = self.get(metric_name)
        if metric:
            return metric.update(value)
        return False

    def get(self, metric_name):
        metric_name = self._replace_alias(metric_name)
        if metric_name in self._objects:
            return self._objects[metric_name]
        return None

    def get_metric_names(self):
        "Return the list of metric names"
        return sorted(self._objects.keys())

    def _replace_alias(self, metric_name):
        "Replace metric alias with their canonical name"

        no_val_name = metric_name.replace('val_', '')
        # alias?
        if no_val_name in _METRIC_ALIAS:
            no_val_replace_name = _METRIC_ALIAS[no_val_name]
            metric_name = metric_name.replace(no_val_name, no_val_replace_name)
        return metric_name

    def to_config(self):
        """Serializable list of metrics.

        Returns:
            list: Collection of metric dict
        """

        names = sorted(self._objects.keys())
        # for each metric returns its serialized form

        out = []
        for name in names:
            obj = self._objects[name]

            cfg = None
            if hasattr(obj, 'to_config'):
                cfg = obj.to_config()
            else:
                cfg = obj.get_config()
            out.append(cfg)
        return out

    @staticmethod
    def from_config(config, with_values=True):
        """Generate a MetricsCollection from a configuration dictionary

        Args:
            config - (dict) The configuration dict returned from to_config.
            with_values - (bool) If True, metric values are copied. If False,
                only the metadata is copied. Defaults to True.

        Returns:
            (MetricsCollection) The collection of metrics defined by the config.
        """
        col = MetricsCollection()
        for metric_config in config:
            metric = Metric.from_config(metric_config, with_values=with_values)

            col.add(metric)
            if metric.is_objective:
                col._objective_name = metric.name
        return col

    def set_objective(self, name):
        "Mark a metric as the tuning objective"
        name = self._replace_alias(name)
        print(name)
        if name not in self._objects:
            found = False

            # accuracy special case which seems to be only a
            # Windows -TF 1.13+ issue
            if 'accuracy' in name:
                # is it a validation metric
                val = False
                if 'val' in name:
                    val = True

                for mm in self.get_metric_names():
                    # none val_* accuracy
                    if not val and 'accuracy' in mm and 'val' not in mm:
                        name = mm
                        found = True

                    # val_* accuracy
                    if val and 'accuracy' in mm and 'val' in mm:
                        name = mm
                        found = True

            if not found:
                metrics = ", ".join(list(self._objects.keys()))
                fatal("can't find objective: %s in metric list:%s" % (name,
                                                                      metrics))
        if self._objective_name:
            fatal("Objective already set to %s" % self._objective_name)
        self._objective_name = name
        self._objects[name].is_objective = True
        return name

    def get_objective(self):
        "Get metric objective"
        if not self._objective_name:
            warning("objective not set yet. returning None")
            return None
        return self._objects[self._objective_name]

    def summary(self, extended=False):
        """Display a table containing the name and best/last value for
        each metric."""
        rows = [['name', 'best', 'last']]
        for m in self.to_list():
            row = [
                m.name,
                m.get_best_value(),
                m.get_last_value(),
            ]
            rows.append(row)
        display_table(rows)
