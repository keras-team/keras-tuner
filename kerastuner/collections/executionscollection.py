import json

from .collections import Collection


class ExecutionsCollection(Collection):
    """ Manage a collection of executions
    """

    def __init__(self):
        super(ExecutionsCollection, self).__init__()

    def get_best_executions(self, objective, N=1):
        objective_name = objective.name
        reverse = objective.direction == "max"

        def objective_sort_key(idx, execution):
            execution_metrics = execution.state.agg_metrics
            metric = execution_metrics.get(objective_name).get_best_value()
            return metric

        return self.to_list(sorted_by=objective_sort_key, reverse=reverse)
