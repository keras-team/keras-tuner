from kerastuner.abstractions.io import glob, read_file
from collections import namedtuple
from copy import deepcopy
import os
import json



class ResultSet(object):
    """ A ResultSet is a light wrapper around a list of dictionaries
    representing the results of multiple models, which simplifies filtering,
    sorting, etc."""

    def __init__(self, results):
        """ Create a result set from a list of dictionaries. 

        Args:
            results (list of dict): The raw results.
        """
        self.results = results

    def hyper_parameters(self):
        """ Returns a sorted list of all hyper parameter names used across all
        executions."""

        hp_set = set()
        for result in self.results:
            for hp in result["hyper_parameters"].keys():
                hp_set.add(hp)
        return sorted(list(hp_set))

    def metrics(self):
        """ Returns a sorted list of all metric names used across all executions."""

        metrics = set() 
        for result in self.results:
            for hp in result["statistics"]["latest"].keys():
                metrics.add(hp)
        return sorted(list(metrics))

    def sorted_by_metric(self, metric, order="min", best=False):
        """ Returns a sorted version of the result set. 

        Args:
            metric (string): The metric name used for sorting.         
            order (str, optional): Defaults to "min". If "min", the smallest
                results are returned first. Otherwise, the largest results
                come first.
            best (bool, optional): Defaults to False. If true, the "best"
                value (across epochs) for each model will be used, instead of
                the latest value.
        Returns:
            A ResultSet containing the sorted results.
        """

        key = "latest"
        if best:
            key = "best"

        reverse = False
        if order == "max":
            reverse = True

        sort_fn = (lambda x: x["statistics"][key][metric])
        return ResultSet(sorted(self.results, key=sort_fn, reverse=reverse))

    def filter_by_fn(self, fn):
        """ Returns a filtered version of the result set.

        Args:
            fn (function): A function, which is expected to take a single
                 argument (the result) and returns True iff the result should
                 be retained.

        Returns:
            A ResultSet containing the results for which the filter function
                returns True.
        """
        out = []
        for res in self.results:
            if fn(res):
                out.append(res)
        return ResultSet(out)

    def limit(self, N):
        """ Limit the resultset to the first N results.

        Args:
            N (int): The maximum number of results to keep.

        Returns:
            A ResultSet containing the filtered results.
        """

        return ResultSet(deepcopy(self.results[0:min(N, len(self.results))]))


def read_results(results_dir):
    """ Read the results from the specified directory.

    Args:
        results_dir (string): Path representing the results directory.

    Returns:
        ResultSet: The set of results, in a sortable/filterable form.
    """

    # Find all of the results files. There should be one per Instance, each of
    # which can have information about multiple executions.
    files = glob(str(os.path.join(results_dir, "*-results.json")))

    results = []
    for filename in files:
        instance = json.loads(read_file(filename))

        # Extract Instance-level metadata.
        metadata = instance["meta_data"]
        result = {
            "instance": metadata["instance"],
            "architecture": metadata["architecture"],
            "project": metadata["project"],
            "hyper_parameters": instance["hyper_parameters"]
        }

        # Generate the prefix used for this instance, e.g.:
        #  "project-architecture-instance"
        instance_prefix = "%s-%s-%s" % (
            result["project"], result["architecture"], result["instance"])
        result["instance_prefix"] = instance_prefix
        result["results_file"] = instance_prefix + "-results.json"

        # Each Instance can have multiple InstanceExecutions, which will be
        # reported separately.
        for execution in instance["executions"]:

            # Clone the instance level data for the specific execution.
            execution_result = deepcopy(result)

            # Pull execution information (including metrics) from the execution.
            execution_meta = execution["meta_data"]
            execution_result["execution"] = execution_meta["execution"]
            execution_result["statistics"] = execution_meta["statistics"]

            # Determine the execution prefix, and store the relevant filenames
            # for the config/weights.
            execution_prefix = "%s-%s" % (
                instance_prefix, execution_result["execution"])
            execution_result["execution_prefix"] = execution_prefix
            execution_result["config_file"] = execution_prefix + "-config.json"
            execution_result["weights_file"] = execution_prefix + "-weights.h5"

            results.append(execution_result)

    return ResultSet(results)
