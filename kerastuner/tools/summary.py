# Copyright 2019 The Keras Tuner Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"Display best models in terminal"
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import operator

from kerastuner.collections.instancestatescollection import InstanceStatesCollection
from kerastuner.abstractions.display import display_table, progress_bar
from kerastuner.abstractions.display import section, subsection, fatal
from kerastuner.abstractions.display import colorize_row


def parse_args():
    "Parse cmdline options"
    parser = argparse.ArgumentParser(description='display tuning results')

    parser.add_argument('--input_dir', '-i', type=str, default='results/',
                        help='Directory containing tuner results')

    parser.add_argument('--project', '-p', type=str, default='default',
                        help='Which project to display result for')

    parser.add_argument('--architecture', '-a', type=str, default=None,
                        help='Restrict results to a given architecture')

    parser.add_argument('--num_models', '-n', type=int,
                        default=10, help='Num models to display')

    parser.add_argument('--metric', '-m', type=str,
                        default=None, help='Metrics to sort by - if None\
                                            use objective')

    parser.add_argument('--display_hyper_parameters', '--hyper', type=bool,
                        default=True, help='Display hyperparameters values')

    parser.add_argument('--use_colors', '-c', type=bool,
                        default=True, help='Use terminal colors.')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        fatal("[Error] Invalid Input directory %s" % args.input_dir,
              raise_exception=False)
        parser.print_help()
        quit()

    return args


def display_hparams(hyper_parameters, hyper_parameters_group, num_models,
                    use_colors):
    "Display hyper parameters values for the top models"

    rows = []
    headers = ['hyper-parameter'] + ["model %s" % x for x in range(num_models)]
    rows.append(headers)

    # hyper params
    for grp, hparams in hyper_parameters_group.items():
        row = [grp] + [''] * num_models
        if use_colors:
            row = colorize_row(row, 'yellow')
        rows.append(row)
        hparams = sorted(hparams)
        for idx, p in enumerate(hparams):
            row = ["|-" + hyper_parameters[p][0]['name']]
            row.extend([v['value'] for v in hyper_parameters[p]])
            if use_colors and idx % 2:
                row = colorize_row(row, 'cyan')
            rows.append(row)
    display_table(rows)


def display_metrics(main_metric, main_metric_values, other_metrics,
                    other_metrics_values, num_models, use_colors):

    rows = []
    headers = ['metric'] + ["model %s" % x for x in range(num_models)]
    rows.append(headers)

    # main metric first
    row = [main_metric]
    row.extend(main_metric_values)
    if use_colors:
        row = colorize_row(row, 'green')
    rows.append(row)

    # other metric
    for idx, metric in enumerate(other_metrics):
        row = [metric]
        row.extend(other_metrics_values[metric])
        if use_colors and idx % 2:
            row = colorize_row(row, 'cyan')
        rows.append(row)
    display_table(rows)


def results_summary(input_dir='results/', project='default',
                    architecture=None, sort_metric=None,
                    display_hyper_parameters=True, num_models=10,
                    use_colors=True):
    """
    Collect kerastuner results and output a summary
    """

    ic = InstanceStatesCollection()
    ic.load_from_dir(input_dir, project=project, verbose=0)
    if sort_metric:
        instance_states = ic.sort_by_metric(sort_metric)
    else:
        # by default sort by objective
        instance_states = ic.sort_by_objective()
        sort_metric = ic.get_last().objective

    other_metrics = ic.get_last().agg_metrics.get_metric_names()
    other_metrics.remove(sort_metric)  # removing the main metric

    sort_metric_values = []
    other_metrics_values = defaultdict(list)
    hyper_parameters = defaultdict(list)
    hyper_parameters_group = defaultdict(set)
    for instance in instance_states[:num_models]:
        val = instance.agg_metrics.get(sort_metric).get_best_value()
        sort_metric_values.append(round(val, 4))

        # other metrics
        for metric in other_metrics:
            val = instance.agg_metrics.get(metric).get_best_value()        
            other_metrics_values[metric].append(round(val, 4))

        # hyper-parameters
        for k, v in instance.hyper_parameters.items():
            hyper_parameters[k].append(v)
            hyper_parameters_group[v['group']].add(k)

    if not len(sort_metric_values):
        fatal("No models found - wrong dir (-i) or project (-p)?")

    num_models = min(len(sort_metric_values), num_models)
    section("Result summary")
    subsection("Metrics")
    display_metrics(sort_metric, sort_metric_values, other_metrics,
                    other_metrics_values, num_models, use_colors)

    if display_hyper_parameters:
        subsection("Hyper Parameters")
        display_hparams(hyper_parameters, hyper_parameters_group, num_models,
                        use_colors)


if __name__ == '__main__':
    args = parse_args()
    results_summary(args.input_dir, args.project, args.architecture,
                    args.metric, args.display_hyper_parameters,
                    args.num_models, args.use_colors)
