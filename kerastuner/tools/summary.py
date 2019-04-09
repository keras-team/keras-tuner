"Display best models in terminal"
import argparse
import json
import os
from termcolor import cprint, colored
from pathlib import Path
from collections import defaultdict
import operator
import numpy as np

from kerastuner.abstractions.display import print_table, get_progress_bar
from kerastuner.abstractions.display import section, subsection, fatal


def parse_args():
    "Parse cmdline options"
    parser = argparse.ArgumentParser(
        description='KerasTuners results to Bigquery table files')

    parser.add_argument('--input_dir', '-i', type=str,
                        default='results/', help='Directory containing \
                        tuner results')

    parser.add_argument('--project', '-p', type=str,
                        help='Restrict result collection to a given project')

    parser.add_argument('--num_models', '-n', type=int,
                        default=10, help='Num models to display')

    parser.add_argument('--metric', '-m', type=str,
                        default='loss', help='Metrics to sort by')

    parser.add_argument('--direction', '-d', type=str,
                        default='min', help='Metric direction {min/max}')

    parser.add_argument('--display_hyper_parameters', '--hyper', type=bool,
                        default=True, help='Display hyperparameters values')

    parser.add_argument('--use_colors', '-c', type=bool,
                        default=True, help='Use terminal colors.')

    parser.add_argument('--extra_fields', '-e', type=str,
                        help='list of extra fields to display. \
                        format: tuner.execution, user_info.myinfo')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        cprint("[Error] Invalid Input directory %s" % args.input_dir, 'red')
        parser.print_help()
        quit()

    return args


def parse_extra_fields(s):
    "convert extra field str into array of field to query from meta_data"
    fields = []
    if not s:
        return fields
    if ',' in s:
        s = s.split(',')
    else:
        s = [s]
    for f in s:
        if '.' in f:
            keys = f.split('.')
        else:
            keys = [f]
        fields.append([keys[-1], keys])
    return fields


def display_hp(hyper_parameters, model_indices):
    """
    Display hyper parameters values for the top models
 
    Args:
        hyper_parameters (defaultdict(list)): hyper-params values
        model_indices (list): list of model indices to display
    """

    hyper_groups = defaultdict(list)
    for hp in hyper_parameters.keys():
        grp, _ = hp.split(':')
        hyper_groups[grp].append(hp)

    # go
    headers = ['hyperparam']
    headers += ["model %s" % x for x in range(len(model_indices))]
    rows = [headers]
    for grp in sorted(hyper_groups.keys()):
        row = [grp] + [""] * len(model_indices)
        rows.append(row)
        for hp in sorted(hyper_groups[grp]):
            row = ["|-" + hp.split(":")[1]]
            for idx in model_indices:
                row.append(hyper_parameters[hp][idx])
            rows.append(row)
    print_table(rows)


def display_metrics(metrics, main_metric, direction, num_models):
    "Display results as table"
    # compute the models indices to display and their order
    indices = np.argsort(metrics[main_metric])
    if direction == 'max':
        indices = sorted(indices, reverse=True)
    indices = indices[:num_models]

    rows = []
    headers = ['metric'] + ["model %s" % x for x in range(num_models)]
    rows.append(headers)
    # main field first
    row = [main_metric]
    for idx in indices:
        row.append(metrics[main_metric][idx])
    rows.append(row)

    # other fields
    for field in sorted(list(metrics.keys())):
        if field == main_metric:
            continue
        row = [field]
        for idx in indices:
            row.append(metrics[field][idx])
        rows.append(row)

    print_table(rows)
    return indices


def summary(input_dir,
            project,
            main_metric,
            extra_fields=[],
            display_hyper_parameters=True,
            direction="min",
            num_models=10,
            use_colors=True):
    """
    Collect kerastuner results and output a summary
    """

    input_dir = Path(input_dir)
    filenames = list(input_dir.glob("*-results.json"))

    # Do an initial pass to collect all of the hyperparameters.
    pb = get_progress_bar(total=len(filenames), desc="Parsing results",
                          unit='file')
    infos = []
    hyper_parameters_list = set()
    for fname in filenames:
        info = json.loads(open(str(fname)).read())
        infos.append(info)
        
        # needed in case of conditional hyperparams
        for h in info["hyper_parameters"].keys():
            hyper_parameters_list.add(h)
        
        pb.update()
    pb.close()

    # collect data as a transpose
    hyper_parameters = defaultdict(list)
    metrics = defaultdict(list)
    for info in infos:

        # filtering if needed
        project_name = info['meta_data']['project']
        if project and project != project_name:
            continue

        if main_metric not in info['key_metrics']:
            fatal("Metric %s not in results files -- available metrics: %s" % (
                  main_metric, ", ".join(info['key_metrics'].keys())))

        if extra_fields:
            for f in extra_fields:
                ks = f[1]
                if ks[0] not in info:
                    fatal("Unknown extra field: %s - valid fields:%s" % (
                          ks[0], " ".join(info.keys())))

                v = info[ks[0]]
                for k in ks[1:]:
                    if k not in v:
                        fatal("Unknown extra field: %s.%s - valid fields:%s" % (
                          ks[0], k, " ".join(v.keys())))
                    v = v[k]
                # adding extra fields in metrics table
                metrics[f] = v

        # collect metrics
        for metric, value in info['key_metrics'].items():
            metrics[metric].append(round(value, 4))

        # collect hyper parameters
        for hp in hyper_parameters_list:
            v = info["hyper_parameters"].get(hp, None)
            if v:
                v = v["value"]
            else:
                v = ""
            hyper_parameters[hp].append(v)

    if not len(metrics):
        fatal("No models found - wrong dir (-i) or project (-p)?")

    num_models = min(len(metrics[metric]), num_models)
    section("Result summary")
    subsection("Metrics")
    mdl_indices = display_metrics(metrics, main_metric, direction, num_models)

    if display_hyper_parameters:
        subsection("Hyper Parameters")
        hp = sorted(hyper_parameters_list)[0]
        display_hp(hyper_parameters, mdl_indices)

if __name__ == '__main__':
    args = parse_args()
    extra_fields = parse_extra_fields(args.extra_fields)
    summary(args.input_dir,
            args.project,
            args.metric,
            extra_fields,
            args.display_hyper_parameters,
            args.direction,
            args.num_models,
            args.use_colors)
