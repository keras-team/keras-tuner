"Display best models in terminal"
import argparse
from tqdm import tqdm
import json
import os
from termcolor import cprint, colored
from pathlib import Path
from collections import defaultdict
import operator
from terminaltables import SingleTable
from kerastuner.engine.display import print_table

MAIN_METRIC_COLOR = 'magenta'
METRICS_COLOR = 'cyan'
HYPERPARAM_COLOR = 'green'


def maybe_colored(value, color, colorize):
  if colorize:
    return colored(value, color)
  else:
    return value

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
                        default='val_loss', help='Metrics to sort by')

    parser.add_argument('--direction', '-d', type=str,
                        default='asc', help='Sort model in asc or desc order')

    parser.add_argument('--display_hyper_parameters', '--hyper', type=bool,
                        default=True, help='Display hyperparameters values')

    parser.add_argument('--display_architecture', '-a', type=bool,
                        default=False, help='Display architecture name')

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


def summary(input_dir,
            project,
            metric,
            extra_fields,
            display_architecture,
            display_hyper_parameters,
            direction="asc",
            num_models=20,
            use_colors=True):
    """
    Collect kerastuner results and output a summary
    """

    input_dir = Path(input_dir)
    filenames = list(input_dir.glob("*-results.json"))

    rows = []
    hyper_parameters = set([])
    infos = []
    # Do an initial pass to collect all of the hyperparameters.
    pb = tqdm(total=len(filenames), desc="Parsing results", unit='files')
    for fname in filenames:
        info = json.loads(open(str(fname)).read())
        for p in info['hyper_parameters'].keys():
            hyper_parameters.add(p)
        infos.append(info)
        pb.update()
    pb.close()

    hyper_parameters = sorted(list(hyper_parameters))

    for info in infos:
        project_name = info['meta_data']['project']
        # filtering if needed
        if project and project != project_name:
            continue

        row = []

        if metric not in info['key_metrics']:
            msg = "\nMetric %s not in results files -- " % metric
            msg += "available metrics: %s" % ", ".join(info['key_metrics'].keys())
            cprint(msg, 'red')
            quit()

        if extra_fields:
            for f in extra_fields:
                ks = f[1]
                if ks[0] not in info:
                    raise ValueError("Unknown extra field: %s - valid fields:\n  %s" % (
                        ks[0], " ".join(info.keys())))

                v = info[ks[0]]
                for k in ks[1:]:
                    if k not in v:
                        raise ValueError("Unknown extra field: %s.%s - valid fields:\n  %s" % (
                            ks[0], k,  " ".join(v.keys())))
                    v = v[k]
                row.append(v)

        sort_value = None
        for k in sorted(info['key_metrics']):
            if k == metric:
                # ensure requested metric is the first one displayed
                sort_value = info['key_metrics'][k]
                main_metric = maybe_colored(round(sort_value, 4), MAIN_METRIC_COLOR, use_colors)
            else:
                row.append(maybe_colored(round(info['key_metrics'][k], 4),
                                         METRICS_COLOR, use_colors))

        if display_hyper_parameters:
            for hp in hyper_parameters:
                v = info["hyper_parameters"].get(hp, None)
                if v:
                    v = v["value"]
                else:
                    v = ""
                row.append(maybe_colored(v, HYPERPARAM_COLOR, use_colors))

        instance = info['meta_data']['instance']
        if display_architecture:
            instance = info['meta_data']['architecture'] + ":" + instance

        row = [sort_value, instance, main_metric] + row
        rows.append(row)

    if not len(rows):
        cprint("No models found - wrong dir (-i) or project (-p)?", 'red')
        quit()
    # headers
    mdl = maybe_colored(' \n', 'white', use_colors) + maybe_colored(
        'model idx', 'white', use_colors)
    mm = maybe_colored(" \n", 'white', use_colors) + maybe_colored(
        metric, MAIN_METRIC_COLOR, use_colors)
    headers = [mdl, mm]

    if extra_fields:
        for f in extra_fields:
            headers.append(f[0])

    # metrics
    for i, k in enumerate(sorted(info['key_metrics'])):
        if k != metric:
            headers.append(
                maybe_colored("\n", METRICS_COLOR, use_colors) +
                maybe_colored(k, METRICS_COLOR, use_colors))

    # hyper_parameters
    if display_hyper_parameters:
        for hp in hyper_parameters:
            # only show group if meaningful

            group, name = hp.split(":", 2)

            if group == 'default':
                group = ''
            g = maybe_colored(group + '\n', HYPERPARAM_COLOR, use_colors)
            s = g + maybe_colored(name, HYPERPARAM_COLOR, use_colors)
            headers.append(s)

    if direction.lower() == "asc":
        reverse = False
    else:
        reverse = True


    rows = sorted(rows, key=lambda x: float(x[0]),
                reverse=reverse)

    # Drop the sort metric value that we temporarily appended
    # to make sorting possible.
    rows = [row[1:] for row in rows]
    if num_models > 0:
        rows = rows[:min(num_models, len(rows))]

    table = []
    table.append(headers)
    for r in rows:
        table.append(r)

    print_table(table)


if __name__ == '__main__':
    args = parse_args()
    extra_fields = parse_extra_fields(args.extra_fields)

    summary(args.input_dir,
            args.project,
            args.metric,
            extra_fields,
            args.display_architecture,
            args.display_hyper_parameters,
            args.direction,
            args.num_models,
            args.use_colors)
