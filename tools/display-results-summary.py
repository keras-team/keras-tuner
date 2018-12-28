"Display best models in terminal"
import argparse
from tqdm import tqdm
import json
import os
from termcolor import cprint, colored
from pathlib import Path
from collections import defaultdict
#from tabulate import tabulate
import operator
from terminaltables import SingleTable

MAIN_METRIC_COLOR = 'magenta'
METRICS_COLOR = 'cyan'
HYPERPARAM_COLOR = 'green' 

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

    parser.add_argument('--hyper_parameters', '--hyper', type=bool,
                        default=True, help='Display hyperparameters values')

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
    print(fields)
    return fields


args = parse_args()
extra_fields = parse_extra_fields(args.extra_fields)

input_dir = Path(args.input_dir)
results_filenames = list(input_dir.glob("*-results.json"))
pb = tqdm(total=len(results_filenames),
          desc="Parsing results files", unit='file')

hyper_parameters = defaultdict(int)

rows = []
for fname in results_filenames:
    info = json.loads(open(str(fname)).read())
    project_name = info['meta_data']['project']
    # filtering if needed
    if args.project and args.project != project_name:
        continue

    row = []

    if args.metric not in info['key_metrics']:
        msg = "\nMetric %s not in results files -- " % args.metric
        msg += "available metrics: %s" % ", ".join(info['key_metrics'].keys())
        cprint(msg, 'red')
        quit()

    if extra_fields:
        for f in extra_fields:
            ks = f[1]
            v = info[ks[0]]
            for k in ks[1:]:
                v = v[k]
            row.append(v)

    for k in sorted(info['key_metrics']):
        if k == args.metric:
            # ensure requested metric is the first one displayed
            v = colored(round(info['key_metrics'][k], 4), MAIN_METRIC_COLOR)
        else:
            row.append(colored(round(info['key_metrics'][k], 4), 
                               METRICS_COLOR))

    if args.hyper_parameters:
        for hp in sorted(info['hyper_parameters'].keys()):
            row.append(colored(info['hyper_parameters'][hp]['value'],
                               HYPERPARAM_COLOR))

    row = [info['meta_data']['instance'], v] + row
    rows.append(row)

# headers
mdl = colored(' \n', 'white') + colored('model idx', 'white')
mm = colored(" \n", 'white') + colored(args.metric, MAIN_METRIC_COLOR)
headers = [mdl, mm]

if extra_fields:
    for f in extra_fields:
        headers.append(f[0])

# metrics
for i, k in enumerate(sorted(info['key_metrics'])):
    if k != args.metric:
        headers.append(colored("\n", METRICS_COLOR) + colored(k, METRICS_COLOR))

# hyper_parameters
if args.hyper_parameters:
    for hp in sorted(info['hyper_parameters'].keys()):
        
        # only show group if meaningful
        group = info['hyper_parameters'][hp]['group']
        if group == 'default':
            group = ''
        g = colored(group + '\n', HYPERPARAM_COLOR)
        s = g + colored(info['hyper_parameters'][hp]['name'], HYPERPARAM_COLOR)
        headers.append(s)

if args.direction.lower() == "asc":
    reverse = False
else:
    reverse = True


rows = sorted(rows, key=operator.itemgetter(1),
              reverse=reverse)[:args.num_models]

table = []
table.append(headers)
for r in rows:
    table.append(r)
table_instance = SingleTable(table, '')

print("")
print(table_instance.table)
