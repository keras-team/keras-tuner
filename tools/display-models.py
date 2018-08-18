"Display best models in terminal"
import argparse
from tqdm import tqdm
import json
import os
from termcolor import cprint
from pathlib import Path
from pprint import pprint
from collections import defaultdict
from tabulate import tabulate
import operator

def parse_args():
    "Parse cmdline options"
    parser = argparse.ArgumentParser(description='KerasTuners results to Bigquery table files')
    parser.add_argument('--input_dir', '-i', type=str, default='results/', help='Directory containing tuner results')
    parser.add_argument('--project', '-p', type=str, help='Restrict result collection to a given project')
    parser.add_argument('--num_models', '-n', type=int, default=10, help='Num models to display')
    parser.add_argument('--metric', '-m', type=str, default='val_loss', help='Metrics to sort by')
    parser.add_argument('--direction', '-d', type=str, default='desc', help='Sort model in asc or desc order')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        cprint("[Error] Invalid Input directory %s" % args.input_dir, 'red')
        parser.print_help()
        quit()

    return args

args = parse_args()
input_dir = Path(args.input_dir)
results_filenames = list(input_dir.glob("*-results.json"))
pb = tqdm(total=len(results_filenames), desc="Parsing results files", unit='file')

table = []
for fname in results_filenames:
    info = json.loads(open(fname).read())
    project_name = info['meta_data']['project']
    #filtering if needed
    if args.project and args.project != project_name:
            continue
    row = []
    for k in sorted(info['key_metrics']):
        if k == args.metric:
            v = round(info['key_metrics'][k],4) #ensure requested metric is the first one displayed
        else:
            row.append(round(info['key_metrics'][k],4))
    row = [info['meta_data']['instance'], v] + row
    table.append(row)

headers = ['idx', args.metric]
metric_idx = 0
for i, k in enumerate(sorted(info['key_metrics'])):
    if k != args.metric:
        headers.append(k)
if args.direction.lower() == "asc":
    reverse = False
else:
    reverse = True
table = sorted(table, key=operator.itemgetter(1), reverse=reverse)[:args.num_models]
print(tabulate(table, headers=headers))