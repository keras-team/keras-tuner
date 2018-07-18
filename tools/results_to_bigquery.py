"Collect tuners results and output them into JSON files that can be uploaded to big query as tables"
import argparse
from tqdm import tqdm
import json
import os
from termcolor import cprint
from pathlib import Path
from pprint import pprint
from collections import defaultdict

#which fields (and their sub_fields) to collect out of the result file
fields_to_collect = ["ts", "meta_data", "key_metrics", "metrics", "training_size", "validation_size", "num_executions",
                    "batch_size", "model_size", "hyper_parameters"]

def parse_args():
    "Parse cmdline options"
    parser = argparse.ArgumentParser(description='KerasTuners results to Bigquery table files')
    parser.add_argument('--input_dir', '-i', type=str, default='results/', help='Directory containing tuner results')
    parser.add_argument('--output_dir', '-o', type=str, default='bigquery/', help='Directory where table files will be outputed')
    parser.add_argument('--project', '-p', type=str, help='Restrict result collection to a given project')
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        cprint("[Error] Invalid Input directory %s" % args.input_dir, 'red')
        parser.print_help()
        quit()

    return args

def clean_value(value):
    if isinstance(value, str):
        return value.replace(" ", "_")
    if isinstance(value, dict):
        return clean_result(value)
    if isinstance(value, list):
        cleaned_v = []
        for x in v:
            cleaned_v.append(clean_value(v))
        return cleaned_v

def clean_result(results):
    clean = {}
    for k, v in results.items():
        k = k.replace(" ", "_")
        v = clean_value(v)
        clean[k] = v
    return clean


args = parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
output_dir = Path(args.output_dir)

input_dir = Path(args.input_dir)

results_filenames = list(input_dir.glob("*-results.json"))
pb = tqdm(total=len(results_filenames), desc="Parsing results files", unit='file')
tables = defaultdict(list)
for fname in results_filenames:
    info = json.loads(open(fname).read())
    project_name = info['meta_data']['project']
    #filtering if needed
    if args.project and args.project != project_name:
        continue
    project_name = project_name.replace(" ", "_").lower()
    
    #collecting necessary fields
    result = {}
    for field in fields_to_collect:
        if field in info:
            result[field] = info[field]

    #normalize the stuff
    result = clean_result(result)
    tables[project_name].append(result)
    pb.update()

for project_name, results in tables.items():
    fname = output_dir / (project_name + ".json")
    out = open(fname, 'w+')
    for result in results:
        line = "%s\n" % json.dumps(result)
        out.write(line)
    out.close()