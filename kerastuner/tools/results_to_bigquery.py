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

"Collect tuners results and output them into JSON files that can be uploaded to big query as tables"
import argparse
from tqdm import tqdm
import json
import os
from termcolor import cprint
from pathlib import Path
from pprint import pprint
from collections import defaultdict
from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.states import InstanceState
from kerastuner.states import TunerState


def results_file_to_line(filename):
    
    with tf.io.gfile.GFile(filename, "r") as i:
        contents = i.read()
        
        instance = InstanceState.from_config(contents["instance"])
        tuner = TunerState.from_config(contents["tuner"])
        output_dictionary = {}
        
        output_dictionary["tuner.name"] = tuner.name
        output_dictionary["host"] = tuner.host
        output_dictionary["architecture"] = tuner.architecture
        output_dictionary["project"] = tuner.project            
        output_dictionary["user_info"] = tuner.user_info
        output_dictionary["performance.num_parameters"] = instance.model_size
        output_dictionary["performance.batch_size"] = instance.batch_size

        

        for name, config in instance.hyper_parameters.items():
            output_dictionary["hyper_parameters.%s" % name] = config["value"]
        
        # Per Execution stats
        for execution in instance.execution_states_collection.to_list():
            output_dictionary["idx"] = execution.idx
            output_dictionary["metrics"] = execution.metrics.to_config()
            output_dictionary[""]

objective = objective name
performance
    model size (#paramters)
    latency per item (inference)
    latency per batch (training)
epochs (# epochs trained)



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
        for x in value:
            cleaned_v.append(clean_value(x))
        return cleaned_v
    else:
        return value

def clean_result(results):
    clean = {}
    for k, v in results.items():
        # Empty dicts cannot be imported in bigquery
        if v == {}:
            continue
        k = k.replace(" ", "_").replace(':', '_')
        v = clean_value(v)
        clean[k] = v
    return clean


args = parse_args()

tf.io.gfile.makedirs(args.output_dir)
output_dir = Path(args.output_dir)

input_dir = Path(args.input_dir)

results_filenames = list(input_dir.glob("*-results.json"))
pb = tqdm(total=len(results_filenames), desc="Parsing results files", unit='file')
tables = defaultdict(list)
for fname in results_filenames:
    info = json.loads(open(str(fname)).read())
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
    out = open(str(fname), 'w+')
    for result in results:
        line = "%s\n" % json.dumps(result)
        out.write(line)
    out.close()
