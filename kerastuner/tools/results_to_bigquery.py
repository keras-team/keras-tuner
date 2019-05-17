"Collect tuners results and output them into JSON files that can be uploaded to big query as tables"

import argparse
import copy
import importlib
import json
import os
# !Force tensorflow to use CPU only - we're not training, just compiling
# !models, and allocating GPU memory would mean we could run far fewer
# !readers.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from termcolor import cprint
from tqdm import tqdm

from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.states import InstanceState, TunerState

from google.cloud import bigquery

# See: https://cloud.google.com/bigquery/docs/reference/libraries#client-libraries-install-python  # nopep8
# for authentication information.


def parse_args():
    "Parse cmdline options"
    parser = argparse.ArgumentParser(
        description='KerasTuners results to Bigquery table files')
    parser.add_argument('--input_directory',
                        '-i',
                        type=str,
                        default='results/',
                        help='Directory containing tuner results')
    parser.add_argument('--bigquery_dataset',
                        '-b',
                        type=str,
                        default='default',
                        help='Directory containing tuner results')

    parser.add_argument('--output_directory',
                        '-o',
                        type=str,
                        default='bigquery/',
                        help='Directory where table files will be outputed')
    parser.add_argument('--project',
                        '-p',
                        type=str,
                        help='Restrict result collection to a given project')
    parser.add_argument('--custom_modules',
                        '-m',
                        type=str,
                        help='Comma separated list of dependency modules to '
                        'import, to define custom layers, etc.')
    parser.add_argument('--reader_processes',
                        '-n',
                        type=int,
                        help='Number of processes to use to read the files.')

    args = parser.parse_args()

    for module in args.custom_modules.split(","):
        module = module.strip()
        if module:
            importlib.import_module(module)

    if not os.path.exists(args.input_directory):
        cprint("[Error] Invalid Input directory %s" % args.input_directory,
               'red')
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
        k = k.replace(" ", "_").replace(':', '_').replace(".", "_")
        v = clean_value(v)
        clean[k] = v
    return clean


def results_file_to_dicts(filename):
    output = []
    with tf.io.gfile.GFile(filename, "r") as i:
        contents = json.loads(i.read())

        instance = InstanceState.from_config(contents["instance"])
        tuner = contents["tuner"]
        output_dictionary = {}

        output_dictionary["tuner.name"] = tuner["name"]
        output_dictionary["host"] = tuner["host"]["hostname"]
        output_dictionary["architecture"] = tuner["architecture"]
        output_dictionary["project"] = tuner["project"]
        output_dictionary["user_info"] = tuner["user_info"]
        output_dictionary["objective"] = tuner["objective"]
        output_dictionary["performance"] = {
            "num_parameters": instance.model_size,
            "batch_size": instance.batch_size
        }

        output_dictionary["hyper_parameters"] = []
        for name, config in instance.hyper_parameters.items():
            # TODO - figure out how to handle hparam range/type information.
            output_dictionary["hyper_parameters"].append({
                "name":
                name,
                "value":
                str(config["value"]),
                "group":
                config["group"],
                "type":
                str(type(config["value"]))
            })

        output_dictionary["idx"] = instance.idx

        # Per Execution stats
        for execution in instance.execution_states_collection.to_list():
            execution_dictionary = copy.deepcopy(output_dictionary)

            execution_dictionary["execution_idx"] = execution.idx

            execution_dictionary["metrics"] = {}
            for metric in execution.metrics.to_list():
                execution_dictionary["metrics"][
                    metric.name] = metric.get_last_value()

            if execution.classification_metrics:
                if execution.classification_metrics["one_example_latency_millis"]:
                    latency = execution.classification_metrics[
                        "one_example_latency_millis"]
                    execution_dictionary["metrics"][
                        "one_example_latency_millis"] = latency

            execution_dictionary["epochs"] = execution.epochs
            output.append(execution_dictionary)
            tf_utils.clear_tf_session()

    return output


def collect_files(input_directory):
    results_filenames = []
    for root, _, files in os.walk(input_directory):
        print(root)
        for file in files:
            if file.endswith("-results.json"):
                results_filenames.append(os.path.join(root, file))
    return results_filenames


def process_file(filename):
    ress = results_file_to_dicts(filename)
    out = []
    for res in ress:
        project = res["project"]
        res = clean_result(res)
        out.append([project, res])
    return out


def process_files(results_filenames):
    tables = defaultdict(list)

    pb = tqdm(total=len(results_filenames),
              desc="Parsing results files",
              unit='file')

    pool = Pool(64)
    for res in pool.imap_unordered(process_file, results_filenames):
        for project, results in res:
            tables[project].append(results)
        pb.update()
    return tables


def write_tables_to_disk(tables, output_directory):
    for project_name, results in tables.items():
        fname = os.path.join(output_directory, project_name + ".json")
        with tf.io.gfile.GFile(str(fname), 'w+') as out:
            for result in results:
                line = "%s\n" % json.dumps(result)
                out.write(line)
            out.close()


def write_tables_to_bigquery(tables, dataset):
    for project_name, results in tables.items():
        table_name = "%(dataset)s.%(project_name)s" % {
            "dataset": dataset,
            "project_name": project_name
        }


# def create_dataset(dataset):
#     client = bigquery.Client()
#     try {
#         dataset = client.get_dataset(@staticmethod

#     dataset = bigquery.Dataset(dataset)


def main():
    args = parse_args()
    tf.io.gfile.makedirs(args.output_directory)
    results_filename = collect_files(args.input_directory)
    tables = process_files(results_filename)
    write_tables_to_disk(tables, args.output_directory)
    write_tables_to_bigquery(tables, args.bigquery_dataset)


if __name__ == '__main__':
    main()
