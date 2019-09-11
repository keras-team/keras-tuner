import glob
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from kerastuner.tuners.randomsearch import RandomSearch

import issue_74_context


def test_val_accuracy_as_objective(tmp_path):
    (x_train, y_train), (x_test, y_test) = issue_74_context.make_testdata()

    tuner_directory = str(tmp_path / "tuner")
    trial_pattern = str(tmp_path / "tuner" / "issue_74" / "trial_*" /
                        "trial.json")

    tuner = RandomSearch(issue_74_context.build_model,
                         objective='val_accuracy',
                         max_trials=5,
                         executions_per_trial=5,
                         directory=tuner_directory,
                         project_name='issue_74')

    tuner.search_space_summary()

    tuner.search(x_train,
                 y_train,
                 epochs=2,
                 validation_data=(x_test, y_test),
                 shuffle=True)

    summary = tuner.results_summary()

    ## Manually look at the scores for all of the runs.
    # Accuracy => higher is better
    max_score = sys.float_info.min
    for file in glob.glob(trial_pattern):
        with open(file, "rt") as f:
            trial_json = json.load(f)
            score = trial_json['score']
            print(file, score)
            max_score = max(max_score, score)

    assert summary['best_objective_value'] == max_score


if __name__ == '__main__':
    test_issue_74_reproduction(pathlib.Path("/tmp/"))
