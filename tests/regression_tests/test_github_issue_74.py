import glob
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from kerastuner.tuners.randomsearch import RandomSearch
from kerastuner.engine.trial import Trial
import issue_74_context


def test_val_accuracy_as_objective(tmp_path):
    (x_train, y_train), (x_test, y_test) = issue_74_context.make_testdata()

    tuner_directory = str(tmp_path / "tuner")
    trial_pattern = str(tmp_path / "tuner" / "issue_74" / "trial_*" /
                        "trial.json")

    tuner = RandomSearch(issue_74_context.build_model,
                         objective='val_accuracy',
                         max_trials=5,
                         executions_per_trial=2,
                         directory=tuner_directory,
                         project_name='issue_74')

    tuner.search_space_summary()

    tuner.search(x_train,
                 y_train,
                 epochs=2,
                 validation_data=(x_test, y_test),
                 shuffle=True)

    summary_lines = tuner.results_summary()
    _, _, _, best_val_line = summary_lines

    # Manually look at the scores for all of the runs.
    # For accuracy, we want the maximum value.
    max_score = sys.float_info.min
    for file in glob.glob(trial_pattern):
        trial = Trial.load(file)
        score = trial.score
        max_score = max(max_score, score)

    expected_best_val_line = 'Best val_accuracy: %.4f' % max_score
    assert expected_best_val_line == best_val_line
