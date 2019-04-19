from __future__ import absolute_import

from time import time

from kerastuner.abstractions.display import fatal

from .state import State
from .checkpointstate import CheckpointState


class TunerStatsState(State):
    "Track hypertuner statistics"

    def __init__(self):

        self.num_generated_models = 0  # overall number of model generated
        self.num_invalid_models = 0  # how many models didn't work
        self.num_mdl_previously_trained = 0  # how many models already trained
        self.num_collisions = 0  # how many time we regenerated the same model
        self.num_over_sized_models = 0  # num models with params> max_params

    def to_config(self):
        return {
            'num_generated_models': self.num_generated_models,
            'num_invalid_models': self.num_invalid_models,
            "num_mdl_previously_trained": self.num_mdl_previously_trained,
            "num_collision": self.num_collisions,
            "num_over_sized_models": self.num_over_sized_models
        }
