from __future__ import absolute_import

from time import time

from kerastuner.abstractions.display import fatal, subsection, display_settings

from .state import State
from .checkpointstate import CheckpointState


class TunerStatsState(State):
    "Track hypertuner statistics"

    def __init__(self):

        self.generated_instances = 0  # overall number of instances generated
        self.invalid_instances = 0  # how many models didn't work
        self.instances_previously_trained = 0  # num instance already trained
        self.collisions = 0  # how many time we regenerated the same model
        self.over_sized_models = 0  # num models with params> max_params

    def summary(self, extended=False):
        "display statistics summary"
        subsection("Tuning stats")
        display_settings(self.to_config())

    def to_config(self):
        return {
            'num_generated_models': self.generated_instances,
            'num_invalid_models': self.invalid_instances,
            "num_mdl_previously_trained": self.instances_previously_trained,
            "num_collision": self.collisions,
            "num_over_sized_models": self.over_sized_models
        }
