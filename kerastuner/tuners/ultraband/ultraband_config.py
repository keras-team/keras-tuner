# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# # Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import numpy as np


class UltraBandConfig():
    def __init__(self, factor, min_epochs, max_epochs, budget):

        self.budget = budget
        self.factor = factor
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.num_brackets = self.get_num_brackets()
        self.model_sequence = self.get_model_sequence()
        self.epoch_sequence = self.get_epoch_sequence()
        self.delta_epoch_sequence = self.get_delta_epoch_sequence()

        self.total_epochs_per_band = self.get_total_epochs_per_band()

        self.num_batches = float(self.budget) / self.total_epochs_per_band
        self.partial_batch_epoch_sequence = self.get_models_per_final_band()

    def get_models_per_final_band(self):
        remaining_budget = self.budget - \
            (int(self.num_batches) * self.total_epochs_per_band)
        fraction = float(remaining_budget) / self.total_epochs_per_band
        models_per_final_band = np.floor(
            np.array(self.model_sequence) * fraction).astype(np.int32)

        return models_per_final_band

    def get_num_brackets(self):
        "Compute the number of brackets based of the scaling factor"
        n = 1
        v = self.min_epochs
        while v < self.max_epochs:
            v *= self.factor
            n += 1
        return n

    def get_epoch_sequence(self):
        """Compute the sequence of epochs per bracket."""
        sizes = []
        size = self.min_epochs
        for _ in range(self.num_brackets - 1):
            sizes.append(int(size))
            size *= self.factor
        sizes.append(self.max_epochs)
        return sizes

    def get_delta_epoch_sequence(self):
        """Compute the sequence of -additional- epochs per bracket.
        
        This is the number of additional epochs to train, when a model moves
        from the Nth bracket in a band to the N+1th."""
        previous_size = 0
        output_sizes = []
        for size in self.epoch_sequence:
            output_sizes.append(size - previous_size)
            previous_size = size
        return output_sizes

    def get_model_sequence(self):
        sizes = []
        size = self.min_epochs
        for _ in range(self.num_brackets - 1):
            sizes.append(int(size))
            size *= self.factor
        sizes.append(self.max_epochs)
        sizes.reverse()
        return sizes

    def get_total_epochs_per_band(self):
        epochs = 0
        for e, m in zip(self.delta_epoch_sequence, self.model_sequence):
            epochs += e * m
        return epochs
