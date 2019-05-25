import copy
import numpy as np


class UltraBandConfig():
    def __init__(
            self,
            factor,
            min_epochs,
            max_epochs,
            budget):

        self.budget = budget
        self.factor = factor
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.num_bands = self.get_num_bands()
        self.model_sequence = self.get_model_sequence()
        self.epoch_sequence = self.get_epoch_sequence()
        self.epochs_per_band = self.get_epochs_per_band()
        self.epochs_per_batch = self.count_epochs_per_batch()
        self.num_batches = float(self.budget) / self.epochs_per_batch
        self.partial_batch_epoch_sequence = self.get_models_per_final_band()

        print(self.__dict__)

    def get_models_per_final_band(self):
        remaining_budget = self.budget - \
            (int(self.num_batches) * self.epochs_per_batch)
        fraction = remaining_budget / self.epochs_per_batch
        models_per_final_band = np.floor(
            np.array(self.model_sequence) * fraction).astype(np.int32)

        if models_per_final_band[-1] == 0:
            return None
        return models_per_final_band

    def get_num_bands(self):
        n = 1
        v = self.min_epochs
        while v < self.max_epochs:
            v *= self.factor
            n += 1
        return n

    def get_epoch_sequence(self):
        sizes = []
        size = self.min_epochs
        for _ in range(self.num_bands - 1):
            sizes.append(int(size))
            size *= self.factor
        sizes.append(self.max_epochs)

        previous_size = 0
        output_sizes = []
        for size in sizes:
            output_sizes.append(size - previous_size)
            previous_size = size
        sizes.reverse()
        return output_sizes

    def get_model_sequence(self):
        sizes = []
        size = self.min_epochs
        for _ in range(self.num_bands - 1):
            sizes.append(int(size))
            size *= self.factor
        sizes.append(self.max_epochs)
        sizes.reverse()
        return sizes

    def get_epochs_per_band(self):

        out = []
        for e, m in zip(self.epoch_sequence, self.model_sequence):
            out.append(e * m)
        return out

    def count_epochs_per_batch(self):

        epoch_count = 0
        for i in range(len(self.epochs_per_band)):
            epoch_count += int(np.sum(self.epochs_per_band[i:]))
        return epoch_count
