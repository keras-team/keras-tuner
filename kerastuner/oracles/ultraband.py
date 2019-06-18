import queue
import numpy as np

from ..engine import oracle as oracle_module


class UltraBand(oracle_module.Oracle):
    """ Oracle class for UltraBand.

    # Attributes:
        trails: An integer. The maximum number of trails allowed.
        queue: An instance of Queue. The elements in the queue are values dictionaries. {name: value}

    """

    def __init__(self, trials):
        super().__init__()
        self.trials = trials
        self.queue = queue.Queue()
        self.during_batch = False
        self._first_time = True
        self._perf_record = {}
        self._bracket_index = 0
        self._trials_count = 0
        self.spaces = []
        self._model_sequence = []
        self.config = UltraBandConfig(budget=self.trials)

    def result(self, trial_id, score):
        self._perf_record[trial_id] = score

    def update_space(self, additional_hps):
        pass

    def populate_space(self, trial_id, space):
        if self._trials_count >= self.trials:
            return 'EXIT', None
        self._trials_count += 1
        if self._trials_count == 1:
            return 'RUN', self._default_values(space)

        # queue not empty means it is in one bracket
        if not self.queue.empty():
            return 'RUN', self._copy_values(space, self.queue.get())

        # check if the current batch ends
        if self._bracket_index >= self.config.num_brackets:
            self._bracket_index = 0

        # check if the band ends
        if self._bracket_index == 0:
            # band ends
            self._generate_candidates(space)
        else:
            # bracket ends
            self._select_candidates()
        self._bracket_index += 1
        return 'RUN', self._copy_values(space, self.queue.get())

    @staticmethod
    def _copy_values(space, values):
        return_values = {}
        for hyperparameter in space:
            if hyperparameter.name in values:
                return_values[hyperparameter.name] = values[hyperparameter.name]
            else:
                return_values[hyperparameter.name] = hyperparameter.default
        return return_values

    def _generate_candidates(self, space):
        self.spaces = []

    def _select_candidates(self):
        for values, _ in sorted(self._perf_record.items(),
                                key=lambda key, value: value)[self._model_sequence[self._bracket_index]]:
            self.queue.put(values)

    def train(self, model, fit_args):
        fit_args['epochs'] = None
        model.fit(**fit_args)

    @classmethod
    def load(cls, filename):
        pass

    def save(self):
        pass

    def _default_values(self, space):
        pass


class UltraBandConfig(object):
    """The parameters in a UltraBand algorithm.

    # Attributes:
        budget: An integer, the number of trials of the Oracle.
    """
    def __init__(self,
                 factor=3,
                 min_epochs=3,
                 max_epochs=10,
                 budget=200):

        self.budget = budget
        self.factor = factor
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.num_brackets = self.get_num_brackets()
        self.model_sequence = self.get_model_sequence()
        self.epoch_sequence = self.get_epoch_sequence()
        self.delta_epoch_sequence = self.get_delta_epoch_sequence()

        self.total_epochs_per_band = self.get_total_epochs_per_band()

        self.num_batches = float(self.budget) / sum(self.model_sequence)
        self.partial_batch_epoch_sequence = self.get_models_per_final_band()

    def get_models_per_final_band(self):
        remaining_budget = self.budget - \
                           (int(self.num_batches) * self.total_epochs_per_band)
        fraction = float(remaining_budget) / self.total_epochs_per_band
        models_per_final_band = np.floor(
            np.array(self.model_sequence) * fraction).astype(np.int32)

        return models_per_final_band

    def get_num_brackets(self):
        """Compute the number of brackets based of the scaling factor"""
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
