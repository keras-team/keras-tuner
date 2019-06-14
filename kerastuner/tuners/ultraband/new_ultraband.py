import queue

from kerastuner.engine import Tuner
from kerastuner.engine.oracle import Oracle


class UltraBandTuner(Tuner):
    def run_trial(self, hp, trial_id, *fit_args, **fit_kwargs):
        model = self._build_model(hp)
        trial = self.add_trial(model, hp)
        fit_kwargs['epochs'] = hp.values['tuner/epochs']
        trial.run_execution(*fit_args, **fit_kwargs)
        result = trial.best_metrics[self.objective]
        self.oracle.result(hp.values, result)


class UltraBandOracle(Oracle):

    def __init__(self, trials):
        super().__init__()
        self.trials = trials
        self.queue = queue.Queue()
        self.during_batch = False
        self._first_time = True
        self._perf_record = {}
        self._current_round = 0
        self.spaces = []
        self._model_sequence = []

    def result(self, trial_id, score):
        self._perf_record[trial_id] = score

    def update_space(self, additional_hps):
        pass

    def populate_space(self, trial_id, space):
        # dispatch new trials with low epochs
        # occasionally re-run pretty good trials with more epochs
        # if run out of trials, IDLE
        # when only one model left, EXIT
        if self._first_time:
            self._generate_candidates(space)
            self._first_time = False

        if not self.queue.empty():
            self._copy_values(space, self.queue.get())
            return

        self._current_round += 1
        self._select_candidates()

    def _copy_values(self, space, values):
        pass

    def _generate_candidates(self, space):
        self.spaces = []

    def _select_candidates(self):
        for values, _ in sorted(self._perf_record.items(),
                                key=lambda key, value: value)[self._model_sequence[self._current_round]]:
            self.queue.put(values)

    def train(self, model, fit_args):
        fit_args['epochs'] = None
        model.fit(**fit_args)
