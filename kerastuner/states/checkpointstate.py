from .state import State
from kerastuner.abstractions.display import warning, fatal, info, cprint
from kerastuner.abstractions.display import subsection, display_settings


class CheckpointState(State):
    "Model checkpoint state abstraction"

    def __init__(self, **kwargs):
        """
        Model checkpointing state

        Args:
            checkpoint_models (bool): Use checkpointing? Defaults to True.
            monitor (str): Metric to monitor. Defaults to objective.
            mode (str): Optimization direction: {min|max}.
            Defaults to objective direction.

        Attributes:
            is_enable (bool): is checkpointing enable
            monitor (str): which metric to checkpoint on?
            mode (str): which direction the metric is going to

        """
        super(CheckpointState, self).__init__(**kwargs)

        self.is_enabled = self._register('checkpoint_models', True)

        if not self.is_enabled:
            warning("models will not be saved are you sure?")
            self.monitor = None
            self.mode = None
            return
        else:
            self.monitor = self._register('checkpoint_monitor', 'loss')
            self.mode = self._register('checkpoint_mode', 'min')

        # errors
        if self.mode not in ['min', 'max']:
            fatal("checkpoint_mode must be either min or max -- typo?")

        # warnings
        suggestion = None
        if 'acc' in self.monitor and self.mode == 'min':
            suggestion = "change checkpoint_mode to 'max'?"
        if 'loss' in self.monitor and self.mode == 'max':
            suggestion = "change checkpoint_mode to 'min'?"
        if suggestion:
            warning("Incorrect checkpoint configuration: %s %s -- %s" % (
                    self.monitor, self.mode, suggestion))

        info("Model checkpoint enabled: monitoring %s %s" % (self.mode,
                                                             self.monitor))

    def summary(self, extended=False):
        subsection('Checkpoint summary')
        if not self.is_enabled:
            cprint('disabled', 'yellow')
        else:
            if extended:
                display_settings(self.to_config())
            else:
                display_settings({'monitor': self.monitor, 'mode': self.mode})

    def to_config(self):
        "return object as dictionnary"
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "is_enable": self.is_enabled
        }
