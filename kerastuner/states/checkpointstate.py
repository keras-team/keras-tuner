from .state import State
from kerastuner.abstractions.display import warning, fatal, info


class CheckpointState(State):
    "Model checkpoint state abstraction"

    def __init__(self, is_enabled, monitor, mode):
        """
        Model checkpointing state

        Args:
            is_enabled (bool): use checkpointing?
            monitor (str): Metric to monitor
            mode (str): which direction to optiomize for: {min|max}
        """

        # list attributes that should be exported
        self.exportable_attributes = ['is_enabled', 'monitor', 'mode']

        self.is_enabled = is_enabled

        if not is_enabled:
            warning("models will not be saved are you sure?")
            self.monitor = None
            self.mode = None
        else:
            self.monitor = monitor
            self.mode = mode

        # errors
        if mode not in ['min', 'max']:
            fatal("checkpoint_mode must be either min or max -- typo?")

        if not isinstance(monitor, str):
            fatal("Invalid metric to monitor - expecting a string")

        # warnings
        suggestion = None
        if 'acc' in monitor and mode == 'min':
            suggestion = "change checkpoint_mode to 'max'?"
        if 'loss' in monitor and mode == 'min':
            suggestion = "change checkpoint_mode to 'min'?"
        if suggestion:
            warning("Incorrect checkpoint configuration: %s %s -- %s" % (
                    monitor, mode, suggestion))

        info("Model checkpoint enabled: monitoiring %s %s" % (mode, monitor))

    def to_dict(self):
        "return object as dictionnary"
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "is_enable": self.is_enabled
        }
