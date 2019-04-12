from kerastuner.abstractions import warning, fatal, info


class CheckpointState(object):
    "Model checkpoint state abstraction"

    def __init__(self, is_enabled, monitor, mode):

        self.is_enabled = is_enabled
        self.monitor = monitor
        self.mode = mode

        if not is_enabled:
            warning("models will not be saved are you sure?")
            return

        suggestion = None
        if 'acc' in monitor and mode == 'min':
            suggestion = "change checkpoint_mode to 'max'?"
        if 'loss' in monitor and mode == 'min':
            suggestion = "change checkpoint_mode to 'min'?"
        if suggestion:
            warning("Incorrect checkpoint configuration: %s %s -- %s" % (
                    monitor, mode, suggestion))

        if mode not in ['min', 'max']:
            fatal("checkpoint_mode must be either min or max -- typo?")

        info("Model checkpoint enabled: monitoiring %s %s" % (mode, monitor))
