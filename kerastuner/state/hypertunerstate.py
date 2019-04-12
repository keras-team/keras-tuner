from time import time


class HypertunerState():
    "hold tuner state"

    def __init__(self, tuner_name):
        self.tuner_name = tuner_name
        self.start_time = int(time())
