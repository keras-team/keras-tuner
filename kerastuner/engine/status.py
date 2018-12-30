"Keep track of the tuner current status"

import time


class Status():

    def __init__(self):

        self.start_time = int(time.time())

    def get_status(self):
        """Return current status
        
        Returns:
            dict: dictionnary containing tuner status
        """

        status = {
            "start_time": self.start_time
        }

        return status