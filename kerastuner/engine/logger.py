"Cloud service related functionality."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import concurrent.futures
import requests
import json


OK = 0
ERROR = 1
CONNECT_ERROR = 2
AUTH_ERROR = 3
UPLOAD_ERROR = 4


class Logger(object):

    def register_tuner(self, tuner_state):
        """Informs the logger that a new search is starting."""
        raise NotImplementedError

    def register_trial(self, trial_id, trial_state):
        """Informs the logger that a new Trial is starting."""
        raise NotImplementedError

    def report_trial_state(self, trial_id, trial_state):
        """Gives the logger information about trial status."""
        raise NotImplementedError

    def exit(self):
        raise NotImplementedError


def url_join(*parts):
    return '/'.join(map(lambda fragment: fragment.rstrip('/'), parts))


def send_to_backend(url, data, key):
    response = requests.post(
        url,
        headers={'X-AUTH': key},
        json=data)

    if not response.ok:
        try:
            response_json = response.json()
        except json.decoder.JSONDecodeError:
            print('Cloud service down -- data not uploaded: %s' % response.text)
            return CONNECT_ERROR

        if response_json['status'] == 'Unauthorized':
            print('Invalid backend API key.')
            return AUTH_ERROR
        else:
            print('Warning! Cloud service upload failed: %s' % response.text)
            return UPLOAD_ERROR
        return ERROR
    else:
        return OK


class CloudLogger(Logger):

    def __init__(self, api_key):
        self.api_key = api_key
        self.log_interval = 5

        self._base_url = (
            'https://us-central1-kerastuner-prod.cloudfunctions.net/api/')
        self._last_update = -1
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._search_id = None
        self._async = True

    def register_tuner(self, tuner_state):
        data = {
            'tuner_state': tuner_state,
        }
        self._send_to_backend('register_tuner', data)

    def register_trial(self, trial_id, trial_state):
        """Informs the logger that a new Trial is starting."""
        data = {
            'trial_id': trial_id,
            'trial_state': trial_state,
        }
        self._send_to_backend('register_trial', data)

    def report_trial_state(self, trial_id, trial_state):
        """Gives the logger information about trial status."""
        data = {
            'trial_id': trial_id,
            'trial_state': trial_state,
        }
        self._send_to_backend('report_trial_state', data)

    def exit(self):
        """Makes sure that all cloud requests have been sent."""
        self._executor.shutdown(wait=True)
        # In case the user wants to do multiple hypertuning sessions,
        # we open another process pool.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _send_to_backend(self, route, data):
        url = url_join(self._base_url, route)
        if self._async:
            self._executor.submit(send_to_backend,
                                  url,
                                  data,
                                  self.api_key)
        else:
            send_to_backend(url, data, self.api_key)
