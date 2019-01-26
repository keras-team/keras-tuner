"Backend related function"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import requests
import json
from .display import warning, info


class Backend():

    def __init__(self, url, api_key, notifications):
        self.base_url = url
        self.api_key = api_key
        self.notfications = notifications
        self.authorized = self._check_access()
        self.log_interval = 5  # fixme expose to the user with a min
        self.last_update = -1

        if self.authorized:
            info("Go to https://.. to track your results in realtime")
        else:
            warning("Invalid cloud API key")

    def _check_access(self):
        "Check if backend configuration is working"
        url = self._url_join(self.base_url, 'v1/check_access')
        response = requests.post(
            url,
            headers={'X-AUTH': self.api_key})
        return response.ok

    def _url_join(self, *parts):
        """Joins a base url with one or more path segments.

        This joins https://example.com/a/b/ with 'update', resulting
        in https://example.com/a/b/update. Removing the trailing slash from
        the first argument will yield the same output.

        Args:
            parts (list): the URL parts to join.

        Returns:
            str: A url.
        """
        return "/".join(map(lambda fragment: fragment.rstrip('/'), parts))

    def send_status(self, status):
        "send tuner status for realtime tracking"

        ts = time.time()
        if ts - self.last_update > self.log_interval:
            self._send("status", status)
            self.last_update = ts

    def _send(self, info_type, info):
        """Send data to the cloud service

        Args:
            info_type (str): type of information sent
            info (dict): the data to send
        """

        # skip if API key don't work or service down
        if not self.authorized:
            return

        url = self._url_join(self.base_url, 'v1/update')
        response = requests.post(
            url,
            headers={'X-AUTH': self.api_key},
            json={
                'type': info_type,
                'data': info
            })

        if not response.ok:
            try:
                response_json = response.json()
            except json.decoder.JSONDecodeError:
                warning("Cloud service down -- data not uploaded")

            if response_json['status'] == 'Unauthorized':
                self.authorized = False
                warning('Invalid backend API key.')
            else:
                warning('Warning! Cloud service upload failed: %s' %
                        response.text)


def cloud_save(local_path, ftype, meta_data):
    """Stores file remotely to backend service

    Args:
        local_path (str): where the file is saved locally
        ftype (str): type of file saved: results, weights, executions, config
        meta_data (dict): tuning meta data information
    """

    return
