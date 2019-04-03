"Backend related function"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import concurrent.futures
import time
import requests
import json
from kerastuner.abstractions.display import warning, info


def send_to_backend(url, api_key, info_type, info):
    """Sends data to the cloud service.

    Args:
        info_type (str): type of information sent
        info (dict): the data to send
    """
    response = requests.post(
        url,
        headers={'X-AUTH': api_key},
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
            warning('Invalid backend API key.')
        else:
            warning('Warning! Cloud service upload failed: %s' %
                    response.text)


class Backend():

    def __init__(self, url, api_key, notifications):
        self.base_url = url
        self.api_key = api_key
        self.notfications = notifications
        self.authorized = self._check_access()
        self.log_interval = 5  # fixme expose to the user with a min
        self.last_update = -1
        self.executor = concurrent.futures.ProcessPoolExecutor()
        if self.authorized:
            info("Go to https://.. to track your results in realtime")
        else:
            warning("Invalid cloud API key")

    def quit(self):
        """Makes sure that all cloud requests have been sent."""
        self.executor.shutdown(wait=True)
        # In case the user wants to do multiple hypertuning sessions,
        # we open another process pool.
        self.executor = concurrent.futures.ProcessPoolExecutor()

    def _check_access(self):
        "Check if backend configuration is working"
        url = self._url_join(self.base_url, 'v1/check-access')
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

    def _send_nonblocking(self, info_type, info):
        if not self.authorized:
            return

        url = self._url_join(self.base_url, 'v1/update')
        self.executor.submit(send_to_backend, url, self.api_key, info_type, info)

    def _send_blocking(self, info_type, info):
        """Send data to the cloud service

        Args:
            info_type (str): type of information sent
            info (dict): the data to send
        """
        # skip if API key don't work or service down
        if not self.authorized:
            return
        url = self._url_join(self.base_url, 'v1/update')
        send_to_backend(url, self.api_key, info_type, info )

    def send_status(self, status):
        "send tuner status for realtime tracking"

        ts = time.time()
        if ts - self.last_update > self.log_interval:
            self._send_nonblocking("status", status)
            self.last_update = ts

    def send_config(self, json_config):
      self._send_nonblocking("config", json.loads(json_config))

    def send_results(self, results):
      self._send_nonblocking("results", results)
