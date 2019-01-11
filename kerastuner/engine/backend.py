"Backend related function"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
import json
from os import path
from termcolor import cprint



def url_join(*parts):
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


def check_access(meta_data):
    url = url_join(meta_data['backend']['url'], 'v1/check_access')
    response = requests.post(
        url,
        headers={'X-AUTH': meta_data['backend']['api_key']})
    return response.ok


def cloud_save(local_path, ftype, meta_data):
    """Stores file remotely to backend service

    Args:
        local_path (str): where the file is saved locally.
        ftype (str): type of file saved -- results, weights, executions, config.
        meta_data (dict): tuning meta data information
    """
    # FIXME: add rate limiting (5sec by default).
    # If the user did not enable this feature, skip this code.
    if 'backend' not in meta_data:
        return
    # The backend only processes json-encoded files.
    if not ".json" in local_path:
        return
    # Uploads the file.
    url = url_join(meta_data['backend']['url'], 'v1/update')
    with open(local_path) as local_file:
      response = requests.post(
          url,
          headers={'X-AUTH': meta_data['backend']['api_key']},
          json={
            'type': ftype,
            'metadata': meta_data,
            'data': json.loads(local_file.read())
          })
    if not response.ok:
      try:
        response_json = response.json()
        print ( response_json )
      except json.decoder.JSONDecodeError:
        raise Exception('Invalid backend URL: %s. Backend replied: %s' % (
            url,
            response.text))
      if response_json['status'] == 'Unauthorized':
        raise Exception('Invalid backend API key.')
      else:
        print ('Warning! Backend upload failed. Backend replied: %s' %
            response.text)


