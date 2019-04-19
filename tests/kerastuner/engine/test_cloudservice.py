import pytest
import os
import json
from kerastuner.engine.cloudservice import CloudService
from kerastuner.engine.cloudservice import OK, ERROR, CONNECT_ERROR, AUTH_ERROR
from kerastuner.engine.cloudservice import DISABLE


def test_is_serializable():
    cs = CloudService()
    cs.enable('test_key_true')
    config = cs.to_config()
    assert json.loads(json.dumps(config)) == config


def test_is_enable_and_status_in_sync():
    cs = CloudService()
    conf = cs.to_config()
    assert not conf['is_enable']
    assert conf['status'] == DISABLE


def test_auth_error():
    cs = CloudService()
    cs.enable("test_key_false")
    conf = cs.to_config()
    assert not conf['is_enable']
    assert conf['status'] == AUTH_ERROR


def test_is_enable_and_status_ok():
    cs = CloudService()
    cs.enable("test_key_true")
    conf = cs.to_config()
    assert conf['is_enable']
    assert conf['status'] == OK


def test_no_api_key_leak():
    cs = CloudService()
    cs.enable('test_key_true')
    config = cs.to_config()
    assert 'api_key' not in config
    assert 'test_key_true' not in config.values()
