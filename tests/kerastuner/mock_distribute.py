# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mock running KerasTuner in a distributed tuning setting."""

import mock
import os
import portpicker
import six
import sys
import threading


class ExceptionStoringThread(threading.Thread):
    def run(self):
        self.raised_exception = None
        try:
            super(ExceptionStoringThread, self).run()
        except BaseException:
            self.raised_exception = sys.exc_info()


class MockEnvVars(dict):
    """Allows setting different environment variables in threads."""
    def __init__(self):
        self.thread_local = threading.local()
        self.initial_env_vars = os.environ.copy()

    def _setup_thread(self):
        if getattr(self.thread_local, 'environ', None) is None:
            self.thread_local.environ = self.initial_env_vars.copy()

    def get(self, name, default=None):
        self._setup_thread()
        return self.thread_local.environ.get(name, default)

    def __setitem__(self, name, value):
        self._setup_thread()
        self.thread_local.environ[name] = value

    def __getitem__(self, name):
        self._setup_thread()
        return self.thread_local.environ[name]

    def __contains__(self, name):
        self._setup_thread()
        return name in self.thread_local.environ


def mock_distribute(fn, num_workers=2):
    """Runs `fn` in multiple processes, setting appropriate env vars."""
    port = str(portpicker.pick_unused_port())
    with mock.patch.object(os, 'environ', MockEnvVars()):

        def chief_fn():
            # The IP address of the chief Oracle. Run in distributed mode when
            # present. Cloud oracle does not run in this mode because the Cloud
            # API coordinates workers itself.
            os.environ['KERASTUNER_ORACLE_IP'] = '127.0.0.1'
            # The port of the chief Oracle.
            os.environ['KERASTUNER_ORACLE_PORT'] = port
            # The ID of this process. 'chief' will run the OracleServicer server.
            os.environ['KERASTUNER_TUNER_ID'] = 'chief'
            fn()
        chief_thread = ExceptionStoringThread(target=chief_fn)
        chief_thread.daemon = True
        chief_thread.start()

        worker_threads = []
        for i in range(num_workers):

            def worker_fn():
                os.environ['KERASTUNER_ORACLE_IP'] = '127.0.0.1'
                os.environ['KERASTUNER_ORACLE_PORT'] = port
                # Workers that are part of the same multi-worker
                # DistributionStrategy should have the same TUNER_ID.
                os.environ['KERASTUNER_TUNER_ID'] = 'worker{}'.format(i)
                fn()
            worker_thread = ExceptionStoringThread(target=worker_fn)
            worker_thread.start()
            worker_threads.append(worker_thread)

        for worker_thread in worker_threads:
            worker_thread.join()

        if chief_thread.raised_exception:
            six.reraise(*chief_thread.raised_exception)
        for worker_thread in worker_threads:
            if worker_thread.raised_exception is not None:
                six.reraise(*worker_thread.raised_exception)
