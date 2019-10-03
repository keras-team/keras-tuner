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

import threading
import os
import portpicker
from unittest import mock


def mock_distribute(fn, num_workers=2):
    """Runs `fn` in multiple processes, setting appropriate env vars."""
    port = str(portpicker.pick_unused_port())

    @mock.patch.dict(os.environ, os.environ.copy())
    def chief_fn():
        # The port for the chief and workers to communicate on.
        os.environ['KERASTUNER_PORT'] = port
        # Run in distributed mode when present. Cloud oracle does not
        # run in this mode because the Cloud API coordinates workers.
        os.environ['KERASTUNER_DISTRIBUTED'] = 'True'
        # The ID of this process. 'chief' should run a server.
        os.environ['KERASTUNER_TUNER_ID'] = 'chief'
        fn()
    chief_process = threading.Thread(target=chief_fn)
    chief_process.daemon = True
    chief_process.start()

    worker_processes = []
    for i in range(num_workers):

        @mock.patch.dict(os.environ, os.environ.copy())
        def worker_fn():
            os.environ['KERASTUNER_PORT'] = port
            os.environ['KERASTUNER_DISTRIBUTED'] = 'True'
            # Workers that are part of the same multi-worker
            # DistributionStrategy should have the smae TUNER_ID.
            os.environ['KERASTUNER_TUNER_ID'] = 'worker{}'.format(i)
            fn()
        worker_process = threading.Thread(target=worker_fn)
        worker_process.start()
        worker_processes.append(worker_process)

    for worker_process in worker_processes:
        worker_process.join()
