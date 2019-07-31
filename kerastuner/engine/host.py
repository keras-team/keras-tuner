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

from __future__ import absolute_import

from .state import State
from ..abstractions.host import Host
from ..abstractions.display import fatal, subsection, warning
from ..abstractions.display import display_settings, fatal
from ..abstractions.tensorflow import TENSORFLOW as tf
from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from .. import config


class Host(object):
    """Track underlying Host state

    Args:
        results_dir (str, optional): Tuning results dir. Defaults to results/.
        tmp_dir (str, optional): Temporary dir. Wiped at tuning start.
        Defaults to tmp/.
        export_dir (str, optional): Export model dir. Defaults to export/.
    """
    def __init__(self, **kwargs):
        super(HostState, self).__init__(**kwargs)

        self.results_dir = self._register('results_dir', 'results/', True)
        self.tmp_dir = self._register('tmp_dir', 'tmp/')
        self.export_dir = self._register('export_dir', 'export/', True)

        # ensure the user don't shoot himself in the foot
        if self.results_dir == self.tmp_dir:
            fatal('Result dir and tmp dir must be different')

        # create directory if needed
        tf_utils.create_directory(self.results_dir)
        tf_utils.create_directory(self.tmp_dir, remove_existing=True)
        tf_utils.create_directory(self.export_dir)

        # init _HOST
        config._Host = Host()
        status = config._Host.get_status()
        tf_version = status['software']['tensorflow']
        if tf_version:
            major, minor, rev = tf_version.split('.')
            if major == '1':
                if int(minor) >= 13:
                    print('ok')
                else:
                    fatal("Keras Tuner only work with TensorFlow version >= 1.13\
                          current version: %s - please upgrade" % tf_version)
        else:
            warning('Could not determine TensorFlow version.')

    def summary(self, extended=False):
        subsection('Directories')
        settings = {
            "results": self.results_dir,
            "tmp": self.tmp_dir,
            "export": self.export_dir
        }
        display_settings(settings)
        if extended:
            config._Host.summary(extended=extended)

    def get_config(self):
        res = {}
        # collect user params
        for name in self.user_parameters:
            res[name] = getattr(self, name)

        # adding host hardware & software information
        res.update(config._Host.to_config())

        return res
