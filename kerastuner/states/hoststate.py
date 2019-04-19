from __future__ import absolute_import

from .state import State

from kerastuner.abstractions.host import Host
from kerastuner.abstractions.display import fatal
from kerastuner.abstractions.io import create_directory
from kerastuner import config


class HostState(State):
    """
    Track underlying Host state

    Args:
        result_dir (str): defaults to results/. Tuning results dir
        tmp_dir (str): deaults to tmp/. Temporary dir. Wipped at tuning start
        export_dir (str): defaults to export/. Export model dir
    """
    def __init__(self, **kwargs):
        super(HostState, self).__init__(**kwargs)

        self.result_dir = self._register('result_dir', 'results/', True)
        self.tmp_dir = self._register('tmp_dir', 'tmp/')
        self.export_dir = self._register('export_dir', 'export/', True)

        # ensure the user don't shoot himself in the foot
        if self.result_dir == self.tmp_dir:
            fatal('Result dir and tmp dir must be different')

        # create directory if needed
        create_directory(self.result_dir)
        create_directory(self.tmp_dir, remove_existing=True)
        create_directory(self.export_dir)

        # init _HOST
        config._Host = Host()

    def to_config(self):
        res = {}
        # collect user params
        for name in self.user_parameters:
            res[name] = getattr(self, name)

        # adding host hardware & software information
        res.update(config._Host.to_config())

        return res
