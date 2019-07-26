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

"Compute hardware statistics"

import os
from time import time
import psutil
import platform
from subprocess import Popen, PIPE
from distutils import spawn
from time import time
import tensorflow as tf
import kerastuner as kt
from .display import subsection, display_settings, display_setting
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils


class Host():
    """Provide an abstract view of host data

    Attributes:
        cache_ttl (int): how long to cache host status

    """

    def __init__(self, results_dir, tmp_dir, export_dir):

        # caching
        self.cached_status = None
        self.cache_ts = None
        self.cache_ttl = 3

        # compute static information once
        self.gpu_driver_version = 'N/A'
        self.nvidia_smi = self._find_nvidia_smi()
        self.cpu_core_count = psutil.cpu_count()
        self.cpu_frequency = psutil.cpu_freq()
        self.partitions = psutil.disk_partitions()
        self.hostname = platform.node()
        self.cpu_name = platform.processor()

        # additional GPU info
        if hasattr(tf, 'test'):
            self.tf_can_use_gpu = tf.test.is_gpu_available()  # before get_software
        else:
            # Support tensorflow versions where tf.test is unavailable.
            self.tf_can_use_gpu = False
        self._get_gpu_usage()  # to get gpu driver info > before get software

        self.software = self._get_software()

        self.results_dir = results_dir or 'results'
        self.tmp_dir = tmp_dir or 'tmp_dir'
        self.export_dir = export_dir or 'export'

        # ensure the user don't shoot himself in the foot
        if self.results_dir == self.tmp_dir:
            fatal('Result dir and tmp dir must be different')

        # create directory if needed
        tf_utils.create_directory(self.results_dir)
        tf_utils.create_directory(self.tmp_dir, remove_existing=True)
        tf_utils.create_directory(self.export_dir)

    def get_status(self, no_cach=False):
        """
        Return various hardware counters

        Args:
            no_cache (Bool): Defaults to True. Disable status cache

        Returns:
            dict: hardware counters
        """

        if not self.cached_status or time() - self.cache_ts > self.cache_ttl:
            status = {}
            status['cpu'] = self._get_cpu_usage()
            status['ram'] = self._get_memory_usage()
            status['gpu'] = self._get_gpu_usage()
            status['uptime'] = self._get_uptime()
            status['disk'] = self._get_disk_usage()
            status['software'] = self.software
            status['hostname'] = self.hostname
            status["available_gpu"] = len(status['gpu'])
            self.cached_status = status
            self.cache_ts = time()
        return self.cached_status

    def summary(self, extended=False):
        """Display a summary of host state

        Args:
            extended (bool, optional): Display an extensive summary.
            Defaults to False.
        """
        status = self.get_status()
        subsection("Host info")
        if not extended:
            summary = {
                "num gpu": len(status['gpu']),
                "num cpu cores": status['cpu']['core_count'],
                "Keras Tuner version": status['software']['kerastuner'],
                "Tensorflow version": status['software']['tensorflow']
            }
            display_settings(summary)
        else:
            summary = {
                "cache ttl": self.cache_ttl,
                "hostname": self.hostname,
                "uptime": status['uptime']
            }
            display_settings(summary)

            # software
            display_setting('software', idx=1)
            display_settings(status['software'], indent_level=2)

            ram = status['ram']
            s = "ram: %s/%s%s" % (ram['used'], ram['total'], ram['unit'])
            display_setting(s, idx=2)

            # disk
            display_setting('disks', idx=3)
            for idx, disk in enumerate(status['disk']):
                s = "%s %s/%s %s" % (disk['name'], disk['used'],
                                     disk['total'], disk['unit'])
                display_setting(s, idx=idx, indent_level=2)

            # cpu
            display_setting('cpu')
            display_settings(status['cpu'], indent_level=2)

            # gpu
            if len(status['gpu']) > 1:
                display_setting('gpus')
                indent = 3
            else:
                indent = 2
            for gpu in status['gpu']:
                display_setting('gpu', indent_level=indent - 1)
                display_settings(gpu, indent_level=indent)

    def get_config(self):
        """
        Return various hardware counters as dict

        implemented to have a consistent interface with State related classes

        Returns:
            dict: hardware counters
        """
        return self.get_status()

    def _get_hostname(self):
        """get system name"""
        return platform.uname()

    def _get_cpu_usage(self):
        """Get CPU usage statistics"""

        # NOTE: use interval=None to make it non-blocking
        cpu = {
            "name": self.cpu_name,
            "core_count": self.cpu_core_count,
            "usage": psutil.cpu_percent(interval=None),
        }

        if self.cpu_frequency:
            cpu["frequency"] = {"unit": 'MHZ', "max": self.cpu_frequency.max}

        return cpu

    def _get_disk_usage(self):
        """Returns disk usage"""
        partitions = []
        for partition in self.partitions:
            name = partition.mountpoint
            usage = psutil.disk_usage(name)
            info = {
                "name": name,
                "unit": "GB",
                "used": int(usage.used / (1024*1024*1024.0)),
                "total": int(usage.total / (1024*1024*1024.0))
            }
            partitions.append(info)
        return partitions

    def _get_software(self):
        """return core software version info"""

        packages = {
            "kerastuner": kt.__version__,
            # Not all tensorflow versions have the __version__ field.
            "tensorflow": tf.__dict__.get('__version__'),
            "tensorflow_use_gpu": self.tf_can_use_gpu,
            "python": platform.python_version(),
            "os": {
                "name": platform.system(),
                "version": platform.version(),
            },
            "gpu_driver": self.gpu_driver_version
        }
        return packages

    def _get_uptime(self):
        return int(time() - psutil.boot_time())

    def _get_memory_usage(self):
        """Returns Memory usage"""
        mem_info = psutil.virtual_memory()
        mem = {
            "unit": "MB",
            "used": int(mem_info.used / (1024*1024.0)),
            "total": int(mem_info.total / (1024*1024.0))
        }
        return mem

    def _find_nvidia_smi(self):
        """
        Find nvidia-smi program used to query the gpu

        Returns:
            str: nvidia-smi path or none if not found
        """

        if platform.system() == "Windows":
            # If the platform is Windows and nvidia-smi
            # could not be found from the environment path,
            # try to find it from system drive with default installation path
            nvidia_smi = spawn.find_executable('nvidia-smi')
            if nvidia_smi is None:
                nvidia_smi = ("{}\\Program Files\\NVIDIA Corporation"
                              "\\NVSMI\\nvidia-smi.exe")
                nvidia_smi.format(os.environ['systemdrive'])
        else:
            nvidia_smi = "nvidia-smi"
        return nvidia_smi

    def _get_gpu_usage(self):
        """gpu usage"""
        if not self.nvidia_smi:
            return []

        metrics = {
                        "index": "index",
                        "utilization.gpu": "usage",
                        "memory.used": "used",
                        "memory.total": "total",
                        "driver_version": "driver",
                        # "cuda_version": "cuda", # doesn't exist
                        "name": "name",
                        "temperature.gpu": "value",
                        "uuid": "uuid"
                    }
        metrics_list = sorted(metrics.keys())  # deterministic ordered list
        query = ','.join(metrics_list)
        try:
            p = Popen([self.nvidia_smi, "--query-gpu=%s" % query,
                      "--format=csv,noheader,nounits"], stdout=PIPE)
            stdout, _ = p.communicate()
        except:
            return []

        info = stdout.decode('UTF-8')
        gpus = []
        for l in info.split('\n'):
            if ',' not in l:
                continue
            info = l.strip().split(',')
            gpu_info = {"memory": {"unit": "MB"}, 'temperature': {"unit": 'C'}}
            for idx, metric in enumerate(metrics_list):
                value = info[idx].strip()
                metric_name = metrics[metric]
                if "memory" in metric:
                    gpu_info['memory'][metric_name] = int(value)
                elif "temperature" in metric:
                    gpu_info['temperature'][metric_name] = int(value)
                elif "driver" in metric:
                    self.gpu_driver_version = value
                else:
                    gpu_info[metric_name] = value
            gpus.append(gpu_info)
        return gpus
