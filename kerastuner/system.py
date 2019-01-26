"Compute hardware statistics"

import psutil
import platform
from subprocess import Popen, PIPE
from distutils import spawn
from time import time
import tensorflow as tf
import kerastuner as kt


class System():

    def __init__(self):

        # compute static information once
        self.nvidia_smi = self._find_nvidia_smi()
        self.cpu_core_count = psutil.cpu_count()
        self.partitions = psutil.disk_partitions()
        self.hostname = platform.node()
        self.cpu_name = platform.processor()

        # additional GPU info
        self.tf_can_use_gpu = tf.test.is_gpu_available()  # before get_software
        self._get_gpu_usage()  # to get gpu driver info > before get software

        # keep it last
        self.software = self._get_software()


    def get_status(self):
        """
        Return various hardware counters

        Returns:
            dict: hardware counters
        """
        status = {}

        status['cpu'] = self._get_cpu_usage()
        status['ram'] = self._get_memory_usage()
        status['gpu'] = self._get_gpu_usage()
        status['uptime'] = self._get_uptime()
        status['disk'] = self._get_disk_usage()
        status['software'] = self.software
        status['hostname'] = self.hostname
        status["available_gpu"] = len(status['gpu'])
        return status

    def _get_hostname(self):
        """get system name"""
        print(platform.uname())

    def _get_cpu_usage(self):
        """Get CPU usage statistics"""

        # NOTE: use interval=None to make it non-blocking
        cpu = {
            "name": self.cpu_name,
            "core_count": self.cpu_core_count,
            "usage": psutil.cpu_percent(interval=None),
        }

        freq = psutil.cpu_freq()
        if freq:
            cpu["frequency"] = {
                        "unit": 'MHZ',
                        "current": freq.current,
                        "max": freq.max
                    }

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
            "tensorflow": tf.__version__,
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
                nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
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
            stdout, stderror = p.communicate()
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
