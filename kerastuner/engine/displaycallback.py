import tensorflow as tf
from kerastuner.abstractions.system import System
from kerastuner.abstractions.display import get_progress_bar
import time
import numpy as np
from multiprocessing.pool import ThreadPool


class DisplayCallback(tf.keras.callbacks.Callback):
    """
    Callback which prints the model progress and results. The
    default Keras progress bar does not function properly in a
    Colab/Jupyter notebook.
    """

    def __init__(self,
                 num_steps,
                 metrics_to_display="all",
                 stats_refresh_rate=5):
        self.pbar = None
        self.num_steps = num_steps
        self.stats_refresh_rate = stats_refresh_rate

        # Display system utilization (CPU/GPU)
        self.system = System()
        self.status = self.system.get_status()
        self.thread_pool = ThreadPool(1)
        self.thread_pool.apply_async(self._report_status_worker)
        self.last_display_time = time.time()

        self.metrics = {}

    def on_epoch_begin(self, epoch, logs={}):
        self.pbar = get_progress_bar(
            desc="",
            unit=" steps",
            total=1)

    def _report_status_worker(self):
        while True:
            self.status = self.system.get_status()
            time.sleep(self.stats_refresh_rate)

    def on_batch_end(self, batch, logs={}):
    
        if time.time() - self.last_display_time < 0.5:            
            return

        self.last_display_time = time.time()

        for metric_name, values in logs.items():
            if metric_name == "batch" or metric_name == "size": 
                continue
            self.metrics[metric_name] = "%4f" % np.average(values)

        desc = ""
        desc += "[CPU: %s%%]" % self.status["cpu"]["usage"]

        for gpu in self.status['gpu']:
            desc += "[GPU%s: %s%%]" % (gpu["index"], gpu["usage"])

        desc += "[Mem: %s/%s%s]" % (
            self.status["ram"]["used"],
            self.status["ram"]["total"],
            self.status["ram"]["unit"]
        )

        self.pbar.set_description(desc)
        self.pbar.set_postfix(self.metrics)

        # fix this
        # self.pbar.write(logs)

    def on_epoch_end(self, epoch, logs={}):
        self.pbar.close()
        #table = [["Metric", "Value"]]
        # for k, v in logs.items():
        #    table.append([k, v])
        #print("Epoch %d results:" % epoch)
        # print_table(table)
