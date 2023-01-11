import collections
import threading
import time


class Synchronized:
    def __init__(self):
        self.locks = collections.defaultdict(lambda: threading.Lock())

    def __call__(self, func, *args, **kwargs):
        def wrapped_func(*args, **kwargs):
            oracle = args[0]
            self.locks[oracle].acquire()
            ret_val = func(*args, **kwargs)
            self.locks[oracle].release()
            return ret_val

        return wrapped_func


class Oracle:
    def __init__(self, name):
        self.name = name

    @Synchronized()
    def create_trial(self):
        print(self.name, "started")
        time.sleep(1)
        print(self.name, "ended")
        return


oracle_instances = [Oracle("oracle_0"), Oracle("oracle_1")]


def thread_function(i):
    oracle = oracle_instances[i % 2]
    oracle.create_trial()


threads = []
for i in range(5):
    t = threading.Thread(target=thread_function, args=(i,))
    threads.append(t)
    t.start()
