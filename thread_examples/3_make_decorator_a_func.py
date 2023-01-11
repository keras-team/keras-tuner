import collections
import threading
import time

LOCKS = collections.defaultdict(lambda: threading.Lock())


def synchronized(func, *args, **kwargs):
    def wrapped_func(*args, **kwargs):
        oracle = args[0]
        LOCKS[oracle].acquire()
        ret_val = func(*args, **kwargs)
        LOCKS[oracle].release()
        return ret_val

    return wrapped_func


class Oracle:
    def __init__(self, name):
        self.name = name

    @synchronized
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
