import collections
import threading
import time

LOCKS = collections.defaultdict(lambda: threading.Lock())
THREAD = collections.defaultdict(lambda: None)


def synchronized(func, *args, **kwargs):
    def wrapped_func(*args, **kwargs):
        oracle = args[0]
        thread_name = threading.currentThread().getName()
        need_acquire = THREAD[oracle] != thread_name

        if need_acquire:
            LOCKS[oracle].acquire()
            THREAD[oracle] = thread_name
        ret_val = func(*args, **kwargs)
        if need_acquire:
            THREAD[oracle] = None
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


class MyOracle(Oracle):
    @synchronized
    def create_trial(self):
        super().create_trial()
        print("this is subclass.")


oracle_instances = [MyOracle("oracle_0"), MyOracle("oracle_1")]


def thread_function(i):
    oracle = oracle_instances[i % 2]
    oracle.create_trial()


threads = []
for i in range(5):
    t = threading.Thread(target=thread_function, args=(i,))
    threads.append(t)
    t.start()
