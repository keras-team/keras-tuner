import threading
import time


class DecoratorClass(object):
    def __init__(self):
        self.thread = None
        self.lock = threading.Lock()

    def __call__(self, func, *args, **kwargs):
        def wrapped_func(*args, **kwargs):

            self.lock.acquire()

            curr_thread = threading.currentThread().getName()
            self.thread = curr_thread

            print("\nthread name before running func:", self.thread)
            ret_val = func()
            print("\nthread name after running func:", self.thread)
            self.lock.release()
            return ret_val

        return wrapped_func


@DecoratorClass()
def decorated_with_class():
    print("running decorated w class")
    time.sleep(1)
    return


threads = []
for i in range(5):
    t = threading.Thread(target=decorated_with_class)
    threads.append(t)
    t.start()
