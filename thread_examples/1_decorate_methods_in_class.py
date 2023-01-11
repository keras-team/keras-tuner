import threading
import time


class DecoratorClass:
    def __init__(self):
        self.thread = None
        self.lock = threading.Lock()

    def __call__(self, func, *args, **kwargs):
        def wrapped_func(*args, **kwargs):

            self.lock.acquire()

            curr_thread = threading.currentThread().getName()
            self.thread = curr_thread

            print("\nthread name before running func:", self.thread)
            print(args[0])
            ret_val = func(*args, **kwargs)
            print("\nthread name after running func:", self.thread)
            self.lock.release()
            return ret_val

        return wrapped_func


class ExampleClass:
    @DecoratorClass()
    def run(self):
        print("running decorated w class")
        time.sleep(1)
        return


def thread_function():
    ExampleClass().run()


threads = []
for i in range(5):
    t = threading.Thread(target=thread_function)
    threads.append(t)
    t.start()
