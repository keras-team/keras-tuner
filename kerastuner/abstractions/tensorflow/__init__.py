from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow

MAJOR_VERSION = int(tensorflow.__version__.split(".")[0])
MINOR_VERSION = int(tensorflow.__version__.split(".")[1])

__UNSUPPORTED_VERSION_MSG = """
FATAL: Unsupported tensorflow version: '%s'.  Kerastuner currently
supports Tensorflow 2.0.x and Tensorflow 1.y (y >=1.12)
""" % tensorflow.__version__


def get():
    if MAJOR_VERSION == 2:
        from . import tensorflow_2_x as proxy
        tf = proxy.Tensorflow_2_x()
        return tf, proxy.Utils_2_x(tf)
    elif MAJOR_VERSION == 1:
        from . import tensorflow_1_x as proxy
        tf = proxy.Tensorflow_1_x()
        return tf, proxy.Utils_1_x(tf)

    print(__UNSUPPORTED_VERSION_MSG)
    exit(1)


_results = get()
TENSORFLOW = _results[0]
TENSORFLOW_UTILS = _results[1]
