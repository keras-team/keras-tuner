from kerastuner.abstractions.tensorflow import TENSORFLOW as tf


def write_file(path, contents):
    with tf.io.gfile.Open(path, 'w') as output:
        output.write(contents)


def read_file(path, mode='r'):
    with tf.io.gfile.Open(path, mode) as i:
        return i.read()


def create_directory(path, remove_existing=False):
    # Create the directory if it doesn't exist.
    if not tf.io.gfile.exists(path):
        tf.io.gfile.makedirs(path)

    # If it does exist, and remove_existing it specified, the directory will be
    # removed and recreated.
    elif remove_existing:
        tf.io.gfile.rmtree(path)
        tf.io.gfile.makedirs(path)
