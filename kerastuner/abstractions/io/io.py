import os
import tensorflow

# Cover the tensorflow name changes, so we can handle tf 1.x and tf2.x
if int(tensorflow.__version__.split(".")[0]) == 1:
  from tensorflow.gfile import Open, MakeDirs, Exists # nopep8 pylint: disable=import-error
  from tensorflow.gfile import DeleteRecursively, Glob # nopep8 pylint: disable=import-error
  from tensorflow.gfile import Remove, Copy # nopep8 pylint: disable=import-error
else:
  from tensorflow.io.gfile import GFile as Open
  from tensorflow.io.gfile import makedirs as MakeDirs
  from tensorflow.io.gfile import exists as Exists
  from tensorflow.io.gfile import rmtree as DeleteRecursively
  from tensorflow.io.gfile import glob as Glob
  from tensorflow.io.gfile import remove as Remove
  from tensorflow.io.gfile import copy as Copy

def open_file(filename, mode):
    return Open(filename, mode)

def write_file(path, contents):
    with open_file(path, 'w') as output:
        output.write(contents)


def read_file(path, mode='r'):
    with open_file(path, mode) as i:
        return i.read()


def copy_file(src, dest, overwrite=False):
    Copy(src, dest, overwrite=overwrite)


def exists(path):
    return Exists(path)


def create_directory(path, remove_existing=False):
    # Create the directory if it doesn't exist.
    if not Exists(path):
        MakeDirs(path)

    # If it does exist, and remove_existing it specified, the directory will be
    # removed and recreated.
    elif remove_existing:
        DeleteRecursively(path)
        MakeDirs(path)


def rm(file):
    return Remove(file)


def glob(pattern):
    return Glob(pattern)
