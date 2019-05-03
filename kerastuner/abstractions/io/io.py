from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
import os


def Open(name, mode):
    """Open a file.

    Args:
        name (str): name of the file
        mode (str): one of 'r', 'w', 'a', 'r+', 'w+', 'a+'. Append 'b' for bytes mode.

    Returns:
        GFile - a GFile object representing the opened file.      
    """
    return tf.io.gfile.Open(name, mode)


def makedirs(path):
    """Creates a directory and all parent/intermediate directories.

    It succeeds if path already exists and is writable.

    Args:
        path (str): string, name of the directory to be created

    Raises:
        errors.OpError: If the operation fails.
    """
    return tf.io.gfile.makedirs(path)


def exists(path):
    """Determines whether a path exists or not.
    Args:
        path: string, a path

    Returns:
        True if the path exists, whether it's a file or a directory.
        False if the path does not exist and there are no filesystem errors.

    Raises:
        errors.OpError: Propagates any errors reported by the FileSystem API.
    """
    return tf.io.gfile.exists(path)


def rmtree(path):
    """Deletes everything under path recursively.

    Args:
      path: string, a path

    Raises:
      errors.OpError: If the operation fails.
    """
    return tf.io.gfile.rmtree(path)


def glob(pattern):
    """Returns a list of files that match the given pattern(s).

    Args:
      pattern: string or iterable of strings. The glob pattern(s).

    Returns:
      A list of strings containing filenames that match the given pattern(s).

    Raises:
      errors.OpError: If there are filesystem / directory listing errors.
    """
    return tf.io.gfile.glob(pattern)


def remove(path):
    """Deletes the path located at 'path'.

    Args:
      path: string, a path

    Raises:
      errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
      NotFoundError if the path does not exist.
    """
    return tf.io.gfile.remove(path)


def copy(src, dst, overwrite=False):
    """Copies data from src to dst.

    Args:
      src: string, name of the file whose contents need to be copied
      dst: string, name of the file to which to copy to
      overwrite: boolean, if false its an error for newpath to be occupied by an
        existing file.

    Raises:
      errors.OpError: If the operation fails.
    """
    return tf.io.gfile.copy(src, dst, overwrite=overwrite)


def write_file(path, contents):
    """Writes the specified string to a file.

    Args:
      path: string, name of the file to write.
      contents: string, contents to write to the file.
    """
    with tf.io.gfile.Open(path, 'w') as output:
        output.write(contents)


def read_file(path, binary_mode=False):
    """Return the contents of a file.

    Args:
      path: string, name of the file to read.
      binary_mode: Boolean, True if the file contents are binary.

    Returns:
      A the contents of the file.
    """
    mode = "rb" if binary_mode else "r"
    with tf.io.gfile.Open(path, mode) as i:
        return i.read()


def create_directory(path, remove_existing=False):
    """Create a directory, potentially removing and recreating the directory.

    Args:
      path: string, path for the directory to create.
      remove_existing: Boolean, if True and the specified directory exists,
          remove it and recreate the directory.
    """

    # Create the directory if it doesn't exist.
    if not tf.io.gfile.exists(path):
        tf.io.gfile.makedirs(path)

    # If it does exist, and remove_existing it specified, the directory will be
    # removed and recreated.
    elif remove_existing:
        tf.io.gfile.rmtree(path)
        tf.io.gfile.makedirs(path)


def get_config_filename(tuner_state, instance_state, execution_state):
    """ Get the filename containing the model configuration for the given instance/execution.

    Args:
      tuner_state: TunerState, state of the tuner.
      instance: Instance, the instance for which we are getting the configuration.
      execution: Execution, the execution for which we are getting the configuration.

    Returns:
      str, the file which holds the model configuration.
    """
    results_dir = tuner_state.host.result_dir
    project = tuner_state.project
    architecture = tuner_state.architecture
    instance_id = instance_state.idx
    execution_id = execution_state.idx

    return os.path.join(results_dir, "%s-%s-%s-%s-config.json" % (
        project, architecture, instance_id, execution_id))


def get_weights_filename(tuner_state, instance_state, execution_state):
    """ Get the filename containing the model weights for the given instance/execution.

    Args:
      tuner_state: TunerState, state of the tuner.
      instance: Instance, the instance for which we are getting the weights.
      execution: Execution, the execution for which we are getting the weights.

    Returns:
      str, the file which holds the model configuration.
    """
    results_dir = tuner_state.host.result_dir
    project = tuner_state.project
    architecture = tuner_state.architecture
    instance_id = instance_state.idx
    execution_id = execution_state.idx

    return os.path.join(results_dir, "%s-%s-%s-%s-weights.h5" % (
        project, architecture, instance_id, execution_id))


def get_results_filename(tuner_state, instance_state):
    """ Get the filename containing the results metadata for the given instance.

    Args:
      tuner_state: TunerState, state of the tuner.
      instance: Instance, the instance for which we are getting the weights.

    Returns:
      str, the file which holds the model configuration.
    """
    results_dir = tuner_state.host.result_dir
    project = tuner_state.project
    architecture = tuner_state.architecture
    instance_id = instance_state.idx

    return "%s-%s-%s-results.json" % (project, architecture, instance_id)
    return os.path.join(results_dir, "%s-%s-%s-results.json" % (
        project, architecture, instance_id))


def reload_model(tuner_state, instance_state, execution_state, compile=False):
    """Reload the model for the given instance and execution.

    Args:
        tuner_state: TunerState, the state of the tuner. 
        instance_state: InstanceState, the instance to reload.
        execution_state: ExecutionState, the execution to reload.
        compile: bool, if true, compile the model before returning it.

    Returns:
        tf.keras.models.Model, the reloaded model.
    """

    model = instance_state.recreate_model()
    weights = get_weights_filename(tuner_state, instance_state,
                                   execution_state)
    model.load_weights(weights)

    if compile:
        model.compile(loss=model.loss, optimizer=model.optimizer)
    return model
