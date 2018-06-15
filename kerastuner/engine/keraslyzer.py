"keraslyzer related functions"
from os import path
from tensorflow.python.lib.io import file_io # allows to write to GCP or local
from termcolor import cprint

def cloud_save(category, architecture, instance, local_path, execution=None, training_size=None, gs_dir=None, debug=1):
  """Stores file remotely in a given GS bucket path."""
  if not gs_dir:
    return
  ftype = '%s.json' % (execution or training_size or 'results')
  remote_path = path.join(gs_dir, architecture, instance, category, ftype)
  if debug:
      cprint("[INFO] Uploading %s to %s" % (local_path, remote_path), 'cyan')
  with file_io.FileIO(local_path, mode='r') as input_f:
    with file_io.FileIO(remote_path, mode='w+') as output_f:
      output_f.write(input_f.read())

