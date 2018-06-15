"keraslyzer related functions"
from os import path
from tensorflow.python.lib.io import file_io # allows to write to GCP or local
from termcolor import cprint

def cloud_save(category, architecture, instance, local_path, execution=None, training_size=None, gs_dir=None, debug=1):
  """Stores file remotely in a given GS bucket path."""
  if not gs_dir:
    return
  if ".json" in local_path:
    ftype = '%s.json' % (execution or training_size or 'results')
    binary = ''
  elif '.h5' in local_path:
    ftype = '%s.h5' % (execution or training_size or 'results')
    binary = 'b'
  else:
    Exception("unknown  file type for file:", local_path)

  remote_path = path.join(gs_dir, architecture, instance, category, ftype)
  if debug:
      cprint("[INFO] Uploading %s to %s" % (local_path, remote_path), 'cyan')

  with file_io.FileIO(local_path, mode= 'r' + binary) as input_f:
    with file_io.FileIO(remote_path, mode=  binary + 'w+') as output_f:
      output_f.write(input_f.read())

