"keraslyzer related functions"
from os import path
from tensorflow.python.lib.io import file_io # allows to write to GCP or local
from termcolor import cprint

def cloud_save(local_path, ftype, meta_data, debug=0):
    """Stores file remotely to Kerastuner service or fork

    Args:
        local_path (str): where the file is saved localy.
        ftype (str): type of file saved -- results, weights, executions, config 
    """

    if 'gs_dir' not in meta_data:
        cprint('No gs_dir available')
        return
    if ftype not in ['config', 'results', 'weights', 'execution']:
        Exception('Invalid ftype: ', fname)

    if ".json" in local_path:
        binary = ''
        fname = "%s.json" % ftype
    elif '.h5' in local_path:
        binary = 'b'
        fname = "%s.h5" % ftype
    else:
        Exception("unknown file type for file:", local_path)

    if 'execution' in meta_data:
        fname = "%s-%s" % (meta_data['execution'], fname)

    remote_path = path.join(meta_data['gs_dir'],  meta_data['username'], meta_data['project'],  meta_data['architecture'], meta_data['instance'], fname)
    
    if debug:
        cprint("[INFO] Uploading %s to %s" % (local_path, remote_path), 'cyan')

    with file_io.FileIO(local_path, mode= 'r' + binary) as input_f:
        with file_io.FileIO(remote_path, mode=  binary + 'w+') as output_f:
            output_f.write(input_f.read())

