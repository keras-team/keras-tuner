def compute_max_batch_size(model, gpu_mem, max_size=8000):
    """Compute the largest batch size possible
      max_size (int): maximum batch size. 8k seems the limit based of https://research.fb.com/publications/accurate-large-minibatch-sgd-training-imagenet-in-1-hour/
    """

    batch_size, num_params = __compute_batch_size(model, gpu_mem, max_size)


def __compute_batch_size(model, memory, max_size):
    "Find the largest batch size usuable so we maximize ressources usage"
    batch_size = 16
    while 1:
        bs = batch_size + 2
        if bs >= max_size:
            break
        memory_needed, model_num_params = __get_model_memory_usage(model, bs)
        if memory_needed > memory:
            break
        batch_size = bs
    return batch_size, model_num_params


def __get_model_memory_usage(model, batch_size):
    "comput the memory usage for a given model and batch "
    import numpy as np
    from tensorflow.keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p)
                              for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p)
                                  for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size * \
        (shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    # print("train count ", trainable_count, "mem per instance", 
    # total_memory, "gbytes ", gbytes)
    return gbytes, trainable_count

import platform
from distutils import spawn
from subprocess import Popen, PIPE


if platform.system() == "Windows":
    # If the platform is Windows and nvidia-smi 
    # could not be found from the environment path, 
    # try to find it from system drive with default installation path
    nvidia_smi = spawn.find_executable('nvidia-smi')
    if nvidia_smi is None:
        nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
else:
    nvidia_smi = "nvidia-smi"

def get_gpu_usage():
    if not nvidia_smi:
        return []
    try:
        p = Popen([nvidia_smi,"--query-gpu=index,utilization.gpu,memory.used,memory.total,name,temperature.gpu", "--format=csv,noheader,nounits"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []
    info = stdout.decode('UTF-8')
    gpus = []
    for l in info.split('\n'):
        if ',' not in l:
            continue
        l = l.strip().split(',')

        gpus.append(l)
    return gpus
