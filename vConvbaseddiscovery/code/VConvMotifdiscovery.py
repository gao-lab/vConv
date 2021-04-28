# -*- coding: utf-8 -*-
import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)
    print("stop warning")
import os
import sys
import tensorflow as tf
import pdb

from VConvMDcore import runvConvC

from keras.backend.tensorflow_backend import set_session

import subprocess, re

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def get_session():
    """
    Select the size of the GPU memory used
    :return:
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # sess = tf.Session(config=config)
    # set_session(sess)

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False




if __name__ == '__main__':
    outputRoot = "../result/vConvB/"
    import glob
    mkdir(outputRoot)
    fileNamelist =glob.glob("../demofasta/*")
    try:
        get_session()
    except:
        print("No GPU")
    
    HyperParaMeters={
        "kernel_init_size":12,
        "number_of_kernel":64,
        "max_ker_len":50,
        "batch_size":100,
        "epoch_scheme": 1000,
        "random_seed": 233
    }
    
    for fileName in fileNamelist:
        Name = fileName.split("/")[-1].split(".")[0]
        outputDir = outputRoot + Name + "/"
        if not os.path.exists(outputDir + "/over.txt"):
            runvConvC(fileName,outputDir,HyperParaMeters)
        else:
            print("already trained")

