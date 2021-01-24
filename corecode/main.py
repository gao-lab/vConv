# -*- coding: utf-8 -*-
import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)
    print("stop warning")
import tensorflow as tf
from build_models import *
import sys
import os
import h5py
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
    # if sorted(memory_gpu_map)[0][1]==1:
    #     best_memory, best_gpu = sorted(memory_gpu_map)[1]
    # else:
    #     best_memory, best_gpu = sorted(memory_gpu_map)[0]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)


def load_data(dataset):
    """
    load training and test data set
    :param dataset: path of dataset
    :return:
    """
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return ([sequence_code, label])

def if_trained(path):
    """
    find if the model has been trained
    :param path: model path
    :return:
    """
    return os.path.isfile(path+"best_info.txt")


def get_session(gpu_fraction=0.5):
    """
    Select the size of the GPU memory used
    :param gpu_fraction:  Proportion
    :return:
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)


if __name__ == "__main__":

    # choose the GPU memory


    data_path = sys.argv[1]
    result_root = sys.argv[2]
    data_info = sys.argv[3]
    mode = sys.argv[4]
    KernelLen = int(sys.argv[5])
    KernelNum = int(sys.argv[6])
    RandomSeed = int(sys.argv[7])
    try:
        lr = 1
        rho = float(sys.argv[8])
        epsilon = float(sys.argv[9])
    except:
        lr=1
        rho = 0.99
        epsilon = 1.0e-8



    # loading the data set
    test_dataset = data_path + "test.hdf5"
    training_dataset = data_path + "train.hdf5"
    X_test, Y_test = load_data(test_dataset)
    X_train, Y_train = load_data(training_dataset)
    data_set = [[X_train,Y_train],[X_test,Y_test]]
    seq_len = X_test[0].shape[0]
    input_shape = X_test[0].shape

    # init hyper-parameter
    max_ker_len = min(int(seq_len * 0.5),40)
    batch_size = 100

    # get_session()

    # Determine the type of model
    if mode == "CNN":
        print("training CNN")
        time.sleep(2)
        result_path = result_root + data_info
        mkdir(result_path)
        modelsave_output_prefix = result_path + '/CNN/'
        mkdir(modelsave_output_prefix)

        auc, info = train_CNN(input_shape = input_shape,modelsave_output_prefix=modelsave_output_prefix,
                               data_set = data_set, number_of_kernel=KernelNum, kernel_size=KernelLen,
                               random_seed=RandomSeed, batch_size=batch_size,epoch_scheme=1000,lr=lr,rho=rho,epsilon=epsilon)

    elif mode == "vCNN":
        print("training vCNN")
        time.sleep(2)

        result_path = result_root + data_info
        mkdir(result_path)
        modelsave_output_prefix = result_path + '/vCNN/'
        mkdir(modelsave_output_prefix)
        kernel_init_dict = {str(KernelLen): KernelNum}
        auc, info = train_vCNN(input_shape=input_shape, modelsave_output_prefix=modelsave_output_prefix,
                               data_set=data_set, number_of_kernel=KernelNum,
                               init_ker_len_dict=kernel_init_dict, max_ker_len=max_ker_len,
                               random_seed=RandomSeed, batch_size=batch_size, epoch_scheme=1000,lr=lr,rho=rho,epsilon=epsilon)

    elif mode == "vCNNNSHL":
        print("training vCNNNSHL")
        time.sleep(2)

        result_path = result_root + data_info
        mkdir(result_path)
        modelsave_output_prefix = result_path + '/vCNNNSHL/'
        mkdir(modelsave_output_prefix)
        kernel_init_dict = {str(KernelLen): KernelNum}
        auc, info = train_vCNNnoSHL(input_shape=input_shape, modelsave_output_prefix=modelsave_output_prefix,
                               data_set=data_set, number_of_kernel=KernelNum,
                               init_ker_len_dict=kernel_init_dict, max_ker_len=max_ker_len,
                               random_seed=RandomSeed, batch_size=batch_size, epoch_scheme=1000,lr=lr,rho=rho,epsilon=epsilon)




