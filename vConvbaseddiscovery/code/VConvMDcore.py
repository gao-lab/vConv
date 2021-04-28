import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)
    print("stop warning")
import os
import numpy as np
from build_models import *
from seq_to_matrix import *
import glob
import pdb
from keras import backend as K
from datetime import datetime
def tictoc():
    return datetime.now().minute * 60 + datetime.now().second + datetime.now().microsecond*(10**-6)
import gc
from sklearn import mixture
import scipy.stats


###############vCNN based model######################################

def recover_ker(model, modeltype, KernelIndex=0):
    """
    restore mask
    :param resultPath:
    :param modeltype:
    :param input_shape:
    :return:
    """
    try:
        KernelIndex.shape
    except:
        KernelIndex = range(K.get_value(model.layers[0].kernel).shape[2])



    def CutKerWithMask(MaskArray, KernelArray):

        CutKernel = []
        for Kid in range(KernelArray.shape[-1]):
            MaskTem = MaskArray[:, :, Kid].reshape(2, )
            leftInit = int(round(max(MaskTem[0] - 3, 0), 0))
            rightInit = int(round(min(MaskTem[1]+ 3, KernelArray.shape[0] - 1), 0))
            if rightInit - leftInit >= 5:
                kerTem = KernelArray[leftInit:rightInit, :, Kid]
                CutKernel.append(kerTem)
        return CutKernel

    # load model
    if modeltype == "CNN":
        kernelTem = K.get_value(model.layers[0].kernel)[:,:,KernelIndex]
        kernel = []
        for i in range(kernelTem.shape[2]):
            kernel.append(kernelTem[:,:,i])

    elif modeltype == "vCNN":
        k_weights = K.get_value(model.layers[0].k_weights)[:,:,KernelIndex]
        kernelTem = K.get_value(model.layers[0].kernel)[:,:,KernelIndex]
        kernel = CutKerWithMask(k_weights, kernelTem)
    else:
        kernel = model.layers[0].get_kernel()[:,:,KernelIndex] * model.layers[0].get_mask()[:,:,KernelIndex]
    return kernel

def NormPwm(seqlist, Cut=False):
    """
    Incoming seqlist returns the motif formed by the sequence
    :param seqlist:
    :return:
    """
    SeqArray = np.asarray(seqlist)
    Pwm = np.sum(SeqArray,axis=0)
    Pwm = Pwm / Pwm.sum(axis=1, keepdims=1)
    
    if not Cut:
        return Pwm

    return Pwm

def KSselect(KernelSeqs):
    """
    Screen the corresponding sequence for the selected kernel
    :param KSconvValue:
    :param KernelSeqs:
    :return:
    """
    PwmWork = NormPwm(KernelSeqs, True)

    return PwmWork

def KernelSeqDive(tmp_ker, seqs, Pos=True):
    """
    The kernel extracts the fragments on each sequence and the corresponding volume points.
     At the same time retain the position information on the sequence fragments mined by the kernel [sequence number, sequence start position, end position]
    :param tmp_ker:
    :param seqs:
    :return:
    """
    ker_len = tmp_ker.shape[0]
    inputs = K.placeholder(seqs.shape)
    ker = K.variable(tmp_ker.reshape(ker_len, 4, 1))
    conv_result = K.conv1d(inputs, ker, padding="valid", strides=1, data_format="channels_last")
    max_idxs = K.argmax(conv_result, axis=1)
    max_Value = K.max(conv_result, axis=1)
    # sort_idxs = tensorflow.nn.top_k(tensorflow.transpose(max_Value,[1,0]), 100, sorted=True).indices

    f = K.function(inputs=[inputs], outputs=[max_idxs, max_Value])
    ret_idxs, ret = f([seqs])

    if Pos:
        seqlist = []
        SeqInfo = []
        for seq_idx in range(ret.shape[0]):
            start_idx = ret_idxs[seq_idx]
            seqlist.append(seqs[seq_idx, start_idx[0]:start_idx[0] + ker_len, :])
            SeqInfo.append([seq_idx, start_idx[0], start_idx[0] + ker_len])
        del f
        return seqlist, ret, np.asarray(SeqInfo)
    else:
        return ret


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



############

def runvConvC(filePath, OutputDir, HyperParaMeters, SaveData=False):
    """
    Generate the final motif directly from the fasta file
    :param filePath: directory of fasta files
    :param OutputDir: The model where all output results are located
    :param HyperParaMeters: All the hyperparameters of vConv
    :param SaveData: Whether to store hdf5 data results
    :return:
    """
    
    if os.path.exists(OutputDir+"/over.txt"):
        print("already trained")
        return 0
    
    ########################generate dataset#########################
    GeneRateOneHotMatrixTest = GeneRateOneHotMatrix()
    OutputDirHdf5 = OutputDir + "/Hdf5/"
    GeneRateOneHotMatrixTest.runSimple(filePath, OutputDirHdf5, SaveData=SaveData)
    ########################train model#########################
    data_set = [[GeneRateOneHotMatrixTest.TrainX, GeneRateOneHotMatrixTest.TrainY],
                [GeneRateOneHotMatrixTest.TestX, GeneRateOneHotMatrixTest.TestY]]
    dataNum = GeneRateOneHotMatrixTest.TrainY.shape[0]

    input_shape = GeneRateOneHotMatrixTest.TestX[0].shape
    modelsave_output_prefix = OutputDir + "/ModelParameter/"
    kernel_init_dict = {str(HyperParaMeters["kernel_init_size"]): HyperParaMeters["number_of_kernel"]}
    
    auc, info, model = train_vCNN(input_shape=input_shape, modelsave_output_prefix=modelsave_output_prefix,
                                  data_set=data_set, number_of_kernel=HyperParaMeters["number_of_kernel"],
                                  init_ker_len_dict=kernel_init_dict, max_ker_len=HyperParaMeters["max_ker_len"],
                                  random_seed=HyperParaMeters["random_seed"],
                                  batch_size=HyperParaMeters["batch_size"],
                                  epoch_scheme=HyperParaMeters["epoch_scheme"])
    
    ############Select Kernel ####################
    
    DenseWeights = K.get_value(model.layers[4].kernel)
    meanValue = np.mean(np.abs(DenseWeights))
    std = np.std(np.abs(DenseWeights))
    workWeightsIndex = np.where(np.abs(DenseWeights) > meanValue-std)[0]
    kernels = recover_ker(model, "vCNN", workWeightsIndex)
    print("get kernels")

    PwmWorklist = []
    for ker_id in range(len(kernels)):
        kernel = kernels[ker_id]

        KernelSeqs, KSconvValue, seqinfo = KernelSeqDive(kernel, GeneRateOneHotMatrixTest.seq_pos_matrix_out,)
        KernelSeqs = np.asarray(KernelSeqs)
        PwmWork = NormPwm(KernelSeqs, True)
        PwmWorklist.append(PwmWork)


    
    pwm_save_dir = OutputDir + "/recover_PWM/"
    mkdir(pwm_save_dir)
    for i in range(len(PwmWorklist)):
        mkdir(pwm_save_dir + "/")
        np.savetxt(pwm_save_dir + "/" + str(i) + ".txt", PwmWorklist[i])

    del model, KernelSeqs, KSconvValue, seqinfo
    gc.collect()
    np.savetxt(OutputDir + "/over.txt", np.zeros(1))


