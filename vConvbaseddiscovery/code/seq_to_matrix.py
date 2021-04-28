# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
import random
import glob
from sklearn.model_selection import StratifiedKFold
import os
from ushuffle import Shuffler, shuffle
import pdb



"""
sequence to matrix
"""

class GeneRateOneHotMatrix():

    def __init__(self):

        self.OriSeq = None
        self.Seqlabel = None
        self.SeqMatrix = None
        self.TrainX = None
        self.TestX = None
        self.TrainY = None
        self.TestY = None
        self.TrainID = None
        self.TestID = None
        self.Trainlist = []
        self.Testlist = []
        self.seq_pos_matrix_out = None
        
    def mkdir(self, path):
        """
        :param path:
        :return:
        """
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return (True)
        else:
            return False

    def seq_to_matrix(self,seq, seq_matrix, seq_order):
        '''
        change target 3D tensor according to sequence and order
        :param seq: Input single root sequence
        :param seq_matrix: Input initialized matrix
        :param seq_order: This is the first sequence
        :return:
        '''
        for i in range(len(seq)):
            if ((seq[i] == 'A') | (seq[i] == 'a')):
                seq_matrix[seq_order, i, 0] = 1
            if ((seq[i] == 'C') | (seq[i] == 'c')):
                seq_matrix[seq_order, i, 1] = 1
            if ((seq[i] == 'G') | (seq[i] == 'g')):
                seq_matrix[seq_order, i, 2] = 1
            if ((seq[i] == 'T') | (seq[i] == 't')):
                seq_matrix[seq_order, i, 3] = 1
        return seq_matrix

    def genarate_matrix_for_train(self, seq_shape, seq_series):
        """
        generate a large matrix composed of sequences.
        :param shape: (seq number, sequence_length, 4)
        :param seq_series: A file in dataframe format composed of seq
        :return:seq
        """
        seq_matrix = np.zeros(seq_shape)
        for i in range(seq_series.shape[0]):
            seq_tem = seq_series[i]
            seq_matrix = self.seq_to_matrix(seq_tem, seq_matrix, i)
        return seq_matrix

    def cross_validation(self, number_of_folds, total_number, random_seeds=233):
        """
        This function is used to generate the index of n-fold cross validation
        :param number_of_folds:
        :param total_number:
        :param random_seeds:
        :return:
        """
        x = np.zeros((total_number,), dtype=np.int)
        split_iterator = StratifiedKFold(n_splits=number_of_folds, random_state=random_seeds, shuffle=True)
        split_train_index_and_test_index_list = [
            (train_index, test_index)
            for train_index, test_index in split_iterator.split(x, x)
            ]
        return (split_train_index_and_test_index_list)

    def split_dataset(self, split_index_list, fold, data_x, data_y, data_id=None):
        """
        Return training set and test set as needed
        :param split_index_list:Index of each fold returned by cross_validation
        :param fold:
        :param data_id:
        :param data_x:X
        :param data_y:Y
        :return:
        """
        id_train = data_id[split_index_list[fold][0].tolist()]
        x_train = data_x[split_index_list[fold][0].tolist()]
        y_train = data_y[split_index_list[fold][0].tolist()]
        id_test = data_id[split_index_list[fold][1].tolist()]
        x_test = data_x[split_index_list[fold][1].tolist()]
        y_test = data_y[split_index_list[fold][1].tolist()]
        return [x_train, y_train, id_train, x_test, y_test, id_test]

    def StoreTrainSet(self, rootPath):
        """
        Store multi-fold  data
        :param rootPath:
        :param ValNum: Generate valNum different validation data sets
        :param RandomSeeds:
        :param allData:
        :return:
        """
        self.mkdir(rootPath)
        training_path = rootPath + "/training_set.hdf5"
        test_path = rootPath + "/test_set.hdf5"

        f_train = h5py.File(training_path)
        f_test = h5py.File(test_path)

        f_train.create_dataset("sequences", data=self.TrainX)
        f_train.create_dataset("labs", data=self.TrainY)
        f_train.create_dataset("seq_idx", data=self.TrainID)
        f_train.close()
        f_test.create_dataset("sequences", data=self.TestX)
        f_test.create_dataset("labs", data=self.TestY)
        f_test.create_dataset("seq_idx", data=self.TestID)
        f_test.close()

    def k_mer_shuffle(self,seq_shape, seq_series, k=2):
        """
        Return a shuffled negative matrix
        :param seq_series:sequence list
        :param seq_shape:
        :param k:kshuffle
        :return:
        """
        seq_shuffle_matrix = np.zeros(seq_shape)

        for i in range(seq_shape[0]):
            seq = seq_series[i]
            shuffler = Shuffler(seq, k)
            seqres = shuffler.shuffle()
            seq_shuffle_matrix = self.seq_to_matrix(seqres, seq_shuffle_matrix, i)

        return seq_shuffle_matrix


    def runSeveral(self, SeqPath, OutputDir, SaveData=False):
        """
        Call this function to directly generate data, seq format requirements, odd-numbered lines are names,
        even-numbered lines are corresponding sequences
        :return:
        """
        allFileFaList = glob.glob(SeqPath)
        for allFileFa in allFileFaList:
            AllTem = allFileFa.split("/")[-1].split(".")[0]
            output_dir = OutputDir + AllTem + "/"
            SeqLen = 0
            ChipSeqlFileFa = pd.read_csv(allFileFa, sep='\t', header=None, index_col=None)
            seq_positive_series = np.asarray(ChipSeqlFileFa[1::2]).reshape(np.asarray(ChipSeqlFileFa[1::2]).shape[0], )
            seq_positive_shape = seq_positive_series.shape[0]
            for i in range(seq_positive_shape):
                SeqLen = max(SeqLen, len(seq_positive_series[i]))
            seq_positive_name = np.asarray(ChipSeqlFileFa[::2]).reshape(seq_positive_shape, )

            self.seq_pos_matrix_out = self.genarate_matrix_for_train((seq_positive_shape, SeqLen, 4),
                                                           seq_positive_series)
            seq_pos_label_out = np.ones(seq_positive_shape, )

            seq_negative_name = seq_positive_name
            seq_neg_matrix_out = self.k_mer_shuffle((seq_positive_shape, SeqLen, 4), seq_positive_series)
            seq_neg_label_out = np.zeros(seq_positive_shape, )

            seq = np.concatenate((self.seq_pos_matrix_out, seq_neg_matrix_out), axis=0)
            label = np.concatenate((seq_pos_label_out, seq_neg_label_out), axis=0)
            id_tem = np.concatenate((seq_positive_name, seq_negative_name), axis=0)
            index_shuffle = range(seq_positive_shape + seq_positive_shape)
            random.shuffle(index_shuffle)
            seq_matrix_out = seq[index_shuffle, :, :]
            label_out = label[index_shuffle]
            id_out = id_tem[index_shuffle].astype("string_")
            outData = [seq_matrix_out, label_out, id_out]
            self.GeneRateTrain(allData=outData, ValNum=10, RandomSeeds=233)
            if SaveData:
                self.StoreTrainSet(rootPath=output_dir)

    def GeneRateTrain(self, allData, ValNum=10, RandomSeeds=233):
        """

        """
        dataNum = allData[1].shape[0]
        split_train_index_and_test_index_list = self.cross_validation(number_of_folds=ValNum, total_number=dataNum,
                                                                 random_seeds=RandomSeeds)
        i = 0
        outDataTem = self.split_dataset(split_train_index_and_test_index_list, fold=i, data_x=allData[0], data_y=allData[1],
                                   data_id=allData[2])

        self.TrainX = outDataTem[0]
        self.TestX = outDataTem[3]
        self.TrainY = outDataTem[1]
        self.TestY = outDataTem[4]
        self.TrainID = outDataTem[2]
        self.TestID = outDataTem[5]

    def runSimple(self, SeqPath, OutputDir, SaveData=False):
        """
        Call this function to directly generate data, here is a separate data set, including Training and test
        seq format requirements, odd-numbered lines are names, even-numbered lines are corresponding sequences
        :return:
        """
        SeqLen = 0
        ChipSeqlFileFa = pd.read_csv(SeqPath, sep='\t', header=None, index_col=None)
        self.OriSeq = np.asarray(ChipSeqlFileFa[1::2]).reshape(np.asarray(ChipSeqlFileFa[1::2]).shape[0], )
        seq_positive_shape = self.OriSeq.shape[0]
        for i in range(seq_positive_shape):
            SeqLen = max(SeqLen, len(self.OriSeq[i]))
        seq_positive_name = np.asarray(ChipSeqlFileFa[::2]).reshape(seq_positive_shape, )

        self.seq_pos_matrix_out = self.genarate_matrix_for_train((seq_positive_shape, SeqLen, 4),
                                                       self.OriSeq)
        seq_pos_label_out = np.ones(seq_positive_shape, )

        seq_negative_name = seq_positive_name
        seq_neg_matrix_out = self.k_mer_shuffle((seq_positive_shape, SeqLen, 4), self.OriSeq)
        seq_neg_label_out = np.zeros(seq_positive_shape, )

        seq = np.concatenate((self.seq_pos_matrix_out, seq_neg_matrix_out), axis=0)
        label = np.concatenate((seq_pos_label_out, seq_neg_label_out), axis=0)
        id_tem = np.concatenate((seq_positive_name, seq_negative_name), axis=0)
        index_shuffle = range(seq_positive_shape + seq_positive_shape)
        random.shuffle(index_shuffle)
        self.SeqMatrix = seq[index_shuffle, :, :]
        self.Seqlabel = label[index_shuffle]
        id_out = id_tem[index_shuffle].astype("string_")
        outData = [self.SeqMatrix, self.Seqlabel, id_out]
        self.GeneRateTrain(allData=outData, ValNum=10, RandomSeeds=233)
        if SaveData:
            self.StoreTrainSet(rootPath=OutputDir)

if __name__ == '__main__':
    base = {0:"A",1:"C",2:"G",3:"T"}
    allFileFaList = glob.glob("./chip-seqFa/*fa")
    OutputDir = "./chip-seqFa/TrainingData/"
    GeneRateOneHotMatrixTest = GeneRateOneHotMatrix()

    GeneRateOneHotMatrixTest.runSeveral("./chip-seqFa/*fa", OutputDir, SaveData=True)

    GeneRateOneHotMatrixTest.runSimple(allFileFaList[0], OutputDir, SaveData=True)
