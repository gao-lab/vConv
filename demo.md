# A motif discovery demo of vConv-based model

vConv is a novel convolutional layer, which can replace the classic convolutional layer. Here we provide a demo of vConv's application to motif discovery from ChIP-Seq reads.


## Prerequisites

### Software

- Python 2 and its packages:
  - numpy
  - h5py
  - pandas
  - seaborn
  - scipy
  - keras (version 2.2.4)
  - tensorflow (version 1.3.0)
  - sklearn
  - ushuffle

You can install them with the following `conda` command:
```{bash}
conda env create -f environment_vConv.yml
conda activate vConv
```

# How to run the demo

Run the demo with the following commands:

```{bash}
cd ./vConvbaseddiscovery/code/
python VConvMotifdiscovery.py
```

This script will iterate over all ChIP-Seq read files (in fasta format) under `./vConvbaseddiscovery/demofasta/`. For each of these file, the script will train (and discover motifs from) a separate vConv-based model for each of them with the core function `runvConvC` in `./vConvbaseddiscovery/code/VConvMDcore.py`. 

We have prepared an example input file from ENCODE (`./vConvbaseddiscovery/demofasta/wgEncodeAwgTfbsSydhHelas3Brf1UniPk.fa`, HeLa-S3 TFBS Uniform Peaks of BRF1 from ENCODE/Harvard/Analysis ; downloaded from https://www.genome.ucsc.edu/cgi-bin/hgTrackUi?hgsid=1087730107_iKxBSlJS5lkfutvSOo3tzUwJAaD7&g=wgEncodeAwgTfbsSydhHelas3Brf1UniPk), but the user can test their own ChIP-Seq read files as well.

# Explanation of each step in `runvConvC`

## Generate training and test dataset from fasta file

The codes at Line 155-158 in `runvConvC`:
```{python}
########################generate dataset#########################
GeneRateOneHotMatrixTest = GeneRateOneHotMatrix()
OutputDirHdf5 = OutputDir + "/Hdf5/"
GeneRateOneHotMatrixTest.runSimple(filePath, OutputDirHdf5, SaveData=SaveData)
```
1. Read in the fasta file;
2. Encode the read sequences into an N\*L\*4 tensor (the tensor of positive samples), where N is the number of reads and L is the length of each read;
3. Generate negative samples by shuffling the reads while preserving the distribution of dinucleotides and encode them into another N\*4\*L tensor (the tensor of negative samples); 
4. Mix the two tensors and divided them into the training and test sets.

## Build and train vConv-based neural network

The codes at Line 159-173 in `runvConvC`:
```{python}
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
```
1. Build a vConv-based model in a similar way as illustrated in [README.md](https://github.com/AUAShen/vConv/blob/main/README.md). This model consists of a one-dimensional vConv layer (vConv1D) with 64 kernels of unmasked length=50 and initial length of unmasked region=12, a Max-Pooling layer with pooling length=10, a Global Max-Pooling layer, and finally a Dense layer with a sigmoid activation that outputs a scalar;
2. Train the vConv-based model with the following strategies:
  - A Dropout mechanism (with dropout_rate=0.1) is added between the Global Max-Pooling layer and the Dense layer
  - The loss is the sum of the BCE loss for the prediction + 0.0025 * the Shannon loss from the masked kernels
  - Adadelta is chosen as the optimizer with lr=1, rho=0.99, epsilon=1.0e-8, and decay=0.0
  - The batch size is 100
  - Before training, the training dataset is shuffled, and 10% of it is taken as the validation subset
  - A total number of 1,000 epoch is used with an EarlyStopping mechanism that stops training when the loss on validation dataset does not increase for 50 consecutive epochs
  - Mask parameters start to updates from the 10th epoch.
  
3. Save the trained model at `./vConvbaseddiscovery/result/vConvB/{chipseq_fasta_name}/ModelParameter`

## Discover and visualize the motifs

The codes at Line 175-203 in `runvConvC`:
```{python}
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
```
1. Select only those kernels with absolute value of Dense layer weights large enough (here defined as being larger than the mean minus the standard deviation of absolute values of all Dense layer weights);
2. Select the highest-scored subsequences from positive sequences for each kernel and generate as its motif a Position Weight Matrix (PWM) by normalizing each position's nucleotide composition to 1
3. Save the PWMs at `./vConvbaseddiscovery/result/vConvB/{chipseq_fasta_name}/recover_PWM`



# Possible extensions

1. **Using refined negative datasets**. When negative datasets that are "biologically meaningful" are available, one can skip the shuffling step and use them instead. It is worth noticing that in this situation, the highest-scored subsequences from negative sequences should also be considered for motif discovery, because these negative datasets might also contain motifs.
2. **Identify motifs for sequences other than DNA/RNA.** The **k*L** one-hot encoding of the sequence, where **k** is the alphabet size, might no longer be 4; for example, in protein **k** could be 20.
3. **Build and train multi-layer vConv-based models**. Although the demo model only has one vConv layer, the vConv layer can be adapted into more sophisticated model structures. In our [manuscript](https://doi.org/10.1101/508242), we presented how to build and train a multi-layer vConv-based version of the [Basset model](https://pubmed.ncbi.nlm.nih.gov/27197224/).  





