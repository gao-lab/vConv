# A motif discovery demo of vConv-based model

Here we provide a demo of vConv's application in motif discovery using chipseq pick data as input. The input file is in fasta format, each reads is a pick sequence identified from chipseq experiment. The demo will train the model and output model's parameters and the predicted motifs.


# Folder structure:


**../demofasta/**  input folder, saves fasta files. The demo script will first build a vConv-based model. For each fasta file under this folder, it will generate input data and train the model on the dataset. Finally a set of motifs will be generated for each input fasta file. 


**../result/vConvB/** output folder. For each input fasta file, the script will generate a subfolder under this directory, under which predicted motifs will be saved in **recover_PWM** folder and model's parameters will be saved in **ModleParaMeter**


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

Alternatively, if you want to guarantee working versions of each dependency, you can install via a fully pre-specified environment.
```{bash}
conda env create -f environment_vConv.yml
```

# Overview of the pipeline

Under the vConvbaseddiscovery/code/ folder, when running the command "python VConvMotifdiscovery.py", a demo script of vConv's application in motif discovery will be executed. This demo shows one of the real-world applications of vConv layer. Detailed workflow is explained below. It is worth to note that vConv is a key component of the model in this demo. Although the demo model only has one vConv layer, the vConv layer has the capability to be adapted into more sophisticated model structures. In an other word, vConv is a generalised convolution layer, which can be applied to a variety of model structures. In our manuscript [reference to add], we presented examples of multi-layers vConv based neural network.    


## Generating training dataset from fasta file

The input files are in fasta format. Each sequence is collected from a chipseq peak [reference to the data source]. The first step is to generate "negative" samples by shuffling the chipseq reads, while keeping the dimer frequency [double check if this is true]. Then the reads are one-hot represented in to a 4*L matrix, where L is the length of a read. Finally, both "positive" and "negative" samples are mixed together and subdivided into "training set" and "test set".  

## Build vConv based neural network

A vConv based model is built in a similar way as illustrated in **README.md**. vConv is a novel convolutional layer, which can replace the classic conv layer. Shannon loss should be also added into the final loss function to fully use vConv layer's function. 

## Train the model

The model is trained different from classical CNN network. [detailed training strategy refer to the manuscript]

## Motif visualization

After the training process, motifs (PWM format) are recovered from kernels. we select the sequence fragments with the highest convolutional value belong to each kernel to rebuild the motif. 

# General applications

The demo code here can be applied to a variety of motif discovery problems. For example, given any set of sequences of interest, identify the common motifs. These motifs are shared, conserved region among the input reads.  





#
