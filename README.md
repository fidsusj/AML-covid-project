# AML Project - Prediction of the next SARS-CoV-2 mutations by a Transformer based GAN Framework

## Introduction  

This project is part of the Advanced Machine Learning course at Heidelberg university. The project is located in the area 
of Covid-19 research. The goal of this project is to predict the next possible mutations of the Covid-19 virus.

During the genome replication random mutations can appear. As a consequence the encoded protein sequence could be changed, which can lead to different behavior. If this change increases the fitness, it is probably passed on to the next generation. 

A currently well known example for a mutating virus is SARS-CoV-2. Due to the developed vaccines the hope for an end of the pandemic arises. Nevertheless this is only true, if the vaccines, which are developed against the wild type of SARS-CoV-2, also remain effective against new mutations. To enable fast responses to new arising mutations it would be helpful to know the possible next mutations in advance. This can influence the treatment and prevention of diseases, by enabling the development of countermeasures and preventive measures in advance.

Machine Learning, especially Deep Learning enabled improvements in lots of different domains. This work applies Deep Learning to the area of virus genome mutation prediction. Due to the fact, that genome sequences could be treated as text data, methods from the NLP area can be applied. The success of Deep Learning for NLP tasks has already been shown in various areas such as text generation, text summarization or translation.

## Research question and proposed novelties  

**Our research question is whether a Machine Learning model can be trained to predict the next possible SARS-CoV-2 mutations. In this work, we propose three novelties:**

- **Model architecture**: A new GAN based architecture influenced by [Berman et al.](https://arxiv.org/abs/2008.11790). Our novelty is the usage of transformers instead of LSTM in the seq2seq model
- **Dataset**: Generation of a dataset for SARS-CoV-2, consisting of 9199 parent-child data instances
- **Application domain**: The training of the network for SARS-CoV-2


## Setup

First install:

- [Anaconda](https://www.anaconda.com/products/individual) to create isolated python environments
- [NVIDIA GeForce Experience](https://www.nvidia.com/de-de/geforce/geforce-experience/) to update your local graphics driver
- [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive) to utilize GPUs during training (see [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html))
- [NVIDIA cuDNN v8.2.1](https://developer.nvidia.com/cudnn) to accelerate DNN implementations on GPU (see [installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html))

Then simply run:

        conda env create
        conda activate aml

## Development guidelines

In case you need to add additional dependencies, adapt the environment.yml file and run

        conda env update

List all installed dependencies with their versions:

        conda list

Run isort from project root:

        isort .

Open tensorboard with:

        tensorboard --logdir .\src\training\tensorboard\pretraining\
        tensorboard --logdir .\src\training\tensorboard\training\discriminator\
        tensorboard --logdir .\src\training\tensorboard\training\generator\

## Data sources

Due to the regulations of the GISAID platform the raw datasources and the dataset are not part of this repository. 
The structure of the data folder, where the raw data can be inserted can be seen in the following image.

![data folder](/data_folder_structure.png)



## Project details

For details about the project (e.g. used dataset, choosen Machine Learning approach, results) see the [report](https://github.com/nilskre/AML-covid-project/blob/main/docs/report/report.pdf).

## Project team

- Felix Hausberger
- Nils Krehl
