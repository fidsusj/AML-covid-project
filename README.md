# AML Project - Prediction of the next SARS-CoV-2 variants

## Introduction  

This project is part of the Advanced Machine Learning course at Heidelberg university. The project is located in the area 
of Covid-19 research. The goal of this project is to:

- Predict mutations of the Covid-19 virus


## Setup

First install:

- [Anaconda](https://www.anaconda.com/products/individual) to create isolated python environments
- [NVIDIA GeForce Experience](https://www.nvidia.com/de-de/geforce/geforce-experience/) to update your local graphics driver
- [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive) to utilize GPUs during training (see [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html))
- [NVIDIA cuDNN v8.2.1](https://developer.nvidia.com/cudnn) to accelerate DNN implementations on GPU (see [installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html))

Then create a conda environment and download all necessary dependencies for this project:

        conda create -n aml python=3.9.6
        conda activate aml
        conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
        conda install pandas
        conda install -c anaconda biopython        
        conda install -c conda-forge spacy
        conda install -c conda-forge cupy
        conda install -c conda-forge tensorboard
        conda install -c pytorch torchtext
        conda install -c conda-forge scikit-learn

From within the conda terminal from the aml environment (choose run with python) run:

        $ import pip
        $ pip.main(['install', 'dendropy'])

## Project team

- Felix Hausberger
- Nils Krehl

## Open topics

Documentation:
- Add motivation section to the README
- Provide Docker image parallel to setup guide

Data:
- What is your sequence max length? -> 33
- Dictionary for source and target sequences for BLEU score

      score = bleu_score([["AGA", "G", "T", "T", "T"], ["AGB", "G", "T", "T", "T"]], [[["AGA", "G", "T", "T", "T"]], [["AGA", "G", "T", "T", "T"]]])

- Can sequences be of different length? -> Add padding_idx parameter to transformer
- Wie lange sind die sequences ursprünglich? 
- Warum sind im dictionary auch 'N' und '-' drinnen? => Kann man doch vorher raus nehmen aus den sequencen
- Logik zur auswahl von parent-child sequences verbessern

- Only choose sequences from [101,200]

- Train, Test, Val Dataset
- Flake und Isort
- We expect sequences to be padded

- Im Datensatz sind nur gleiche genome
- legacy löschen
- Numnericalize only onvce
- Does eval set the dropouts off in our custom module?
- Training config
- Moduke kommentare
- Training
- 