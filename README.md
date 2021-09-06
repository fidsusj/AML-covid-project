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

## About the dataset
- All parent and child sequences are of fixed length 29904
- All sequences are positionally aligned and padded to the length of 29904

## Project team

- Felix Hausberger
- Nils Krehl
