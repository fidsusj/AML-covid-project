#!/bin/bash

# echo " - Combine fasta input files into one"
cd data/raw/separate_input_files
cat *.fasta.gz* > ../example_sequences.fasta.gz
