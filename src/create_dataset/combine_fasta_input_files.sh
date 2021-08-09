#!/bin/bash

# echo " - Combine fasta input files into one"
cd data/raw/separate_input_files
cat *.fasta* > ../example_sequences.fasta
cd ..
gzip -f example_sequences.fasta
