#!/bin/bash

echo "     -> Create conda envs"
source ~/miniconda3/etc/profile.d/conda.sh

conda update  -y -n base conda
conda install -y -n base -c conda-forge mamba

mamba create -y -n nextstrain -c conda-forge -c bioconda \
  python=3.8.10 augur auspice nextstrain-cli nextalign snakemake awscli git pip

conda activate nextstrain
nextstrain check-setup --set-default

echo "     -> Download workflow"
cd create_dataset
DIR="ncov"
if [ -d "$DIR" ]; then
  echo "     -> Worflow repo exists: update workflow ${DIR}"
  cd ncov
  git pull
else
  echo "     -> Worflow repo does not exist: clone workflow repo ${DIR}"
  git clone https://github.com/nextstrain/ncov.git
  cd ncov
fi

echo "     -> Copy input data files (sequences, metadata, build.yaml) to ncov/data directory"
rm -r data/*
cp ../../data/raw/example_metadata.tsv data/
cp ../../data/raw/example_sequences.fasta.gz data/
rm my_profiles/getting_started/builds.yaml
cp ../builds.yaml my_profiles/getting_started

echo "     -> Run phylogenetic analysis with ncov"
nextstrain build . --cores 4 --use-conda \
  --configfile ./my_profiles/getting_started/builds.yaml

echo "     -> Copy results (sequence alignments, metadata, phylogenetic tree) to data folder"
cp -u results/aligned_focal.fasta.xz ../../data/dataset
cp -u results/sanitized_metadata_focal.tsv.xz ../../data/dataset
cp -u results/global/tree.nwk ../../data/dataset
cd ../../data/dataset
xz -f --decompress aligned_focal.fasta.xz
xz -f --decompress sanitized_metadata_focal.tsv.xz
