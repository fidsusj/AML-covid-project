""" Main Module """

from create_dataset.create_dataset import run_create_dataset
from preprocessing.preprocessing import run_preprocessing

if __name__ == "__main__":
    RECREATE_DATASET_FROM_FASTA_FILES = False
    PREPROCESSING = False
    MODEL_TRAINING = False

    df_dataset = run_create_dataset(RECREATE_DATASET_FROM_FASTA_FILES)
    df_preprocessed = run_preprocessing(PREPROCESSING, df_dataset)
