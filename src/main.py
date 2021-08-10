""" Main Module """

from create_dataset.create_dataset import run_create_dataset
from models.model_trainer import train_models
from preprocessing.preprocessing import run_preprocessing

if __name__ == "__main__":
    RECREATE_DATASET_FROM_FASTA_FILES = True
    PREPROCESSING = True
    MODEL_TRAINING = False

    df_dataset = run_create_dataset(RECREATE_DATASET_FROM_FASTA_FILES)
    df_preprocessed = run_preprocessing(PREPROCESSING, df_dataset)
    train_models(MODEL_TRAINING, df_preprocessed)
