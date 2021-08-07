""" Main Module """

from src.create_dataset.create_dataset import run_create_dataset
from src.models.model_trainer import train_models
from src.preprocessing.preprocessing import run_preprocessing

if __name__ == "__main__":
    RECREATE_DATASET_FROM_FASTA_FILES = False
    PREPROCESSING = True
    MODEL_TRAINING = False

    df_dataset = run_create_dataset(RECREATE_DATASET_FROM_FASTA_FILES)
    df_preprocessed = run_preprocessing(PREPROCESSING, df_dataset)
    train_models(MODEL_TRAINING, df_preprocessed)
