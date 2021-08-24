""" Main Module """
from create_dataset.create_dataset import run_create_dataset
from training.evaluate import evaluate
from training.train import run_training

if __name__ == "__main__":
    # run_create_dataset(run_from_scratch=False)
    run_training(model_training_from_scratch=True)
    # evaluate()
