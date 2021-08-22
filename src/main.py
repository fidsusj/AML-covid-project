""" Main Module """
from training.evaluate import evaluate
from training.train import run_training

if __name__ == "__main__":
    # run_training(model_training_from_scratch=False)
    evaluate()
