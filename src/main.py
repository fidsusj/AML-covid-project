""" Main Module """
#from create_dataset.create_dataset import run_create_dataset
from training.evaluate import evaluate
from training.train import run_gan_training, run_transformer_pretraining

if __name__ == "__main__":
    #run_create_dataset(run_from_scratch=False)
    run_transformer_pretraining(model_training_from_scratch=True)
    #run_gan_training(model_training_from_scratch=True)
    evaluate(pretraining=True)
