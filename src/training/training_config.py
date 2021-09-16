""" Module for hyperparameters """
# Training hyperparameters
num_epochs = 1000 #10000
batch_size = 32
learning_rate = 3e-4  # Maybe choose different learning rates for pretraining, generator and discriminator
save_model_every = 10

# Model hyperparameters
embedding_size = 128  # Must be dividable by 4
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
dim_feed_forward = 512

# Models
pretraining_model = "16_09_2021_15_11_10.pth.tar"
training_model = ""
evaluation_model = "16_09_2021_15_11_10.pth.tar"
