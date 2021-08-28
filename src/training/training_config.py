""" Module for hyperparameters """
# Training hyperparameters
num_epochs = 10000
batch_size = 32
learning_rate = 3e-4
save_model_every = 10

# Model hyperparameters
embedding_size = 128  # Must be dividable by 4
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
dim_feed_forward = 512
training_model = "24_08_2021_13_28_10.pth.tar"
evaluation_model = "24_08_2021_13_28_10.pth.tar"
