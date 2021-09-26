## Features to be implemented

Create Dataset:
- Split the data into training, test and validation dataset on .csv file level

Preprocessing:
- Minimal frequency to be added to the vocabulary?

Discriminator:
- Flatten MLP input? -> Shape (batch_size, sequence_length * embedding_size) or (batch_size, sequence_length, embedding_size)

Generator:
- Beam search instead of greedy decoding?

Training:
- Loss of generator for GAN: Wasserstein + sparse categorical cross entropy
