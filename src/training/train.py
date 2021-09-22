""" Module for model training """
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.discriminator import TransformerEncoder
from models.generator import Transformer
from path_helper import get_project_root
from preprocessing.dataset import get_loader
from training.training_config import (batch_size, dim_feed_forward, dropout,
                                      embedding_size,
                                      learning_rate_discriminator,
                                      learning_rate_generator,
                                      learning_rate_pretraining,
                                      num_decoder_layers, num_encoder_layers,
                                      num_epochs_pretraining,
                                      num_epochs_training, num_heads,
                                      pretraining_model, save_model_every,
                                      training_model)

################################
#  Plain Transformer
################################


def run_transformer_pretraining(model_training_from_scratch=False):
    # Config and load data
    print("Loading data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, data_loader = get_loader(str(get_project_root()) + "/data/dataset/final.csv", test_set_size=0.05, batch_size=batch_size, train=True)

    src_vocab_size = len(train_dataset.parent_vocab)
    trg_vocab_size = len(train_dataset.child_vocab)
    max_len = train_dataset.strain_end - train_dataset.strain_begin
    src_pad_idx = train_dataset.parent_vocab.stoi["<PAD>"]

    # Tensorboard to get nice loss plot
    writer = SummaryWriter("training/tensorboard/pretraining/loss_plot")
    step = 0

    # Training objects
    print("Initialize model...")
    model = Transformer(embedding_size, dim_feed_forward, num_heads, num_encoder_layers, num_decoder_layers, dropout,
                        src_vocab_size, trg_vocab_size, src_pad_idx, max_len, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_pretraining)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    epochs_trained = 0
    if not model_training_from_scratch:
        epochs_trained, step = load_checkpoint_pretraining(model, optimizer)

    # Training loop
    print("Starting training...")
    model.train()  # Set model to training mode
    for epoch in range(epochs_trained, num_epochs_pretraining):
        print(f"[Epoch {epoch + 1} / {num_epochs_pretraining}]")

        losses = []
        for _, batch in enumerate(data_loader):
            # Batch shape: (2, batch_size, sequence_length)
            parent_sequence = batch[0].to(device)
            child_sequence = batch[1].to(device)

            # Forward propagation + Remove <EOS> token of target sequence (input of decoder is shifted one to the right)
            output = model(parent_sequence, child_sequence[:, :-1])

            # Calculate the loss
            # Reshape (batch_size, sequence_length, features) -> (batch_size * sequence_length, features) for CrossEntropyLoss + Remove <SOS> token
            output = output.reshape(-1, output.shape[2])
            child_sequence = child_sequence[:, 1:].reshape(-1)
            loss = criterion(output, child_sequence)
            losses.append(loss.item())

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure gradients are within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Batch gradient descent step
            optimizer.step()

            # Plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)

        if (epoch + 1) % save_model_every == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs_trained": epoch + 1,
                "step": step,
            }
            save_checkpoint_pretraining(checkpoint, epoch + 1)


################################
#  Transformer inside GAN
################################

# Note: We first need to do a pretraining of the generator using the normal cross entropy loss

def run_gan_training(model_training_from_scratch=False):
    # Config and load data
    print("Loading data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, data_loader = get_loader(str(get_project_root()) + "/data/dataset/final.csv", test_set_size=0.05, batch_size=batch_size, train=False)

    src_vocab_size = len(train_dataset.parent_vocab)
    trg_vocab_size = len(train_dataset.child_vocab)
    max_len = train_dataset.strain_end - train_dataset.strain_begin
    src_pad_idx = train_dataset.parent_vocab.stoi["<PAD>"]

    # Tensorboard to get nice loss plot
    writer_discriminator = SummaryWriter("training/tensorboard/training/discriminator/loss_plot")
    writer_generator = SummaryWriter("training/tensorboard/training/generator/loss_plot")

    # Training objects
    print("Initialize model...")
    generator = Transformer(embedding_size, dim_feed_forward, num_heads, num_encoder_layers, num_decoder_layers, dropout,
                            src_vocab_size, trg_vocab_size, src_pad_idx, max_len, device).to(device)

    word_embedding_weights = generator.trg_word_embedding.state_dict()
    position_embedding_weights = generator.trg_position_embedding.state_dict()
    discriminator = TransformerEncoder(copy.deepcopy(generator.transformer.encoder), word_embedding_weights, position_embedding_weights,
                                       embedding_size, dropout, trg_vocab_size, src_pad_idx, max_len, device).to(device)

    optimizer_discriminator = optim.Adam(generator.parameters(), lr=learning_rate_discriminator)
    scheduler_discriminator = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_discriminator, factor=0.1, patience=10, verbose=True
    )

    optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate_generator)
    scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_generator, factor=0.1, patience=10, verbose=True
    )

    criterion = nn.BCELoss()  # PyTorch does not offer WassersteinLoss

    if not model_training_from_scratch:
        epochs_trained, step = load_checkpoint_training(discriminator, generator, optimizer_discriminator, optimizer_generator)
    else:
        epochs_trained, step = load_checkpoint_pretraining(generator, optimizer_generator)

    # Training loop
    print("Starting training...")
    # Set models to training mode
    generator.train()
    discriminator.train()
    for epoch in range(epochs_trained, num_epochs_training):
        print(f"[Epoch {epoch + 1} / {num_epochs_training}]")

        losses_discriminator = []
        losses_generator = []
        for _, batch in enumerate(data_loader):
            # Batch shape: (2, batch_size, sequence_length)
            parent_sequence = batch[0].to(device)
            child_sequence = batch[1].to(device)

            ############################################
            # Train discriminator
            # max log(D(x)) + log(1 - D(G(z)))
            ############################################

            # Forward propagation through generator + Remove <EOS> token of target sequence (input of decoder is shifted one to the right)
            prediction = generator(parent_sequence, child_sequence[:, :-1])
            prediction = torch.cat(
                (
                    functional.one_hot(torch.tensor(train_dataset.child_vocab.stoi["<SOS>"]), trg_vocab_size)
                        .view(1, 1, -1)  # Add dimension 0 and 1, dimension 2 remains (trg_vocab_size)
                        .expand(parent_sequence.shape[0], -1, -1)  # Expand batch_size times -> Shape (batch_size, 1, trg_vocab_size)
                        .to(device),
                    prediction
                ), dim=1)  # Prepend <SOS> to predicted output again!

            # Forward propagation through discriminator
            fake = discriminator(parent_sequence, prediction).view(-1)

            # The true child sequence must also use the dense/linear layer in the discriminator, not the embedding layer
            one_hot_encoded_child_sequence = functional.one_hot(child_sequence, trg_vocab_size).float()
            real = discriminator(parent_sequence, one_hot_encoded_child_sequence).view(-1)

            # Calculate the loss
            loss_fake = criterion(fake, torch.zeros_like(fake))
            loss_real = criterion(real, torch.ones_like(real))
            loss_total_discriminator = (loss_real + loss_fake) / 2
            losses_discriminator.append(loss_total_discriminator.item())

            # Backward propagation
            discriminator.zero_grad()
            loss_total_discriminator.backward(retain_graph=True)

            # Clip to avoid exploding gradient issues, makes sure gradients are within a healthy range
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)

            # Batch gradient descent step
            optimizer_discriminator.step()

            # Plot to tensorboard
            writer_discriminator.add_scalar("Training loss", loss_total_discriminator, global_step=step)

            ############################################
            # Train generator
            # min log(1 - D(G(z))) <-> max log(D(G(z))
            ############################################

            # Forward propagation through discriminator reusing the previous output of the generator
            output = discriminator(parent_sequence.detach(), prediction.detach()).view(-1)

            # Calculate the loss
            loss_total_generator = criterion(output, torch.ones_like(output))
            losses_generator.append(loss_total_generator.item())

            # Backward propagation
            generator.zero_grad()
            loss_total_generator.backward()

            # Clip to avoid exploding gradient issues, makes sure gradients are within a healthy range
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1)

            # Batch gradient descent step
            optimizer_generator.step()

            # Plot to tensorboard
            writer_generator.add_scalar("Training loss", loss_total_generator, global_step=step)
            step += 1

        mean_loss_discriminator = sum(losses_discriminator) / len(losses_discriminator)
        scheduler_discriminator.step(mean_loss_discriminator)

        mean_loss_generator = sum(losses_generator) / len(losses_generator)
        scheduler_generator.step(mean_loss_generator)

        if (epoch + 1) % save_model_every == 0:
            checkpoint_discriminator = {
                "state_dict": discriminator.state_dict(),
                "optimizer": optimizer_discriminator.state_dict(),
                "epochs_trained": epoch + 1,
                "step": step,
            }
            checkpoint_generator = {
                "state_dict": generator.state_dict(),
                "optimizer": optimizer_generator.state_dict(),
                "epochs_trained": epoch + 1,
                "step": step,
            }
            save_checkpoint_training(checkpoint_discriminator, checkpoint_generator, epoch + 1)


################################
#  Utils
################################


def save_checkpoint_pretraining(state, epoch_count):
    print("Saving checkpoint...")
    training_timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    torch.save(state, f"training/checkpoints/pretraining/{training_timestamp}_{epoch_count}.pth.tar")


def save_checkpoint_training(state_discriminator, state_generator, epoch_count):
    print("Saving checkpoint...")
    training_timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    torch.save(state_discriminator, f"training/checkpoints/training/discriminator/{training_timestamp}_{epoch_count}.pth.tar")
    torch.save(state_generator, f"training/checkpoints/training/generator/{training_timestamp}_{epoch_count}.pth.tar")


def load_checkpoint_pretraining(model, optimizer):
    print("Loading checkpoint...")
    checkpoint = torch.load(f"{str(get_project_root())}/training/checkpoints/pretraining/{pretraining_model}")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epochs_trained"], checkpoint["step"]


def load_checkpoint_training(model_discriminator, model_generator, optimizer_discriminator, optimizer_generator):
    print("Loading checkpoint...")
    checkpoint_discriminator = torch.load(f"{str(get_project_root())}/training/checkpoints/training/discriminator/{training_model}")
    checkpoint_generator = torch.load(f"{str(get_project_root())}/training/checkpoints/training/generator/{training_model}")
    model_discriminator.load_state_dict(checkpoint_discriminator["state_dict"])
    optimizer_discriminator.load_state_dict(checkpoint_discriminator["optimizer"])
    model_generator.load_state_dict(checkpoint_generator["state_dict"])
    optimizer_generator.load_state_dict(checkpoint_generator["optimizer"])
    return checkpoint_discriminator["epochs_trained"], checkpoint_discriminator["step"]
