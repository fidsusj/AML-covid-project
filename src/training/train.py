""" Module for model training """
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.transformer import Transformer
from path_helper import get_project_root
from preprocessing.dataset import get_loader
from training.training_config import (dim_feed_forward, dropout,
                                      embedding_size, learning_rate,
                                      num_decoder_layers, num_encoder_layers,
                                      num_epochs, num_heads, save_model_every,
                                      training_model)


def run_training(model_training_from_scratch=False):
    # Config and load data
    print("Loading data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, data_loader = get_loader(str(get_project_root()) + "/data/dataset/final.csv", test_set_size=0.05, batch_size=32, train=True)

    src_vocab_size = len(train_dataset.parent_vocab)
    trg_vocab_size = len(train_dataset.child_vocab)
    max_len = train_dataset.strain_end - train_dataset.strain_begin
    src_pad_idx = train_dataset.parent_vocab.stoi["<PAD>"]

    # Tensorboard to get nice loss plot
    writer = SummaryWriter("training/tensorboard/loss_plot")
    step = 0

    # Training objects
    print("Initialize model...")
    model = Transformer(embedding_size, dim_feed_forward, num_heads, num_encoder_layers, num_decoder_layers, dropout,
                        src_vocab_size, trg_vocab_size, src_pad_idx, max_len, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    epochs_trained = 0
    if not model_training_from_scratch:
        epochs_trained = load_checkpoint(model, optimizer)

    # Training loop
    print("Starting training...")
    model.train()  # Set model to training mode
    for epoch in range(epochs_trained, num_epochs):
        print(f"[Epoch {epoch+1} / {num_epochs}]")

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

        if (epoch+1) % save_model_every == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs_trained": epoch+1
            }
            save_checkpoint(checkpoint, epoch+1)


def save_checkpoint(state, epoch_count):
    print("Saving checkpoint...")
    training_timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    torch.save(state, f"training/checkpoints/{training_timestamp}_{epoch_count}.pth.tar")


def load_checkpoint(model, optimizer):
    print("Loading checkpoint...")
    checkpoint = torch.load(f"training/checkpoints/{training_model}")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epochs_trained"]
