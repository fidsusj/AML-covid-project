import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer
from path_helper import get_project_root
from preprocessing.dataset import get_loader
from torch.utils.tensorboard import SummaryWriter
from training.training_config import embedding_size, dim_feed_forward, num_heads, num_encoder_layers, dropout, num_decoder_layers, learning_rate, num_epochs


def run_training(model_training_from_scratch=False):
    # Config and load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, data_loader = get_loader(str(get_project_root()) + "/data/dataset/final.csv", batch_size=32, train=True)

    src_vocab_size = len(train_dataset.parent_vocab)
    trg_vocab_size = len(train_dataset.child_vocab)
    max_len = train_dataset.strain_end - train_dataset.strain_begin
    src_pad_idx = train_dataset.parent_vocab.vocab.stoi["<pad>"]

    # Tensorboard to get nice loss plot
    writer = SummaryWriter("runs/loss_plot")
    step = 0

    # Training objects
    model = Transformer(embedding_size, dim_feed_forward, num_heads, num_encoder_layers, num_decoder_layers, dropout,
                        src_vocab_size, trg_vocab_size, src_pad_idx, max_len, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    if model_training_from_scratch:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        losses = []
        for batch_idx, batch in enumerate(data_loader):
            # Get input and targets and get to cuda
            parent_sequence = batch.src.to(device)
            child_sequence = batch.trg.to(device)

            # Forward prop
            output = model(parent_sequence, child_sequence[:-1, :])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            child_sequence = child_sequence[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, child_sequence)
            losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
