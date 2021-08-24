""" Module for model evaluation """
import torch
from torchtext.data.metrics import bleu_score

from models.transformer import Transformer
from path_helper import get_project_root
from preprocessing.dataset import get_loader
from training.training_config import (dim_feed_forward, dropout,
                                      embedding_size, evaluation_model,
                                      num_decoder_layers, num_encoder_layers,
                                      num_heads)


def evaluate():
    # Config and load data
    print("Loading data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset, data_loader = get_loader(str(get_project_root()) + "/data/dataset/final.csv", test_set_size=0.05, batch_size=1, train=False)

    src_vocab_size = len(test_dataset.parent_vocab)
    trg_vocab_size = len(test_dataset.child_vocab)
    max_len = test_dataset.strain_end - test_dataset.strain_begin
    src_pad_idx = test_dataset.parent_vocab.stoi["<PAD>"]

    # Evaluation objects
    print("Initialize model...")
    model = Transformer(embedding_size, dim_feed_forward, num_heads, num_encoder_layers, num_decoder_layers, dropout,
                        src_vocab_size, trg_vocab_size, src_pad_idx, max_len, device).to(device)
    checkpoint = torch.load(f"training/checkpoints/{evaluation_model}")
    model.load_state_dict(checkpoint["state_dict"])
    targets = []
    outputs = []

    # Evaluation loop
    print("Starting evaluation...")
    model.eval()  # Set model to evaluation mode (e.g. deactivate dropout)
    for instance_number, instance in enumerate(data_loader):
        print(f"[Instance {instance_number+1} / {len(test_dataset)}]")

        parent_sequence = instance[0].to(device)
        child_sequence = instance[1].to(device)

        predicted_sequence = torch.LongTensor([test_dataset.child_vocab.stoi["<SOS>"]]).unsqueeze(1).to(device)
        for i in range(child_sequence.shape[1]):  # Loop over each word in the sequence, sequences are always of the same length!
            with torch.no_grad():
                codon = model(parent_sequence, predicted_sequence)

            best_guess = codon.argmax(2)[:, -1].item()
            predicted_sequence = torch.cat((predicted_sequence, torch.LongTensor([best_guess]).unsqueeze(1).to(device)), dim=1)

        predicted_sequence = [test_dataset.child_vocab.itos[idx] for idx in predicted_sequence.flatten().tolist()]
        predicted_sequence = predicted_sequence[1:-1]  # Remove <SOS> and <EOS> token

        child_sequence = [test_dataset.child_vocab.itos[idx] for idx in child_sequence.flatten().tolist()]
        child_sequence = child_sequence[1:-1]  # Remove <SOS> and <EOS> token

        targets.append([child_sequence])
        outputs.append(predicted_sequence)

    score = bleu_score(outputs, targets)
    print(f"Bleu score {score * 100:.2f}")
