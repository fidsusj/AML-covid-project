import torch
from torchtext.data.metrics import bleu_score
from models.transformer import Transformer
from path_helper import get_project_root
from preprocessing.dataset import get_loader, Vocabulary
from training.training_config import embedding_size, dim_feed_forward, num_heads, num_encoder_layers, dropout, num_decoder_layers


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset, data_loader = get_loader(str(get_project_root()) + "/data/dataset/final.csv", batch_size=1, train=False)

    src_vocab_size = len(test_dataset.parent_vocab)
    trg_vocab_size = len(test_dataset.child_vocab)
    max_len = test_dataset.strain_end - test_dataset.strain_begin
    src_pad_idx = test_dataset.parent_vocab.vocab.stoi["<pad>"]

    model = Transformer(embedding_size, dim_feed_forward, num_heads, num_encoder_layers, num_decoder_layers, dropout,
                        src_vocab_size, trg_vocab_size, src_pad_idx, max_len, device).to(device)

    targets = []
    outputs = []

    for instance in test_dataset:
        parent_sequence = vars(instance)["parent"]
        child_sequence = vars(instance)["child"]

        codons = [codon for codon in Vocabulary.tokenizer_codons(parent_sequence)]
        codons.insert(0, "<SOS>")
        codons.append("<EOS>")
        parent_sequence = [test_dataset.vocab.stoi[codon] for codon in codons]
        parent_sequence_tensor = torch.LongTensor(parent_sequence).unsqueeze(1).to(device)

        predicted_sequence = [test_dataset.child_vocab.stoi["<SOS>"]]
        for i in range(max_len):
            predicted_sequence_tensor = torch.LongTensor(predicted_sequence).unsqueeze(1).to(device)

            with torch.no_grad():
                codon = model(parent_sequence_tensor, predicted_sequence_tensor)

            best_guess = codon.argmax(2)[-1, :].item()
            predicted_sequence.append(best_guess)

        predicted_sequence = [test_dataset.child_vocab.itos[idx] for idx in predicted_sequence]
        predicted_sequence = predicted_sequence[1:-1]
        targets.append([child_sequence])
        outputs.append(predicted_sequence)

        score = bleu_score(outputs, targets)
        print(f"Bleu score {score * 100:.2f}")
