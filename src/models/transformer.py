import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, embedding_size, dim_feed_forward, num_heads, num_encoder_layers, num_decoder_layers, dropout, src_vocab_size, trg_vocab_size, src_pad_idx,
                 max_sequence_length, device):
        super(Transformer, self).__init__()

        # Embeddings from input
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_sequence_length, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_sequence_length, embedding_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers,
                                          num_decoder_layers, dim_feed_forward, dropout)

        # Output probability distribution
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.softmax = nn.Softmax(dim=2)

        # Other
        self.src_pad_idx = src_pad_idx
        self.device = device

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
                .unsqueeze(1)
                .expand(src_seq_length, N)
                .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
                .unsqueeze(1)
                .expand(trg_seq_length, N)
                .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )

        out = self.softmax(self.fc_out(out))
        return out

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)
