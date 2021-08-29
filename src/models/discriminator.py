""" Module for building a transformer encoder model """
import torch
import torch.nn as nn
import torch.nn.functional as functional


class TransformerEncoder(nn.Module):
    def __init__(self, encoder, word_embedding_weights, position_embedding_weights, embedding_size, dropout, trg_vocab_size, src_pad_idx, max_sequence_length, device):
        super(TransformerEncoder, self).__init__()

        # Note: The true child sequence can either be provided as:
        # - a one-hot encoded tensor of shape (batch_size, sequence_length, trg_vocab_size)
        #   -> Requires also a dense/linear layer
        # - a normal tensor of shape (batch_size, sequence_length) -> Requires an embedding layer
        # We decided for the second choice (aligned with the MutaGAN paper)

        # Embeddings from input
        self.linear = nn.Linear(trg_vocab_size, embedding_size)
        # self.linear.load_state_dict(word_embedding_weights)

        self.embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.embedding.load_state_dict(word_embedding_weights)

        self.position_embedding = nn.Embedding(max_sequence_length, embedding_size)
        self.position_embedding.load_state_dict(position_embedding_weights)

        self.dropout = nn.Dropout(dropout)

        # Transformer
        self.transformerEncoder = encoder

        # Other
        self.src_pad_idx = src_pad_idx
        self.src_pad_pattern = functional.one_hot(torch.tensor(src_pad_idx), trg_vocab_size).to(device)
        self.device = device

    def forward(self, src, parent_sequence=True):
        # shape src:
        # - predicted: (batch_size, sequence_length, trg_vocab_size)
        # - true: (batch_size, sequence_length)

        word_embedding = self.embedding(src) if parent_sequence else self.linear(src)

        batch_size = src.shape[0]
        sequence_length = src.shape[1]
        positions = (
            torch.arange(0, sequence_length)
                .unsqueeze(dim=0)
                .expand(batch_size, sequence_length)
                .to(self.device)
        )
        position_embedding = self.position_embedding(positions)

        embedding = self.dropout(word_embedding + position_embedding)
        src_padding_mask = self.make_src_mask(src, parent_sequence)

        return self.transformerEncoder(embedding, src_key_padding_mask=src_padding_mask)

    def make_src_mask(self, src, true_sequence):
        src_mask = (src == self.src_pad_idx) if true_sequence else (src == self.src_pad_pattern).all(dim=2)
        return src_mask.to(self.device)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(MultiLayerPerceptron, self).__init__()

        # Sequential model
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        self.first_batch_norm = nn.BatchNorm1d(2*embedding_size)
        self.second_batch_norm = nn.BatchNorm1d(embedding_size)
        self.third_batch_norm = nn.BatchNorm1d(int(embedding_size/2))

        self.first_linear = nn.Linear(2*embedding_size, embedding_size)
        self.second_linear = nn.Linear(embedding_size, int(embedding_size/2))
        self.third_linear = nn.Linear(int(embedding_size/2), 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        src = self.dropout(src)
        src = torch.transpose(src, dim0=1, dim1=2)
        src = self.first_batch_norm(src)
        src = torch.transpose(src, dim0=1, dim1=2)
        src = self.first_linear(src)
        src = self.relu(src)

        src = self.dropout(src)
        src = torch.transpose(src, dim0=1, dim1=2)
        src = self.second_batch_norm(src)
        src = torch.transpose(src, dim0=1, dim1=2)
        src = self.second_linear(src)
        src = self.relu(src)

        src = self.dropout(src)
        src = torch.transpose(src, dim0=1, dim1=2)
        src = self.third_batch_norm(src)
        src = torch.transpose(src, dim0=1, dim1=2)
        src = self.third_linear(src)
        src = self.sigmoid(src)

        return src
