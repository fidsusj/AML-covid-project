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
        # We decided for the first choice (aligned with the MutaGAN paper)

        # Embeddings from input
        self.linear = nn.Linear(trg_vocab_size, embedding_size)
        # self.linear.load_state_dict(word_embedding_weights)  # Does not work yet

        self.embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.embedding.load_state_dict(word_embedding_weights)

        self.position_embedding = nn.Embedding(max_sequence_length, embedding_size)
        self.position_embedding.load_state_dict(position_embedding_weights)

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.transformerEncoder = encoder

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

        # Other
        self.src_pad_idx = src_pad_idx
        self.src_pad_pattern = functional.one_hot(torch.tensor(src_pad_idx), trg_vocab_size).to(device)
        self.device = device

    def forward(self, src, trg):
        # shape src: (batch_size, sequence_length)
        # shape trg: (batch_size, sequence_length, trg_vocab_size)

        batch_size = src.shape[0]
        sequence_length = src.shape[1]

        embedding_parent = self.embedding(src)
        embedding_child = self.linear(trg)

        positions = (
            torch.arange(0, sequence_length)
                .unsqueeze(dim=0)
                .expand(batch_size, sequence_length)
                .to(self.device)
        )
        position_embedding = self.position_embedding(positions)

        embedding_parent = self.dropout(embedding_parent + position_embedding)
        embedding_child = self.dropout(embedding_child + position_embedding)

        src_padding_mask_parent = self.make_src_mask(src, parent_sequence=True)
        src_padding_mask_child = self.make_src_mask(trg, parent_sequence=False)

        encoded_parent = self.transformerEncoder(embedding_parent, src_key_padding_mask=src_padding_mask_parent)
        encoded_child = self.transformerEncoder(embedding_child, src_key_padding_mask=src_padding_mask_child)

        encoder_result = torch.cat((encoded_parent, encoded_child), dim=2)

        result = self.dropout(encoder_result)
        result = torch.transpose(result, dim0=1, dim1=2)
        result = self.first_batch_norm(result)
        result = torch.transpose(result, dim0=1, dim1=2)
        result = self.first_linear(result)
        result = self.relu(result)

        result = self.dropout(result)
        result = torch.transpose(result, dim0=1, dim1=2)
        result = self.second_batch_norm(result)
        result = torch.transpose(result, dim0=1, dim1=2)
        result = self.second_linear(result)
        result = self.relu(result)

        result = self.dropout(result)
        result = torch.transpose(result, dim0=1, dim1=2)
        result = self.third_batch_norm(result)
        result = torch.transpose(result, dim0=1, dim1=2)
        result = self.third_linear(result)
        result = self.sigmoid(result)

        return result

    def make_src_mask(self, src, parent_sequence):
            src_mask = (src == self.src_pad_idx) if parent_sequence else (src == self.src_pad_pattern).all(dim=2)
            return src_mask.to(self.device)
