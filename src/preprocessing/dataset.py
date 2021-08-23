import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class Vocabulary:
    def __init__(self):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_codons(sequence):
        codons = []
        for i in range(0, len(sequence), 3):
            amino_acid = sequence[i:i + 3]
            if amino_acid == "---":
                codons.append("<PAD>")
            elif "-" in amino_acid or "N" in amino_acid or len(amino_acid) < 3:
                codons.append("<UNK>")
            else:
                codons.append(amino_acid)
        return codons

    def build_vocabulary(self, sequence_list):
        index = 4
        for sequence in sequence_list:
            for codon in self.tokenizer_codons(sequence):
                if codon not in self.stoi:
                    self.stoi[codon] = index
                    self.itos[index] = codon
                    index += 1

    def numericalize(self, sequence):
        return [self.stoi[token] for token in self.tokenizer_codons(sequence)]


class CustomGISAIDDataset(Dataset):
    def __init__(self, dataset_file, test_set_size=0.05, train=True, strain_begin=101, strain_end=200):
        # Load the data and restrict to RNA strains
        print("Loading data and cutting to strains...")
        self.df_dataset = pd.read_csv(dataset_file)
        self.df_dataset["parent"] = self.df_dataset["parent"].str[strain_begin:strain_end]
        self.df_dataset["child"] = self.df_dataset["child"].str[strain_begin:strain_end]

        self.strain_begin = strain_begin
        self.strain_end = strain_end
        self.train = train

        # Initialize vocabulary and build vocabulary
        print("Building vocabulary...")
        self.parent_vocab = Vocabulary()
        self.parent_vocab.build_vocabulary(self.df_dataset["parent"].tolist())
        self.child_vocab = Vocabulary()
        self.child_vocab.build_vocabulary(self.df_dataset["child"].tolist())

        # Initialize vocabulary and build vocabulary
        print("Train test split...")
        self.df_train, self.df_test = train_test_split(self.df_dataset, test_size=test_set_size)
        chosen_dataset = self.df_train if train else self.df_test
        self.parent = chosen_dataset["parent"]
        self.child = chosen_dataset["child"]

        # TODO: When doing long training runs one might consider to numericalize the sequences only once here

    def __getitem__(self, index):
        parent = self.parent.iloc[index]
        child = self.child.iloc[index]

        numericalized_parent = [self.parent_vocab.stoi["<SOS>"]]
        numericalized_parent += self.parent_vocab.numericalize(parent)
        numericalized_parent.append(self.parent_vocab.stoi["<EOS>"])

        numericalized_child = [self.child_vocab.stoi["<SOS>"]]
        numericalized_child += self.child_vocab.numericalize(child)
        numericalized_child.append(self.child_vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_parent), torch.tensor(numericalized_child)

    def __len__(self):
        return len(self.df_train) if self.train else len(self.df_test)


def get_loader(dataset_file, test_set_size=0.05, batch_size=32, train=True, num_workers=4):
    dataset = CustomGISAIDDataset(dataset_file, test_set_size=test_set_size, train=train)
    return dataset, DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
