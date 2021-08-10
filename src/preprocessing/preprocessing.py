""" Module for custom dataset preprocessing """
import pandas as pd

from path_helper import get_project_root


def run_preprocessing(run_from_scratch, df_dataset):
    """ Custom dataset preprocessing if run_from_scratch=True """
    if run_from_scratch:
        print("\nPreprocessing dataset:")
        df_selected = select_subpart_of_genome(df_dataset)
        df_dataset_numeric = sequence_to_numeric(df_selected)
        return df_dataset_numeric
    else:
        df_preprocessed = pd.read_csv(
            str(get_project_root()) + "/data/preprocessed/preprocessed.csv")
        df_preprocessed["parent"] = df_preprocessed["parent"].apply(eval)
        df_preprocessed["child"] = df_preprocessed["child"].apply(eval)
        return df_preprocessed


def select_subpart_of_genome(df_dataset):
    """ Select subpart of each genome sequence """
    print(" - Select subpart of genome")
    # TODO: adapt (which section to select?)
    # don't start at the beginning (sequences are aligned and padded (-))
    start = 100
    end = 199
    df_selected = df_dataset.apply(
        lambda x: [select_subpart(x.parent, start, end), select_subpart(x.child, start, end)], axis=1,
        result_type='expand')
    df_selected.columns = ["parent", "child"]
    return df_selected


def select_subpart(sequence, beginning, end):
    """ Select subpart of sequence """
    return sequence[beginning:end]


def sequence_to_numeric(df_dataset):
    """ Convert sequences (string) to numeric representation (each triplet = one id) """
    print(" - Transform to numeric input")
    list_of_chars = ["A", "T", "G", "C", "N", "-"]
    vocab = create_vocab(list_of_chars)

    df_converted = df_dataset.apply(lambda x: [transform(x.parent, vocab), transform(x.child, vocab)], axis=1,
                                    result_type='expand')
    df_converted.columns = ["parent", "child"]
    df_converted.to_csv("./data/preprocessed/preprocessed.csv", index=False)
    return df_converted


def create_vocab(list_of_chars):
    """ Create triplet-vocab from list of char-vocab """
    triplet_dictionary = {}
    id = 0
    for index_1, char_1 in enumerate(list_of_chars):
        for index_2, char_2 in enumerate(list_of_chars):
            for index_3, char_3 in enumerate(list_of_chars):
                # print("id {}: {}: {} = {:5.2f}%".format(index, key, codon_freq[key], codon_freq[key] * 100 / n))
                triplet_dictionary[char_1 + char_2 + char_3] = id
                id = id + 1
    return triplet_dictionary


def transform(seq, vocab):
    """ transform sequence to numeric representation """
    if len(seq) % 3 == 0:
        codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
        transformed = []
        for codon in codons:
            try:
                transformed.append(vocab[codon])
            except:
                # print(seq)
                transformed.append(-1)
        return transformed
    else:
        print("Error: Sequence is not divisible by 3 -> could not be splitted into triplets")

# def count_triplets(seq):
#    if len(seq) % 3 == 0:
#        #split sequence into codons of three
#        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
#        #create Counter dictionary for it
#        codon_freq = Counter(codons)
#        #determine number of codons, should be len(seq) // 3
#        n = sum(codon_freq.values())
#        #print out all entries in an appealing form
#        triplet_dictionary = {}
#        for index, key in enumerate(sorted(codon_freq)):
#            print("id {}: {}: {} = {:5.2f}%".format(index, key, codon_freq[key], codon_freq[key] * 100 / n))
#            triplet_dictionary[key] = index
#        #or just the dictionary
#        print(triplet_dictionary)
#        print(triplet_dictionary["AGC"])
#        print(len(triplet_dictionary))
#
#        transformed = []
#        for codon in codons:
#            transformed.append(triplet_dictionary[codon])
#
#        print(transformed)
#    else:
#        print("Error: Sequence could not be splitted into triplets")
