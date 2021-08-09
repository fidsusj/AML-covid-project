""" Module for creating the dataset """
import glob
import subprocess
from time import strptime

import pandas as pd
from Bio import SeqIO
from dendropy import Tree
from dendropy.calculate.treemeasure import PatristicDistanceMatrix

from src.path_helper import get_project_root


def run_create_dataset(run_from_scratch):
    """ Recreate dataset from fasta files if run_from_scratch=True """
    if run_from_scratch:
        print("\nRecreate dataset:")
        combine_separate_input_files(str(get_project_root()) + "/data/raw/separate_input_files")
        construct_phylogenetic_tree()
        df_parent_child = calculate_related_sequences_based_on_phylogenetic_tree()
        df_dataset = fill_parent_child_combinations_with_genome_data(df_parent_child)
        return df_dataset
    else:
        df_dataset = pd.read_csv(
            str(get_project_root()) + "/data/dataset/final.csv")
        return df_dataset


def combine_separate_input_files(path_to_data_from_gisaid):
    print(" - Combine separate input file into one")
    all_metadata_files = glob.glob(path_to_data_from_gisaid + "/*.tsv")
    list_df_separate_inputs = []
    for filename in all_metadata_files:
        df = pd.read_csv(filename, index_col=None, header=0, sep='\t')
        list_df_separate_inputs.append(df)
    df_combined = pd.concat(list_df_separate_inputs, axis=0, ignore_index=True)
    df_combined.to_csv("./data/raw/example_metadata.tsv", index=False)
    print("     metadata done")
    exit_code = subprocess.call(str(get_project_root()) + '/create_dataset/combine_fasta_input_files.sh')
    assert exit_code == 0
    print("     fasta done")


def construct_phylogenetic_tree():
    """ Data preprocessing (alignment, filtering) with ncov routine and phylogenetic tree generation """
    print(" - Create phylogenetic tree with ncov")
    exit_code = subprocess.call(str(get_project_root()) + '/create_dataset/run_ncov.sh')
    assert exit_code == 0


def calculate_related_sequences_based_on_phylogenetic_tree():
    """ Get parent child combinations from phylogenetic tree """
    print(" - Calculate related sequences")
    tree = Tree.get(path=str(get_project_root()) + "/data/dataset/tree.nwk", schema="newick")
    patristic_distance_matrix = PatristicDistanceMatrix(tree)
    # k nächste finden (k = 1 für den Anfang)
    best_pairs = []
    for i, taxa_1 in enumerate(tree.taxon_namespace):
        best_pair = None
        closest_distance = 10000000000
        for taxa_2 in tree.taxon_namespace[i + 1:]:
            #print("Distance between '%s' and '%s': %s" % (
            #    taxa_1.label, taxa_2.label, patristic_distance_matrix(taxa_1, taxa_2)))
            if (patristic_distance_matrix(taxa_1, taxa_2) < closest_distance):
                closest_distance = patristic_distance_matrix(taxa_1, taxa_2)
                best_pair = [taxa_1.label.replace(" ", "_"), taxa_2.label.replace(" ", "_")]
        best_pairs.append(best_pair)

    best_pairs = [x for x in best_pairs if x is not None]
    #print(best_pairs)

    df_metadata = pd.read_csv(str(get_project_root()) + "/data/dataset/sanitized_metadata_focal.tsv", header=0,
                              sep='\t')
    df_metadata.strain.replace(" ", "_", inplace=True)  # is read wrong _ are read as blank -> fix it here

    # Paare nach Datum in parent child sortieren
    # [[parent, child], [parent, child], ...]
    sorted_by_time = []
    for pair in best_pairs:
        #print(pair)
        taxa_1_row_index = df_metadata[df_metadata["strain"] == pair[0]].index[0]
        taxa_2_row_index = df_metadata[df_metadata["strain"] == pair[1]].index[0]
        taxa_1_date = df_metadata.at[taxa_1_row_index, "date"]
        taxa_2_date = df_metadata.at[taxa_2_row_index, "date"]

        taxa_1_date = strptime(taxa_1_date, "%Y-%m-%d")
        taxa_2_date = strptime(taxa_2_date, "%Y-%m-%d")
        if (taxa_1_date > taxa_2_date):
            parent = pair[1]
            child = pair[0]
        else:
            parent = pair[0]
            child = pair[1]
        sorted_by_time.append([parent, child])

    df_dataset_parent_child_strain_names = pd.DataFrame(sorted_by_time, columns=["parent", "child"])
    df_dataset_parent_child_strain_names.to_csv("./data/dataset/df_dataset_parent_child_strain_names.csv", index=False)
    return df_dataset_parent_child_strain_names


def fill_parent_child_combinations_with_genome_data(df_parent_child):
    """ Insert parent child sequences """
    print(" - Fill parent child combinations with genome data")
    sequences = {}
    for seq_record in SeqIO.parse(str(get_project_root()) + "/data/dataset/aligned_focal.fasta", "fasta"):
        sequences[seq_record.id] = seq_record.seq + "-"

    # matching and fill df
    for index, row in df_parent_child.iterrows():
        #print(row['parent'])
        df_parent_child.at[index, 'parent'] = sequences[row['parent']]
        df_parent_child.at[index, 'child'] = sequences[row['child']]

    df_parent_child.to_csv("./data/dataset/final.csv", index=False)
    return df_parent_child
