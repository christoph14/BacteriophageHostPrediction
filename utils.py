import numpy as np
import pandas as pd

import RBP_functions as RBP_f


def load_data(data_path='RBP_database.csv'):
    column_names = [
        'id',
        'tax_id',
        'embl_id',
        'name',
        'protein_sequence',
        'dna_sequence',
        'org',
        'host',
        'protein_length',
        'protein_200'
    ]
    fibers = pd.read_csv(
        data_path,
        delimiter=',',
        header=0,
        names=column_names,
    )

    classes = []
    for i, item in enumerate(fibers.iloc[:, 7]):
        if 'Staphylococcus aureus' in item:
            classes.append(0)
        elif 'Klebsiella pneumoniae' in item:
            classes.append(1)
        elif 'Acinetobacter baumannii' in item:
            classes.append(2)
        elif 'Pseudomonas aeruginosa' in item:
            classes.append(3)
        elif 'Escherichia coli' in item:
            classes.append(4)
        elif 'Salmonella enterica' in item:
            classes.append(5)
        elif 'Clostridium difficile' in item:
            classes.append(6)
        else:
            classes.append(7)
    fibers['class'] = np.asarray(classes)

    return fibers


def load_alignment(data_path='RBP_alignmentscores.txt'):
    alignment_scores = np.loadtxt(data_path)
    np.fill_diagonal(alignment_scores, 0)
    return alignment_scores


def preprocess_data(data, alignment_scores):
    # Remove entries with class 7
    fibers = data.copy()
    fibers = fibers.loc[fibers['class'] != 7]
    fibers.reset_index(drop=True, inplace=True)

    # Remove duplicates from fibers, dummies and alignment_scores
    _, unique_indices = np.unique(fibers['protein_sequence'], return_index=True)
    remove_indices = np.setdiff1d(np.arange(len(fibers)), unique_indices)
    fibers = fibers.drop(index=remove_indices)
    fibers.reset_index(drop=True, inplace=True)
    alignment_scores = np.delete(alignment_scores, remove_indices, 0)
    alignment_scores = np.delete(alignment_scores, remove_indices, 1)

    return fibers, alignment_scores


def construct_protein_embeddings(rbp_database):
    import torch

    protein_list = list(rbp_database['protein_sequence'])

    # Load ESM-2 model
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [(f'protein{i+1}', protein) for i, protein in enumerate(protein_list)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    # Look at the unsupervised self-attention map contact predictions
    # import matplotlib.pyplot as plt
    # for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    #     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    #     plt.title(seq)
    #     plt.show()
    embedding_db = pd.DataFrame(sequence_representations, dtype=float)
    return embedding_db


def construct_features(fibers):
    dna_list = list(fibers.iloc[:, 5])
    dna_feats = RBP_f.dna_features(dna_list)

    protein_list = list(fibers.iloc[:, 4])
    protein_feats = RBP_f.protein_features(protein_list)

    # protein features: CTD & Z-scale
    extra_feats = np.zeros((len(fibers), 47))

    for i, item in enumerate(protein_list):
        feature_lst = []
        feature_lst += RBP_f.CTDC(item)
        feature_lst += RBP_f.CTDT(item)
        feature_lst += RBP_f.zscale(item)
        extra_feats[i, :] = feature_lst

    extra_feats_df = pd.DataFrame(extra_feats, columns=[
        'CTDC1', 'CTDC2', 'CTDC3', 'CTDT1', 'CTDT2', 'CTDT3', 'CTDT4', 'CTDT5', 'CTDT6', 'CTDT7', 'CTDT8', 'CTDT9',
        'CTDT10', 'CTDT11', 'CTDT12', 'CTDT13', 'CTDT14', 'CTDT15', 'CTDT16', 'CTDT17', 'CTDT18', 'CTDT19', 'CTDT20',
        'CTDT21', 'CTDT22', 'CTDT23', 'CTDT24', 'CTDT25', 'CTDT26', 'CTDT27', 'CTDT28', 'CTDT29', 'CTDT30', 'CTDT31',
        'CTDT32', 'CTDT33', 'CTDT34', 'CTDT35', 'CTDT36', 'CTDT37', 'CTDT38', 'CTDT39', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'
    ])

    features = pd.concat([dna_feats, protein_feats, extra_feats_df], axis=1)
    return features
