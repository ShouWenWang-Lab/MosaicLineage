import os
import time
from copy import deepcopy
from pathlib import Path

import cospar as cs
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat

rng = np.random.default_rng()


def generate_adata_v0(X_clone, state_info=None):
    adata_orig = sc.AnnData(X_clone)
    adata_orig.obs["time_info"] = ["0"] * X_clone.shape[0]
    adata_orig.obsm["X_clone"] = X_clone
    adata_orig.uns["data_des"] = ["hi"]
    if state_info is None:
        adata_orig.obs["state_info"] = pd.Categorical(
            np.arange(X_clone.shape[0]).astype(str)
        )
    else:
        adata_orig.obs["state_info"] = pd.Categorical(state_info)
    return adata_orig


def generate_adata(df_data, use_np_array=False):
    all_mutation = []
    for xx in df_data["allele"]:
        all_mutation += list(xx.split(","))
    all_mutation = np.array(list(set(all_mutation)))

    if use_np_array:
        print("Use np.array")
        X_clone = np.zeros((len(df_data), len(all_mutation)))
        for i, xx in enumerate(df_data["allele"]):
            for yy in list(xx.split(",")):
                idx = np.nonzero(all_mutation == yy)[0]
                X_clone[i, idx] = df_data.iloc[i][
                    "obs_UMI_count"
                ]  # This keeps the count information, and works better
                # X_clone[i,idx]=1
    else:
        print("Use sparse matrix")
        X_clone_row = []
        X_clone_col = []
        X_clone_val = []
        for i, xx in enumerate(df_data["allele"]):
            for yy in list(xx.split(",")):
                idx = np.nonzero(all_mutation == yy)[0]
                if len(idx) > 0:
                    value_temp = np.array(
                        df_data.iloc[i]["obs_UMI_count"]
                    )  # or, value_temp=np.ones(len(idx))
                    X_clone_row.append(i)
                    X_clone_col.append(idx[0])
                    X_clone_val.append(value_temp)
                    # X_clone_COO += [(i, idx_j, value_temp[j]) for j, idx_j in enumerate(idx)]
        X_clone = ssp.coo_matrix(
            (X_clone_val, (X_clone_row, X_clone_col)),
            shape=(len(df_data), len(all_mutation)),
        )

    X_clone = ssp.csr_matrix(X_clone)
    adata_orig = sc.AnnData(X_clone)
    adata_orig.var_names = all_mutation
    adata_orig.obs["time_info"] = ["0"] * X_clone.shape[0]
    adata_orig.obsm["X_clone"] = X_clone
    adata_orig.uns["data_des"] = ["hi"]
    adata_orig.obs["allele"] = np.array(df_data["allele"])
    if "expected_frequency" in df_data.keys():
        adata_orig.obs["expected_frequency"] = np.array(df_data["expected_frequency"])
    adata_orig.obs["obs_UMI_count"] = np.array(df_data["obs_UMI_count"])
    adata_orig.obs["sample"] = np.array(df_data["sample"])
    adata_orig.obs["cell_id"] = [
        f"{xx[-1]}-{j}" for j, xx in enumerate(df_data["mouse"])
    ]
    adata_orig.obs["mouse"] = np.array(df_data["mouse"])
    clone_idx = (X_clone > 0).sum(
        0
    ).A.flatten() > 1  # select clones observed in more than one cells.
    adata_orig.uns["multicell_clones"] = clone_idx
    adata_orig.obs["cells_from_multicell_clone"] = (X_clone[:, clone_idx] > 0).sum(
        1
    ).A.flatten() > 0
    adata_orig.var_names = all_mutation
    return adata_orig


def load_allele_info(data_path):
    pooled_data = loadmat(os.path.join(data_path, "allele_annotation.mat"))
    allele_freqs = pooled_data["allele_freqs"].flatten()
    alleles = [xx[0][0] for xx in pooled_data["AlleleAnnotation"]]
    return pd.DataFrame({"allele": alleles, "UMI_count": allele_freqs})


def query_allele_frequencies(df_reference, df_target):
    return df_reference.merge(df_target, on="allele", how="right").fillna(0)


def onehot(input_dict):
    """
    The input dict provides classification for all samples
    It returns the corresponding onehot encoding for each sample
    """
    output_dict = {}
    aval = set(input_dict.values())

    mapping_dict = {}
    for j, x in enumerate(aval):
        mapping_dict[x] = j

    for key, value in input_dict.items():
        temp = np.zeros(len(aval))
        temp[mapping_dict[value]] = 1
        output_dict[key] = list(temp)

    return output_dict


def tree_reconstruction_accuracy(
    parent_map, node_mapping, origin_score, weight_factor=1, plot=False
):
    """
    origin_score:
        A dictionary to map leaf nodes to a value. The key is taken from the letters before '-' of a node name.
        We recommend to symmetric value like -1 and 1, instead of 0 and 1. Otherwise, our weighting scheme is not working
    """

    node_score = {}
    for key, value in node_mapping.items():
        temp_scores = [origin_score[x] for x in value]
        node_score[key] = np.mean(temp_scores, axis=0) / weight_factor ** len(
            temp_scores
        )

    score_pairs = []
    child_nodes = []
    parent_nodes = []
    for key, value in parent_map.items():
        child_nodes.append(key)
        parent_nodes.append(value)
    df = pd.DataFrame({"child": child_nodes, "parent": parent_nodes}).set_index(
        "parent"
    )
    for unique_parent in list(set(parent_nodes)):
        temp_pair = [node_score[xx] for xx in df.loc[unique_parent]["child"].values]
        score_pairs.append(temp_pair)
    score_pairs = np.array(score_pairs)
    score_pairs_flatten = [
        score_pairs[:, 0, :].flatten(),
        score_pairs[:, 1, :].flatten(),
    ]

    if plot:
        ax = sns.scatterplot(x=score_pairs_flatten[0], y=score_pairs_flatten[1])
        ax.set_xlabel("Membership score for node a")
        ax.set_ylabel("Membership score for node b")
        ax.set_title(f"Decay factor={weight_factor:.2f}")

    corr = np.corrcoef(score_pairs_flatten[0], score_pairs_flatten[1])[0, 1]
    return corr, score_pairs_flatten


def shuffle_matrix(matrix_0, run=10):
    matrix = matrix_0.copy()
    sub_size = int(np.max([1, matrix.shape[0] / 1000]))
    for x in range(run):
        y_shuffling = rng.permutation(np.arange(0, matrix.shape[1]))
        sel_x_idx = rng.permutation(np.arange(0, matrix.shape[0]))[:sub_size]
        matrix[sel_x_idx] = matrix[sel_x_idx][:, y_shuffling]
    return matrix


def evaluate_coupling_matrix(
    coupling, fate_names, origin_score, decay_factor=1, plot=False
):
    print("Use updated pipeline")
    size = len(fate_names)
    leaf_score = np.array([origin_score[xx] for xx in fate_names])
    weight_function = np.exp(-decay_factor * np.arange(size))
    map_score = np.zeros(size)
    coupling_1 = coupling.copy()
    for i in range(size):
        coupling_1[
            i, i
        ] = 0  # reset the diagonal term to be 0 so that it will not contribute to the final score
        ref_score = leaf_score[i]
        idx = np.argsort(coupling_1[i])[::-1]
        temp = 0
        for j, new_id in enumerate(idx):
            temp += np.sum(leaf_score[new_id] * ref_score) * weight_function[j]
        map_score[i] = temp

    if plot:
        ax = sns.histplot(x=map_score)
        ax.set_xlabel("Lineage coupling accuracy")
        ax.set_ylabel("Count")
        # ax.set_title(f'Mean: {np.mean(map_score):.2f}')
        ax.set_title(f"Fraction (>1.3): {np.mean(map_score>1.3):.2f}")
    return map_score
