import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cospar.pl import custom_hierachical_ordering
from scipy.stats import linregress

rng = np.random.default_rng()

############################

# Concise utility functions

############################


def estimate_exponent(X, xmin=None):
    X = np.array(X)
    if xmin is None:
        xmin = np.min(X)
    X_new = X[X >= xmin]
    return 1 + len(X_new) / np.sum(np.log(X_new / xmin))


def shuffle_matrix(matrix_0, run=10):
    """
    requires a numpy array
    """
    matrix = matrix_0.copy()
    sub_size = int(np.max([1, matrix.shape[0] / 1000]))
    for x in range(run):
        y_shuffling = rng.permutation(np.arange(0, matrix.shape[1]))
        sel_x_idx = rng.permutation(np.arange(0, matrix.shape[0]))[:sub_size]
        matrix[sel_x_idx] = matrix[sel_x_idx][:, y_shuffling]
    return matrix


def sub_sample(df, size=1000, replace=True):
    dist = np.array(df["UMI_count"] / np.sum(df["UMI_count"]))
    sel_idx = np.random.choice(np.arange(len(df)), size=size, p=dist, replace=replace)
    return df.iloc[sel_idx].drop_duplicates("allele")


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


def reverse_compliment(seq):
    reverse = np.array(list(seq))[::-1]
    map_seq = {"A": "T", "C": "G", "T": "A", "G": "C"}
    complement = "".join([map_seq[x] for x in reverse])
    return complement
