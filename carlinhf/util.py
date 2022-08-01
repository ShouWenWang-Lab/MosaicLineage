import numpy as np

rng = np.random.default_rng()

############################

# Concise utility functions

############################


def map_dictionary(X1, X2):
    """
    construct a dictionary from x1 in X1 to x2 in X2, if x2 contains x1
    """
    dict_tmp = {}
    for x1 in X1:
        for x2 in X2:
            if x1 in x2:
                dict_tmp[x1] = x2
    return dict_tmp


def extract_first_sample_from_a_nesting_list(SampleList):
    """
    For a nesting list like ['a',['b','c'],['d','e','f']],
    it will return the first in each element, i.e, ['a','b','d']
    """
    selected_fates = []
    for x in SampleList:
        if type(x) is list:
            selected_fates.append(x[0])
        else:
            selected_fates.append(x)
    return selected_fates


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
