import numpy as np
import pandas as pd

rng = np.random.default_rng()

############################

# Concise utility functions

############################


def map_dictionary(X1, X2):
    """
    Construct a dictionary mapping elements of X1 to containing elements in X2.

    Parameters
    ----------
    X1 : iterable
        Source elements to map from.
    X2 : iterable
        Target elements to map to. Each element in X2 that contains an element
        from X1 will be used as the mapping target.

    Returns
    -------
    dict
        Dictionary mapping each element in X1 to the first element in X2 that
        contains it.
    """
    dict_tmp = {}
    for x1 in X1:
        for x2 in X2:
            if x1 in x2:
                dict_tmp[x1] = x2
    return dict_tmp


def estimate_exponent(X, xmin=None):
    """
    Estimate the power-law exponent using the maximum likelihood method.

    Parameters
    ----------
    X : array-like
        Data points to estimate exponent from.
    xmin : float, optional
        Minimum value to consider. If None, uses the minimum of X.

    Returns
    -------
    float
        Estimated power-law exponent.
    """
    X = np.array(X)
    if xmin is None:
        xmin = np.min(X)
    X_new = X[X >= xmin]
    return 1 + len(X_new) / np.sum(np.log(X_new / xmin))


def shuffle_matrix(matrix_0, run=10):
    """
    Shuffle a matrix to disrupt correlation structure while preserving marginal distributions.

    Parameters
    ----------
    matrix_0 : ndarray
        Input numpy array to shuffle.
    run : int, optional
        Number of shuffling iterations to perform. Default is 10.

    Returns
    -------
    ndarray
        Shuffled matrix with preserved dimensions.
    """
    matrix = matrix_0.copy()
    sub_size = int(np.max([1, matrix.shape[0] / 1000]))
    for x in range(run):
        y_shuffling = rng.permutation(np.arange(0, matrix.shape[1]))
        sel_x_idx = rng.permutation(np.arange(0, matrix.shape[0]))[:sub_size]
        matrix[sel_x_idx] = matrix[sel_x_idx][:, y_shuffling]
    return matrix


def sub_sample(df, size=1000, replace=True):
    """
    Subsample a dataframe based on UMI counts.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'UMI_count' column.
    size : int, optional
        Number of samples to draw. Default is 1000.
    replace : bool, optional
        Whether to sample with replacement. Default is True.

    Returns
    -------
    pd.DataFrame
        Subsampled dataframe with duplicates removed based on 'allele' column.
    """
    dist = np.array(df["UMI_count"] / np.sum(df["UMI_count"]))
    sel_idx = np.random.choice(np.arange(len(df)), size=size, p=dist, replace=replace)
    return df.iloc[sel_idx].drop_duplicates("allele")


def onehot(input_dict):
    """
    Convert a classification dictionary to one-hot encoding.

    Parameters
    ----------
    input_dict : dict
        Dictionary mapping keys to class labels.

    Returns
    -------
    dict
        Dictionary mapping each key to its one-hot encoded representation as a list.
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
    """
    Compute the reverse complement of a DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence string containing A, C, G, T, N.

    Returns
    -------
    str
        Reverse complement of the input sequence.
    """
    reverse = np.array(list(seq))[::-1]
    map_seq = {"A": "T", "C": "G", "T": "A", "G": "C", "N": "N"}
    complement = "".join([map_seq[x] for x in reverse])
    return complement


def order_sample_by_fates(sample_list):
    """
    Order samples by hematopoietic lineage progression.

    Parameters
    ----------
    sample_list : list
        List of sample names to order.

    Returns
    -------
    ndarray
        Ordered array of sample names sorted by lineage progression.

    Raises
    ------
    ValueError
        If a sample cannot be matched to any known fate type.
    """
    sample_order_0 = [
        "LT-HSC",
        "ST-HSC",
        "HSC",
        "MPP2",
        "MPP3",
        "MPP3-4",
        "LK",
        "MEG",
        "ERY",
        "GR",
        "MONO",
        "B",
    ]
    sample_order = dict(zip(sample_order_0, np.arange(len(sample_order_0))))
    order_list = []
    keys = np.array(list(sample_order.keys()))

    for x in sample_list:
        flag = False
        for y in keys:
            if y in x.upper():
                order_list.append(sample_order[y])
                flag = True
                break
        if not flag:
            raise ValueError(f"{x} not found in sample key {keys}")
    df = pd.DataFrame({"sample": sample_list, "lineage_order": order_list})
    df["mouse"] = df["sample"].apply(lambda x: x.split("-")[0])
    return df.sort_values(["mouse", "lineage_order"], ascending=True)["sample"].values
