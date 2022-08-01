import os

import cospar as cs
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

import carlinhf.plotting as plotting

rng = np.random.default_rng()

###########################################################

## downstream allele analysis, tailored for CARLIN dataset
# like frequency, mutation-allele relation etc.

###########################################################


def mutations_per_allele(
    df_input, count_key="UMI_count", save=False, save_path=".", plot=False
):

    mutation_per_allele = []
    for j, x in enumerate(list(df_input["allele"].apply(lambda x: x.split(",")))):
        mutation_per_allele.append(len(x))

    if plot:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax = sns.histplot(x=mutation_per_allele, binwidth=0.5)
        ax.set_xlabel("Mutation number per allele")
        ax.set_ylabel("Count")
        ax.set_title(f"Mean: {np.mean(mutation_per_allele):.1f}")
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        if save:
            fig.savefig(f"{save_path}/mutation_per_allele.pdf")
    return mutation_per_allele


def mutations_per_allele_ins_del(df_input):
    """
    Count the insertion and deletion events. Note that
    some mutations have both insertion and deletion, and we are
    double counting here.
    """

    ins_per_allele = []
    del_per_allele = []
    for j, x in enumerate(list(df_input["allele"].apply(lambda x: x.split(",")))):
        temp_ins = [y for y in x if "ins" in y]
        temp_del = [y for y in x if "del" in y]
        ins_per_allele.append(len(temp_ins))
        del_per_allele.append(len(temp_del))

    return ins_per_allele, del_per_allele


def mutations_length_per_allele_ins_del(df_input):
    """
    Count the insertion and deletion length. Note that
    some mutations have both insertion and deletion, and we are
    double counting here.
    """

    ins_per_allele = []
    del_per_allele = []
    for i, x in enumerate(list(df_input["allele"].apply(lambda x: x.split(",")))):
        temp_del = []
        temp_ins = []
        for y in x:
            if "del" in y:
                temp = y.split("del")[0].split("_")
                del_len = int(temp[1]) - int(temp[0])
                temp_del.append(del_len)

            if "ins" in y:
                temp = y.split("ins")[1]
                temp_ins.append(len(temp))

        ins_per_allele.append(temp_ins)
        del_per_allele.append(temp_del)

    return ins_per_allele, del_per_allele


def mutations_deletion_statistics(df_input):
    """
    Record initial and final deletion position and the corresponding mutant
    """

    del_per_allele = []
    mutation_record = []
    for i, x in enumerate(list(df_input["allele"].apply(lambda x: x.split(",")))):
        for y in x:
            if "del" in y:
                temp = y.split("del")[0].split("_")
                del_len = int(temp[1]) - int(temp[0])
                del_per_allele.append([del_len, temp[0], temp[1]])
                mutation_record.append(y)

    periodic_record = []
    for j, x in enumerate(del_per_allele):
        periodic_record.append(x + [mutation_record[j]])

    my_dict = {}
    my_dict["del_length"] = np.array(periodic_record)[:, 0].astype(int)
    my_dict["del_initial"] = np.array(periodic_record)[:, 1].astype(int)
    my_dict["del_end"] = np.array(periodic_record)[:, 2].astype(int)
    my_dict["mutation"] = np.array(periodic_record)[:, 3]

    ins_length_array = []
    for x in my_dict["mutation"]:
        if "ins" in x:
            ins_len = len(list(x.split("ins")[1]))
            ins_length_array.append(ins_len)
        else:
            ins_length_array.append(0)

    my_dict["ins_length"] = ins_length_array
    df = pd.DataFrame(my_dict)

    return df


def mutation_frequency(df_input, save=False, save_path=".", plot=True):
    """
    df_Input: should ahve 'allele' and 'UMI_count'
    df_mutation_ne: should have 'mutation' and 'UMI_count'
    """
    mut_list = []
    UMI_count = []
    for j, x in enumerate(list(df_input["allele"].apply(lambda x: x.split(",")))):
        mut_list += x
        UMI_count += list(np.repeat(df_input["UMI_count"].iloc[j], len(x)))
    df_mutation = pd.DataFrame({"mutation": mut_list, "UMI_count": UMI_count})
    df_mutation_new = df_mutation.groupby("mutation", as_index=False).agg(
        {"UMI_count": "sum"}
    )

    if plot:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax = sns.histplot(data=df_mutation_new, x="UMI_count", log_scale=True)
        plt.yscale("log")
        singleton_fraction = np.sum(df_mutation_new["UMI_count"] == 1) / np.sum(
            df_mutation_new["UMI_count"] > 0
        )
        ax.set_title(f"1 frac. ={singleton_fraction:.3f}")
        if save:
            fig.savefig(f"{save_path}/mutation_frequency.pdf")

    return df_mutation_new


def effective_allele_number(UMI_counts):
    x = np.array(UMI_counts) / np.sum(UMI_counts)
    entropy = -np.sum(np.log2(x) * x)
    return 2**entropy


def effective_allele_over_cell_fraction(df_input, editing_efficiency: float = None):

    if editing_efficiency is not None:
        UMI_count_temp = np.array(df_input["UMI_count"]).astype(float)
        null_index = np.nonzero(np.array(df_input["allele"] == "[]"))[0][0]
        new_null_count = (
            (np.sum(UMI_count_temp) - UMI_count_temp[null_index])
            / editing_efficiency
            * (1 - editing_efficiency)
        )
        UMI_count_temp[null_index] = new_null_count
    else:
        UMI_count_temp = np.array(df_input["UMI_count"]).astype(float)

    df_input_1 = df_input.copy()
    df_input_1["UMI_count"] = UMI_count_temp
    df_sort = df_input_1.sort_values("UMI_count", ascending=False)
    UMI_counts = df_sort["UMI_count"]
    UMI_counts_no_null = df_sort[df_sort.allele != "[]"]["UMI_count"]
    tot_counts = np.sum(UMI_counts)
    effective_allele_N = np.zeros(len(df_sort) - 1)
    cell_fraction = np.zeros(len(df_sort) - 1)
    null_idx = np.nonzero(np.array(df_sort.allele == "[]"))[0]
    for j in tqdm(range(len(df_sort) - 1)):
        if j < null_idx:
            temp = effective_allele_number(UMI_counts_no_null[j:])
        else:
            temp = effective_allele_number(UMI_counts_no_null[j + 1 :])
        effective_allele_N[j] = temp
        cell_fraction[j] = np.sum(UMI_counts[:j])
    cell_fraction = 1 - cell_fraction / tot_counts
    df_data = pd.DataFrame(
        {"cell_fraction": cell_fraction, "effective_allele_N": effective_allele_N}
    )
    return df_data


def generate_FrequencyCounts(df_raw, save_dir=None):
    """
    df_raw is a pandas object, with
    'allele','UMI_count'

    A speeded up version
    """
    df_input = df_raw.reset_index()
    df_new = df_input.groupby("allele", as_index=False).agg({"UMI_count": "sum"})

    UMI_count = list(df_new["UMI_count"])
    unique_count = np.sort(list(set(UMI_count))).astype(int)
    count_frequency = np.zeros(len(unique_count), dtype=int)
    for j, x in enumerate(unique_count):
        count_frequency[j] = np.sum(UMI_count == x)

    df_count = pd.DataFrame(
        {"UMI_count": unique_count, "Frequency": count_frequency}
    ).set_index("UMI_count")
    if save_dir is not None:
        df_count.to_csv(f"{save_dir}/FrequencyCounts.csv", header=None)
    return df_count


def subsample_allele_frequency_count(df_input, sp_fraction, out_dir):
    """
    df_input: pd object from FrequencyCounts.csv
    """

    allele_frequency = []
    for j in range(len(df_input)):
        allele_frequency += list(
            np.zeros(df_input.iloc[j].Count) + df_input.iloc[j].Frequency
        )

    allele_frequency_sp = []
    for x in allele_frequency:
        y = rng.binomial(x, sp_fraction)
        allele_frequency_sp.append(y)

    unique_frequency = np.array(list(set(allele_frequency_sp)))
    allele_frequency_sp = np.array(allele_frequency_sp)
    freq_array = np.zeros(len(unique_frequency), dtype=int)
    count_array = np.zeros(len(unique_frequency), dtype=int)
    unique_frequency = np.sort(unique_frequency)
    for j, x in enumerate(unique_frequency):
        freq_array[j] = x
        count = np.sum(allele_frequency_sp == x)
        count_array[j] = count

    df_sp = pd.DataFrame(
        {"Frequency": freq_array[1:], "Count": count_array[1:]}
    ).set_index("Frequency")
    ax = sns.scatterplot(data=df_input, x="Frequency", y="Count", label="Original")
    ax = sns.scatterplot(
        data=df_sp, x="Frequency", y="Count", label=f"Sample:{sp_fraction:.1f}", ax=ax
    )
    plt.xscale("log")
    plt.show()
    # plt.yscale('log')

    data_path = f"{out_dir}/sp_{sp_fraction:.1f}"
    os.makedirs(data_path, exist_ok=True)
    df_sp.to_csv(f"{data_path}/FrequencyCounts.csv", header=None)


def subsample_allele_freq_histogram(
    df, sample_key="sample_key", sample_fraction=0.1, plot=True
):
    """
    Sub-sample a data frame according to the allele frequency.
    df should be the allele dataframe with corresponding UMI_count information
    """

    df["allele_frequency"] = df["UMI_count"] / df["UMI_count"].sum()
    sel_idx = np.random.choice(
        np.arange(len(df)),
        size=int(sample_fraction * len(df)),
        p=df["allele_frequency"].to_numpy(),
        replace=True,
    )
    df_new = df.iloc[sel_idx]
    df_new["UMI_count"] = 1  # df["UMI_count"].sum() / len(df_new)
    df_new = df_new.groupby("allele", as_index=False).agg({"UMI_count": "sum"})

    singleton_ratio = np.sum(df_new["UMI_count"] <= df_new["UMI_count"].min()) / len(
        df_new
    )

    if plot:
        print(f"Singleton ratio: {singleton_ratio}")

        x_var, y_var = plotting.plot_loghist(list(df_new["UMI_count"]), cutoff_y=3)
        plt.tight_layout()
        plt.xlabel("Occurence # per allele (UMI count)")
        plt.ylabel("Histogram")
        plt.title(f"Singleton ratio: {singleton_ratio:.2f}")
        plt.savefig(f"figure/{sample_key}/subsampled_allele_frequency_distribution.pdf")

    return df_new, singleton_ratio


def subsample_singleton_fraction(df, sample_key, sample_fraction_array, plot=True):
    """
    Check the effect of sub-sampling on singleton fraction
    """

    singleton_ratio_orig = np.sum(df["UMI_count"] == 1) / len(df)

    singleton_ratio_array = np.zeros(len(sample_fraction_array))
    allele_fraction_array = np.zeros(len(sample_fraction_array))
    for j, x in enumerate(sample_fraction_array):
        (df_new, singleton_ratio_array[j],) = subsample_allele_freq_histogram(
            df, sample_key, sample_fraction=x, plot=False
        )
        allele_fraction_array[j] = len(df_new) / len(df)

    if plot:
        fig, ax = plt.subplots()
        plt.plot(sample_fraction_array, singleton_ratio_array)
        # plt.plot(x_range, [singleton_ratio_orig, singleton_ratio_orig])
        plt.xlabel("Subsample ratio")
        plt.ylabel("Singleton ratio")
        plt.xscale("log")
        plt.savefig(f"figure/{sample_key}/subsample_singleton_fraction.pdf")

        fig, ax = plt.subplots()
        plt.plot(sample_fraction_array, allele_fraction_array)
        plt.xlabel("Subsample ratio")
        plt.ylabel("Allele fraction")
        plt.xscale("log")
        plt.savefig(f"figure/{sample_key}/subsample_allele_fraction.pdf")

    result = {}
    allele_norm_factor = allele_fraction_array[np.array(sample_fraction_array) == 1][0]
    singleton_norm_factor = singleton_ratio_array[np.array(sample_fraction_array) == 1][
        0
    ]
    result["singleton_fraction"] = (
        singleton_ratio_array * singleton_ratio_orig / singleton_norm_factor
    )
    result["allele_fraction"] = allele_fraction_array / allele_norm_factor
    return result


def query_allele_frequencies(df_reference, df_target):
    return df_reference.merge(df_target, on="allele", how="right").fillna(0)


def correct_null_allele_frequency(df_input, editing_efficiency=0.3):
    """
    Correct the allele frequency of un-edited alleles based on known
    editing efficiency.

    This is used as our Cas9 mouse has higher than expected editing efficiency
    """
    UMI_count_temp = np.array(df_input["UMI_count"]).astype(float)
    null_index = np.nonzero(np.array(df_input["allele"] == "[]"))[0][0]
    new_null_count = (
        (np.sum(UMI_count_temp) - UMI_count_temp[null_index])
        / editing_efficiency
        * (1 - editing_efficiency)
    )
    UMI_count_temp[null_index] = int(new_null_count)
    df_output = df_input.copy()
    df_output["UMI_count"] = UMI_count_temp.astype(int)
    return df_output


def check_allele_frequency_prediction(
    df,
    UMI_cutoff=30,
    mutation_N_cutoff=1,
    markersize=25,
    df_mutation=None,
    norm_factor=None,
):
    """
    Based on the observed mutation frequency, estimate the
    observed allele frequency
    """
    if df_mutation is None:
        df_mutation = mutation_frequency(df, plot=False)

    df_mutation = df_mutation.set_index("mutation")
    if norm_factor is None:
        norm_factor = df_mutation["UMI_count"].sum()
    df_mutation["Frequency"] = df_mutation["UMI_count"].apply(lambda x: x / norm_factor)

    df["mutation_N"] = df["allele"].apply(lambda x: len(x.split(",")))
    df_test = df[(df.UMI_count > UMI_cutoff) & (df["mutation_N"] > mutation_N_cutoff)]
    tot_observations = np.sum(df["UMI_count"] * df["mutation_N"])

    predicted_frequency = []
    for allele_tmp in df_test["allele"]:
        freq = 1
        for x in allele_tmp.split(","):
            freq = freq * df_mutation.loc[x]["Frequency"]
        freq = freq * tot_observations
        predicted_frequency.append(freq)
    df_test["Predicted_Freq"] = predicted_frequency

    fig, ax = plt.subplots()
    df_test = df_test.sort_values("UMI_count", ascending=True)
    ax = sns.scatterplot(
        data=df_test,
        x="UMI_count",
        y="Predicted_Freq",
        s=markersize,
        alpha=1,
        edgecolor="k",
    )
    plt.yscale("log")
    plt.xscale("log")
    return df_test


#################

## tree and coupling analysis

#################


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


def get_fate_count_coupling(X_clone):
    """
    X_clone:
        should be a coarse-grained X_clone: cell_type by clones
        Numpy array
    """
    fate_N = X_clone.shape[0]
    X_count = np.zeros((fate_N, fate_N))
    for i in range(fate_N):
        for j in range(fate_N):
            X_count[i, j] = np.sum((X_clone[i, :] > 0) & (X_clone[j, :] > 0))

    norm_X_count = X_count.copy()
    for i in range(fate_N):
        norm_X_count[i, :] = X_count[i, :] / X_count[i, i]
    return X_count, norm_X_count


def evaluate_coupling_matrix(
    coupling: np.array, fate_names: list, origin_score: dict, decay_factor=1, plot=False
):
    """
    This is designed when we know a dictionary that maps a fate_name to a score.

    origin_score:
        A dictionary to map leaf nodes to a value. The key is taken from the letters before '-' of a node name.
        We recommend to symmetric value like -1 and 1, instead of 0 and 1. Otherwise, our weighting scheme is not working
    """

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

        fig, ax = plt.subplots(figsize=(5, 4))
        # plot the cumulative histogram
        n, bins, patches = ax.hist(
            map_score,
            bins=100,
            density=True,
            histtype="step",
            cumulative=True,
            label="Empirical",
        )
        # tidy up the figure
        ax.grid(True)
        ax.legend(loc="upper left")
        ax.set_title("Cumulative step histograms")
        ax.set_xlabel("Lineage coupling accuracy")
        ax.set_ylabel("Cumulative probability")
    return map_score


def conditional_heatmap(
    coarse_X_clone,
    fate_names: list,
    included_fates: list = None,
    excluded_fates: list = None,
    binarize=True,
    mode="and",
    plot=True,
    **kwargs,
):
    """
    Plot a heatmap by conditionally including or removing a set of fates

    log_transform: True or False
    """
    fate_names = np.array(fate_names)

    if mode == "and":
        valid_clone_idx = np.ones(coarse_X_clone.shape[1]).astype(bool)
        if included_fates is not None:
            for x_name in included_fates:
                valid_clone_idx_tmp = coarse_X_clone[fate_names == x_name].sum(0) > 0
                valid_clone_idx = valid_clone_idx & valid_clone_idx_tmp

        if excluded_fates is not None:
            for x_name in excluded_fates:
                valid_clone_idx_tmp = coarse_X_clone[fate_names == x_name].sum(0) > 0
                valid_clone_idx = valid_clone_idx & ~valid_clone_idx_tmp
    elif mode == "or":
        valid_clone_idx = np.zeros(coarse_X_clone.shape[1]).astype(bool)
        if included_fates is not None:
            for x_name in included_fates:
                valid_clone_idx_tmp = coarse_X_clone[fate_names == x_name].sum(0) > 0
                valid_clone_idx = valid_clone_idx | valid_clone_idx_tmp

        if excluded_fates is not None:
            for x_name in excluded_fates:
                valid_clone_idx_tmp = coarse_X_clone[fate_names == x_name].sum(0) > 0
                valid_clone_idx = valid_clone_idx | ~valid_clone_idx_tmp

    new_matrix = coarse_X_clone[:, valid_clone_idx]
    X_count, norm_X_count = get_fate_count_coupling(new_matrix)
    new_matrix = cs.pl.custom_hierachical_ordering(
        np.arange(new_matrix.shape[0]), new_matrix
    )

    if binarize:
        new_matrix = (new_matrix > 0).astype(int)

    for j, x in enumerate(fate_names):
        print(f"Clone # ({x}): {(new_matrix>0).sum(1)[j]:.2f}")
    if plot:
        cs.pl.heatmap(
            new_matrix.T,
            order_map_x=False,
            order_map_y=False,
            x_ticks=fate_names,
            **kwargs,
        )
        plt.title(f"{np.sum(valid_clone_idx)} clones")
    return X_count, norm_X_count


##########################

# adata-oriented operations

###########################


def generate_adata_from_X_clone(X_clone, state_info=None):
    """
    Convert X_clone matrix to adata, and also add it to adata.obsm['X_clone'].
    You can run cospar on it directly.
    """
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


def generate_adata_sample_by_allele(df_data, count_value_key="UMI_count", use_UMI=True):
    """
    Take input from CARLIN output, like from `load_allele_info` or `extract_CARLIN_info`
    and generate an sample-by-allele adata object
    """
    all_mutation = np.array(list(set(df_data["allele"])))
    all_cells = np.array(list(set(df_data["sample"])))
    X_clone = np.zeros((len(all_cells), len(all_mutation)))
    for i, xx in enumerate(df_data["allele"]):
        yy = df_data.iloc[i]["sample"]
        idx_1 = np.nonzero(all_cells == yy)[0]
        idx_2 = np.nonzero(all_mutation == xx)[0]
        X_clone[idx_1, idx_2] = df_data.iloc[i][
            count_value_key
        ]  # This keeps the count information, and works better
        # X_clone[i,idx]=1

    X_clone = ssp.csr_matrix(X_clone)
    adata_orig = sc.AnnData(X_clone)
    adata_orig.var_names = all_mutation
    adata_orig.obs["time_info"] = ["0"] * X_clone.shape[0]
    adata_orig.obs["state_info"] = all_cells
    adata_orig.obsm["X_clone"] = X_clone
    adata_orig.uns["data_des"] = ["hi"]
    if "expected_frequency" in df_data.keys():
        adata_orig.uns["expected_frequency"] = np.array(df_data["expected_frequency"])
    adata_orig.uns[count_value_key] = np.array(df_data[count_value_key])

    if "mouse" in df_data.keys():
        adata_orig.uns["mouse"] = np.array(df_data["mouse"])

    return adata_orig


def generate_adata_allele_by_mutation(
    df_data, use_np_array=False, count_value_key="UMI_count", use_UMI=True
):
    """
    Take input from CARLIN output, like from `load_allele_info` or `extract_CARLIN_info`
    and generate an allele-by-mutation adata object
    """
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
                    count_value_key
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
                    if use_UMI:
                        value_temp = np.array(
                            df_data.iloc[i][count_value_key]
                        )  # or, value_temp=np.ones(len(idx))
                    else:
                        value_temp = 1
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
    adata_orig.obs[count_value_key] = np.array(df_data[count_value_key])

    if "mouse" in df_data.keys():
        adata_orig.obs["mouse"] = np.array(df_data["mouse"])
    adata_orig.var_names = all_mutation

    # adata_orig.obs["cell_id"] = [f"{xx}_{j}" for j, xx in enumerate(df_data["sample"])]
    if "sample" in df_data.keys():
        adata_orig.obs["sample"] = np.array(df_data["sample"])
        allele_dict = {}
        unique_alleles = sorted(list(set(adata_orig.obs["allele"])))
        for j, x in enumerate(unique_alleles):
            allele_dict[x] = str(j)
        allele_id_array = []
        for j, x in enumerate(adata_orig.obs["allele"]):
            allele_id_array.append(str(allele_dict[x]) + "_" + str(j))
        adata_orig.obs["cell_id"] = adata_orig.obs["sample"] + "*" + allele_id_array

    return adata_orig


def keep_informative_cell_and_clones(
    adata, clone_size_thresh=2, barcode_num_per_cell=1
):
    """
    Given an annadata object, select clones observed in more than one cells.
    and keep cells that have at least one clone
    """
    clone_idx = (adata.X > 0).sum(0).A.flatten() >= clone_size_thresh
    cell_idx = (adata.X[:, clone_idx] > 0).sum(1).A.flatten() >= barcode_num_per_cell
    adata_new = adata[:, clone_idx][cell_idx]
    adata_new.obsm["X_clone"] = adata_new.X
    return adata_new


def generate_clonal_fate_table(df_allele, thresh=0.2):
    """
    Convert df_allele table to a clone-fate matrix, also with an annotated fate outcome for each
    allele

    We perform normalization for each clone across all fates. Note that the clonal data might have
    cell-type-size normalized using the cell number information from the bulk data.

    We use the relative threshld `thresh` to determine the fate outcome of a given clone,
    but we also return the raw fate matrix.

    Return clonal fate table, and clonal fate matrix.
    """

    adata = generate_adata_sample_by_allele(df_allele)
    adata.obs_names = adata.obs["state_info"]
    adata.obs["mouse"] = [str(x).split("-")[0] for x in adata.obs_names]
    clone_id = adata.var_names
    clone_matrix = adata.X.A
    norm_X_tmp = (
        clone_matrix / clone_matrix.sum(axis=1)[:, np.newaxis]
    )  # normalize by row, i.e., within the same cell type
    norm_X = norm_X_tmp / (
        norm_X_tmp.sum(axis=0)[np.newaxis, :]
    )  # normalize by column, i.e., within the same clone
    # norm_X = adata.X.A / adata.X.A.sum(0)  # normalize each clone across all fates
    df_fate_matrix = pd.DataFrame(norm_X.T, index=clone_id, columns=adata.obs_names)
    fate_list = [",".join(adata.obs_names[x]) for x in norm_X.T > thresh]
    fate_N = [len(adata.obs_names[x]) for x in norm_X.T > thresh]
    mouse_N = [len(set(adata.obs["mouse"][x])) for x in norm_X.T > thresh]
    mouse_list = [",".join(set(adata.obs["mouse"][x])) for x in norm_X.T > thresh]
    df_allele_fate = pd.DataFrame(
        {
            "allele": clone_id,
            "fate": fate_list,
            "fate_N": fate_N,
            "fate_mouse_N": mouse_N,
            "fate_mouse": mouse_list,
        }
    )

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    ax = sns.histplot(norm_X[norm_X > 0].flatten(), bins=50, ax=axs[0])
    ax.set_xlabel("lineage weight")
    sns.histplot(df_allele_fate["fate_N"], ax=axs[1])
    sns.histplot(df_allele_fate["fate_mouse_N"], ax=axs[2])
    plt.tight_layout()
    return df_allele_fate, df_fate_matrix  # [df_allele_fate["mouse_N"] < 2]
