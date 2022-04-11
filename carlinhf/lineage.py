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
from tqdm import tqdm

import carlinhf.LINE1 as line1

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


def generate_adata_sample_by_allele(df_data, use_UMI=True):
    all_mutation = np.array(list(set(df_data["allele"])))
    all_cells = np.array(list(set(df_data["sample"])))
    X_clone = np.zeros((len(all_cells), len(all_mutation)))
    for i, xx in enumerate(df_data["allele"]):
        yy = df_data.iloc[i]["sample"]
        idx_1 = np.nonzero(all_cells == yy)[0]
        idx_2 = np.nonzero(all_mutation == xx)[0]
        X_clone[idx_1, idx_2] = df_data.iloc[i][
            "obs_UMI_count"
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
    adata_orig.uns["obs_UMI_count"] = np.array(df_data["obs_UMI_count"])

    if "mouse" in df_data.keys():
        adata_orig.uns["mouse"] = np.array(df_data["mouse"])

    return adata_orig


def generate_adata(df_data, use_np_array=False, use_UMI=True):
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
                    if use_UMI:
                        value_temp = np.array(
                            df_data.iloc[i]["obs_UMI_count"]
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
    adata_orig.obs["obs_UMI_count"] = np.array(df_data["obs_UMI_count"])

    # adata_orig.obs["cell_id"] = [f"{xx}_{j}" for j, xx in enumerate(df_data["sample"])]
    if "mouse" in df_data.keys():
        adata_orig.obs["mouse"] = np.array(df_data["mouse"])
    adata_orig.var_names = all_mutation

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


def load_allele_info(data_path):
    pooled_data = loadmat(os.path.join(data_path, "allele_annotation.mat"))
    allele_freqs = pooled_data["allele_freqs"].flatten()
    alleles = [xx[0][0] for xx in pooled_data["AlleleAnnotation"]]
    return pd.DataFrame({"allele": alleles, "UMI_count": allele_freqs})


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
    return 2 ** entropy


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


def keep_informative_cell_and_clones(
    adata, clone_size_thresh=2, barcode_num_per_cell=1
):
    # select clones observed in more than one cells.
    # and keep cells that have at least one clone
    clone_idx = (adata.X > 0).sum(0).A.flatten() >= clone_size_thresh
    cell_idx = (adata.X[:, clone_idx] > 0).sum(1).A.flatten() >= barcode_num_per_cell
    adata_new = adata[:, clone_idx][cell_idx]
    adata_new.obsm["X_clone"] = adata_new.X
    return adata_new


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


def subsample_allele_freq_histogram(df, sample_key, sample_fraction=0.1, plot=True):

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

    # if 1 * sample_fraction < 1:
    #     threshold = 1
    # else:
    #     # a middle barrier between 1 and 2, expected UMI
    #     threshold = 1 * sample_fraction

    # print(f"The threshold: {threshold}")
    singleton_ratio = np.sum(df_new["UMI_count"] <= df_new["UMI_count"].min()) / len(
        df_new
    )

    if plot:
        print(f"Singleton ratio: {singleton_ratio}")

        x_var, y_var = line1.plot_loghist(list(df_new["UMI_count"]), cutoff_y=3)
        plt.tight_layout()
        plt.xlabel("Occurence # per allele (UMI count)")
        plt.ylabel("Histogram")
        plt.title(f"Singleton ratio: {singleton_ratio:.2f}")
        plt.savefig(f"figure/{sample_key}/subsampled_allele_frequency_distribution.pdf")

    return df_new, singleton_ratio


def subsample_singleton_fraction(df, sample_key, sample_fraction_array, plot=True):

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


def visualize_tree(
    input_tree,
    color_coding=None,
    mode="r",
    width=60,
    height=60,
    dpi=300,
    data_des="tree",
    figure_path=".",
):
    """
    mode: r or c
    """

    from ete3 import AttrFace, NodeStyle, Tree, TreeStyle, faces
    from IPython.display import Image, display

    def layout(node):
        if node.is_leaf():
            N = AttrFace("name", fsize=5)
            faces.add_face_to_node(N, node, 100, position="aligned")
            # pass

    if color_coding is not None:
        print("coding")
        for n in input_tree.traverse():
            nst1 = NodeStyle(size=1, fgcolor="#f0f0f0")
            n.set_style(nst1)

        for n in input_tree:
            for key, value in color_coding.items():
                if n.name.startswith(key):
                    nst1 = NodeStyle(size=1)
                    nst1["bgcolor"] = value
                    n.set_style(nst1)

    ts = TreeStyle()
    ts.layout_fn = layout
    ts.show_leaf_name = False
    ts.mode = mode
    # ts.extra_branch_line_color = "red"
    # ts.extra_branch_line_type = 0
    input_tree.render(
        os.path.join(figure_path, f"{data_des}.pdf"),
        tree_style=ts,
        w=width,
        h=height,
        units="mm",
    )
    input_tree.render(
        os.path.join(figure_path, f"{data_des}.png"),
        tree_style=ts,
        w=width,
        h=height,
        dpi=dpi,
        units="mm",
    )

    display(Image(filename=os.path.join(figure_path, f"{data_des}.png")))


def clonal_analysis(
    adata_sub, data_des="all", scenario="coarse", data_path=".", color_coding=None
):

    if scenario == "coarse":
        print("coarse-grained analysis")
        ## generate plots for the coarse-grained data
        # adata_sub.obs["state_info"] = adata_sub.obs["sample"]
        adata_sub.uns["data_des"] = [f"{data_des}_coarse"]
        cs.settings.data_path = data_path
        cs.settings.figure_path = data_path
        os.makedirs(data_path, exist_ok=True)
        cs.pl.barcode_heatmap(
            adata_sub,
            color_bar=True,
            fig_height=10,
            fig_width=10,
            y_ticks=None,  # adata_sub.var_names,
            x_label="Allele",
            y_label="Mutation",
        )
        cs.tl.fate_coupling(adata_sub, source="X_clone")
        cs.pl.fate_coupling(adata_sub, source="X_clone")
        cs.tl.fate_hierarchy(adata_sub, source="X_clone")
        my_tree_coarse = adata_sub.uns["fate_hierarchy_X_clone"]["tree"]
        with open(f"{cs.settings.data_path}/{data_des}_coarse_tree.txt", "w") as f:
            f.write(my_tree_coarse.write())

        visualize_tree(
            my_tree_coarse,
            color_coding=color_coding,
            mode="r",
            data_des=f"{data_des}_coarse",
            figure_path=cs.settings.data_path,
        )

    else:
        print("refined analysis, using all cells")
        ## refined heatmap and coupling, no ticks
        # adata_sub.obs["state_info"] = adata_sub.obs["cell_id"]
        adata_sub.uns["data_des"] = [f"{data_des}_refined"]
        cs.pl.barcode_heatmap(
            adata_sub,
            color_bar=True,
            fig_height=10,
            fig_width=12,
            y_ticks=None,  # adata_sub.var_names,
            x_label="Allele",
            y_label="Mutation",
            x_ticks=None,
        )
        cs.tl.fate_coupling(adata_sub, source="X_clone")
        cs.pl.fate_coupling(
            adata_sub,
            source="X_clone",
            x_ticks=None,
            y_ticks=None,
            x_label="Allele",
            y_label="Allele",
        )
        cs.tl.fate_hierarchy(adata_sub, source="X_clone")
        my_tree_refined = adata_sub.uns["fate_hierarchy_X_clone"]["tree"]
        with open(f"{cs.settings.data_path}/{data_des}_refined_tree.txt", "w") as f:
            f.write(my_tree_refined.write())

        visualize_tree(
            my_tree_refined,
            color_coding=color_coding,
            mode="c",
            data_des=f"{data_des}_refined",
            figure_path=cs.settings.figure_path,
            dpi=300,
        )


def correct_null_allele_frequency(df_input, editing_efficiency=0.3):
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


def conditional_heatmap(
    coarse_X_clone,
    fate_names: list,
    included_fates: list = None,
    excluded_fates: list = None,
    plot=True,
    **kwargs,
):
    """

    log_transform: True or False
    """
    fate_names = np.array(fate_names)
    valid_clone_idx = np.ones(coarse_X_clone.shape[1]).astype(bool)
    if included_fates is not None:
        for x_name in included_fates:
            valid_clone_idx_tmp = coarse_X_clone[fate_names == x_name].sum(0) > 0
            valid_clone_idx = valid_clone_idx & valid_clone_idx_tmp

    if excluded_fates is not None:
        for x_name in excluded_fates:
            valid_clone_idx_tmp = coarse_X_clone[fate_names == x_name].sum(0) > 0
            valid_clone_idx = valid_clone_idx & ~valid_clone_idx_tmp

    new_matrix = coarse_X_clone[:, valid_clone_idx]
    X_count, norm_X_count = get_fate_count_coupling(new_matrix)

    for j, x in enumerate(fate_names):
        print(f"Clone # ({x}): {(new_matrix>0).sum(1)[j]:.2f}")
    if plot:
        cs.pl.heatmap(
            new_matrix.T,
            order_map_x=False,
            x_ticks=fate_names,
            **kwargs,
        )
        plt.title(f"{np.sum(valid_clone_idx)} clones")
    return X_count, norm_X_count


def custom_hierachical_ordering(order_ids, matrix, pseudo_count=0.00001):
    """
    A recursive algorithm to rank the clones.
    The matrix is fate-by-clone, and we order it in the clone dimension
    """
    if (len(order_ids) < 1) or (matrix.shape[1] < 2):
        return matrix

    order_ids = np.array(order_ids)
    valid_clone_idx = np.ones(matrix.shape[1]) > 0
    new_data_list = []
    for j, x in enumerate(order_ids):
        valid_clone_idx_tmp = valid_clone_idx & (matrix[x] > 0)
        data_matrix = matrix[:, valid_clone_idx_tmp].T
        valid_clone_idx = valid_clone_idx & (~valid_clone_idx_tmp)
        if np.sum(valid_clone_idx_tmp) >= 2:
            order_y = cs.hf.get_hierch_order(data_matrix + pseudo_count)
            updated_matrix = data_matrix[order_y].T
        else:
            updated_matrix = data_matrix.T
        updated_matrix_1 = custom_hierachical_ordering(
            order_ids[j + 1 :], updated_matrix, pseudo_count=pseudo_count
        )
        new_data_list.append(updated_matrix_1)
    new_data_list.append(
        matrix[:, valid_clone_idx]
    )  # add the remaining clones not selected before
    return np.column_stack(new_data_list)


def plot_pie_chart(
    matrix,
    fate_names,
    include_fate=None,
    labeldistance=1.1,
    rotatelabels=True,
    counterclock=False,
    textprops={"fontsize": 12},
    **kwargs,
):
    """
    Plot the pie chart for cell numbers overlapped between different fates. The input matrix should be a fate-by-clone matrix.

    matrix.shape[0]=len(fate_names)

    In the first step, we transform the matrix to a boelean matrix
    """

    matrix = matrix > 0
    fate_names = np.array(fate_names)
    if include_fate is None:
        matrix_sub = matrix
    else:
        assert include_fate in fate_names
        clone_idx = matrix[fate_names == include_fate, :].sum(0) > 0
        matrix_sub = matrix[:, clone_idx]

    cell_type_dict = {}
    for i in range(matrix_sub.shape[1]):
        id_tmp = tuple(sorted(np.array(fate_names)[matrix_sub[:, i]]))
        if id_tmp not in cell_type_dict.keys():
            cell_type_dict[id_tmp] = 1  # initial cell number
        else:
            cell_type_dict[id_tmp] += 1

    your_data = dict(sorted(cell_type_dict.items()))
    labels = []
    sizes = []

    for x, y in your_data.items():
        tmp = list(x)
        tmp.append(y)
        labels.append(tmp)
        sizes.append(y)

    # Plot
    plt.pie(
        sizes,
        labels=labels,
        labeldistance=labeldistance,
        rotatelabels=rotatelabels,
        counterclock=counterclock,
        textprops=textprops,
        **kwargs,
    )

    plt.axis("equal")


def plot_venn3(data_1, data_2, data_3, labels=["1", "2", "3"]):

    set_1 = set(data_1)
    set_2 = set(data_2)
    set_3 = set(data_3)

    from matplotlib import pyplot as plt
    from matplotlib_venn import (
        venn2,
        venn2_circles,
        venn2_unweighted,
        venn3,
        venn3_circles,
    )

    vd3 = venn3(
        [set_1, set_2, set_3],
        set_labels=labels,
        set_colors=("#c4e6ff", "#F4ACB7", "#9D8189"),
        alpha=0.8,
    )
    venn3_circles([set_1, set_2, set_3], linestyle="-", linewidth=0.5, color="grey")
    for text in vd3.set_labels:
        text.set_fontsize(16)
    for text in vd3.subset_labels:
        text.set_fontsize(16)


def plot_venn2(data_1, data_2, labels=["1", "2"]):

    set_1 = set(data_1)
    set_2 = set(data_2)

    from matplotlib import pyplot as plt
    from matplotlib_venn import (
        venn2,
        venn2_circles,
        venn2_unweighted,
        venn3,
        venn3_circles,
    )

    vd3 = venn2(
        [set_1, set_2],
        set_labels=labels,
        set_colors=("#c4e6ff", "#F4ACB7"),
        alpha=0.8,
    )
    venn2_circles([set_1, set_2], linestyle="-", linewidth=0.5, color="grey")
    for text in vd3.set_labels:
        text.set_fontsize(16)
    for text in vd3.subset_labels:
        text.set_fontsize(16)


def sub_sample(df, size=1000, replace=True):
    dist = np.array(df["UMI_count"] / np.sum(df["UMI_count"]))
    sel_idx = np.random.choice(np.arange(len(df)), size=size, p=dist, replace=replace)
    return df.iloc[sel_idx].drop_duplicates("allele")


def check_allele_frequency_prediction(
    df, UMI_cutoff=30, mutation_N_cutoff=1, markersize=25, df_mutation=None
):
    if df_mutation is None:
        df_mutation = mutation_frequency(df, plot=False)

    df_mutation = df_mutation.set_index("mutation")
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


def generate_synthetic_alleles(
    df_allele: pd.DataFrame,
    target_sample_N: int = 10 ** 4,
    max_mutation_N: int = 5,
    random_seed=123,
):
    """
    Parameters
    ----------
    df_allele:
        pandas dataframe for allele count
    target_sample_N:
        target sequence number to sample
    max_mutation_N:
        Maximum number of mutations in a synthetic allele. This is usefual because if the mutation_N is too big,
        like 10, then it is really hard to find 5 mutations that would satisfy the constraints of an valid allele.
    random_seed:
        random seed of the generation process. Using the same seed, the result is deterministic.

    Returns
    -------
    df_synthesis:
        pandas dataframe for synthetic allele count
    """

    np.random.seed(random_seed)

    ## extract the mutation data from the allele data
    df_mutation = mutation_frequency(df_allele, plot=False)
    norm_factor = df_mutation["UMI_count"].sum()
    df_mutation["Frequency"] = df_mutation["UMI_count"] / norm_factor

    ## extract the start and end position of a mutation
    start_L = np.zeros(len(df_mutation)) - 10
    end_L = np.zeros(len(df_mutation)) - 10
    for j, x in enumerate(df_mutation["mutation"]):
        if "del" in x:  # del or delins
            temp = x.split("del")[0].split("_")
            start_L[j] = temp[0]
            end_L[j] = temp[1]
        elif "ins" in x:  # ins
            temp = x.split("ins")[0].split("_")
            start_L[j] = temp[0]
            end_L[j] = temp[1]
        elif ">" in x:
            temp = x.split(">")[0][:-1]
            start_L[j] = temp
            end_L[j] = temp
    df_mutation["start_position"] = start_L.astype(int)
    df_mutation["end_position"] = end_L.astype(int)

    ## extract mutation number histogram
    mut_per_allele = mutations_per_allele(df_allele)
    mut_per_UMI = np.concatenate(
        [[mut_per_allele[i]] * int(x) for i, x in enumerate(df_allele["UMI_count"])]
    )
    mut_hist_y, mut_hist_x = np.histogram(mut_per_UMI, bins=np.arange(17))
    mut_hist_UMI = mut_hist_y / np.sum(mut_hist_y)

    ## generate data for different types of mutations, within each type, we normalize the sampling frequency
    df_mutation["delins"] = df_mutation["mutation"].apply(lambda x: "delins" in x)
    df_mutation["del"] = df_mutation["mutation"].apply(
        lambda x: ("del" in x) and ("ins" not in x)
    )
    df_mutation["ins"] = df_mutation["mutation"].apply(
        lambda x: ("del" not in x) and ("ins" in x)
    )
    df_mutation["others"] = df_mutation["mutation"].apply(
        lambda x: ("del" not in x) and ("ins" not in x)
    )

    df_delins = df_mutation[df_mutation["delins"]].filter(
        ["mutation", "UMI_count", "Frequency", "start_position", "end_position"]
    )
    df_delins["Frequency"] = df_delins["Frequency"] / df_delins["Frequency"].sum()
    df_del = df_mutation[df_mutation["del"]].filter(
        ["mutation", "UMI_count", "Frequency", "start_position", "end_position"]
    )
    df_del["Frequency"] = df_del["Frequency"] / df_del["Frequency"].sum()
    df_ins = df_mutation[df_mutation["ins"]].filter(
        ["mutation", "UMI_count", "Frequency", "start_position", "end_position"]
    )
    df_ins["Frequency"] = df_ins["Frequency"] / df_ins["Frequency"].sum()
    df_others = df_mutation[df_mutation["others"]].filter(
        ["mutation", "UMI_count", "Frequency", "start_position", "end_position"]
    )
    df_others["Frequency"] = df_others["Frequency"] / df_others["Frequency"].sum()
    # df_others=df_others[df_others.mutation!='[]']
    df_list = [df_delins, df_del, df_ins, df_others]

    ## estimate probability for different types of mutations
    # mutation_type_prob=[len(df_delins), len(df_del), len(df_ins), len(df_others)] # by allele
    mutation_type_prob = [
        df_delins["UMI_count"].sum(),
        df_del["UMI_count"].sum(),
        df_ins["UMI_count"].sum(),
        df_others["UMI_count"].sum(),
    ]  # by UMI
    mutation_type_prob = np.array(mutation_type_prob) / np.sum(mutation_type_prob)

    ## generate the random number before the actual computation
    prob = mut_hist_UMI[: (max_mutation_N + 1)]
    mutation_N_array = np.random.choice(
        mut_hist_x[: (max_mutation_N + 1)], size=target_sample_N, p=prob / np.sum(prob)
    )
    type_id_array = np.random.choice(
        np.arange(4),
        size=int(target_sample_N * np.mean(mutation_N_array)),
        p=mutation_type_prob,
    )
    mutation_id_list = []
    for cur_id in range(len(df_list)):
        print(f"Current id: {cur_id}")
        size = 500 * target_sample_N * mutation_type_prob[cur_id]
        type_id = np.random.choice(
            np.arange(len(df_list[cur_id])),
            size=int(size),
            p=df_list[cur_id]["Frequency"],
        )
        mutation_id_list.append(type_id)

    from tqdm import tqdm

    ## initialize the simulation
    type_id_cur = 0
    mutation_type_start = np.array([0, 0, 0, 0])
    mutation_type_start_max = np.array([len(x) - 2 for x in mutation_id_list])
    new_allele_array = []
    for j in tqdm(range(target_sample_N)):
        mutation_N = mutation_N_array[j]
        # print(f'round {j}; current mutation number {mutation_N}')

        # select number of mutations in this allele
        type_id_end = type_id_cur + mutation_N
        type_id_temp = type_id_array[type_id_cur:type_id_end]
        type_id_cur = type_id_end

        ## select mutations from different types
        success = False
        while (success is False) and (
            mutation_type_start < mutation_type_start_max
        ).all():
            sel_mutations = []
            start_position_array = []
            end_position_array = []
            for x in type_id_temp:
                start_temp = mutation_type_start[x]
                mutation_type_start[x] += 1
                mutation_index = mutation_id_list[x][start_temp]
                mutation_temp = df_list[x].iloc[mutation_index]
                sel_mutations.append(mutation_temp)
                start_position_array.append(mutation_temp["start_position"])
                end_position_array.append(mutation_temp["end_position"])

            ## check if the selection is reasonable
            reorder_idx_start = np.argsort(start_position_array).flatten()
            reorder_idx_end = np.argsort(end_position_array).flatten()
            unique_start = len(set(start_position_array)) == len(start_position_array)
            unique_end = len(set(start_position_array)) == len(start_position_array)
            if (
                (reorder_idx_start == reorder_idx_end).all()
                and unique_start
                and unique_end
            ):  # they should satisfy the same ordering
                new_alleles = ",".join(
                    [
                        sel_mutations[i0]["mutation"]
                        for i0 in reorder_idx_start.flatten()
                    ]
                )
                new_allele_array.append(new_alleles)
                success = True
            # otherwise, go for the next while loop

        if (mutation_type_start >= mutation_type_start_max).any():
            print("mutation type data insufficient. Break")
            break

    df_synthesis = pd.DataFrame({"allele": new_allele_array})
    df_synthesis["UMI_count"] = 1
    return df_synthesis.groupby("allele").agg({"UMI_count": "sum"}).reset_index()
