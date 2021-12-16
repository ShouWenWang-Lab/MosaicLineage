import gzip
import os

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import scipy.sparse as ssp
import seaborn as sns
import toolz as tz
from matplotlib import pyplot as plt
from tqdm import tqdm
from umi_tools import UMIClusterer


def denoise_clonal_data(
    df_raw,
    target_key="clone_id",
    base_read_cutoff=3,
    per_sample=None,
    denoise_method="Hamming",
    distance_threshold=None,
    whiteList=None,
    read_cutoff_ratio=0.5,
):
    """
    Denoise sequencing/PCR errors at a particular field.
    At the end, it generates a QC plot in terms of the sequence separation

    Parameters:
    -----------
    df_raw:
        The raw data
    target_key:
        The target field to correct sequeuncing/PCR errors.
    denoise_method:
        "Hamming", or "UMI_tools". The "Hamming" method works better.
    per_cell:
        denoise for each sample sepaerately, where we adjust the read threshold per sample.
        This can be cell or library.  The right input could be: None, 'cell_id', 'library'
    distance_threshold:
        distances to connect two sequences.
    whiteList:
        Only works for the method "Hamming"
    read_cutoff_ratio:
        only for per_cell=True. Help to modulate the read_cutoff per cell

    Returns:
    --------
    The corrected sequence is updated at df_out
    """

    df_input = df_raw.copy()
    if (per_sample is not None) and (per_sample in df_input.columns):
        cell_id_list = list(set(df_input[per_sample]))
        df_list = []
        for j in tqdm(range(len(cell_id_list))):
            cell_id_temp = cell_id_list[j]
            df_temp = df_input[df_input[per_sample] == cell_id_temp]

            # read_cutoff = np.max(
            #     [base_read_cutoff, read_cutoff_ratio * np.max(df_temp["read"])]
            # )
            read_cutoff = base_read_cutoff
            sp_idx = df_temp["read"] >= read_cutoff
            print(
                f"Currently cleaning {target_key}; number of unique elements: {len(set(df_temp[target_key][sp_idx]))}"
            )
            if np.sum(sp_idx) > 0:
                mapping, new_seq_list = denoise_sequence(
                    df_temp[sp_idx][target_key],
                    read_cout=df_temp[sp_idx]["read"],
                    distance_threshold=distance_threshold,
                    whiteList=whiteList,
                    method=denoise_method,
                    progress_bar=False,
                )

                df_temp[target_key][sp_idx] = new_seq_list
                df_temp[target_key][~sp_idx] = np.nan
                df_temp[target_key][df_temp[target_key] == "nan"] = np.nan
            df_list.append(df_temp)
        df_HQ = pd.concat(df_list).dropna()
    else:
        sp_idx = df_input.read >= base_read_cutoff
        print(
            f"Currently cleaning {target_key}; number of unique elements: {len(set(df_input[target_key][sp_idx]))}"
        )
        mapping, new_seq_list = denoise_sequence(
            df_input[sp_idx][target_key],
            read_cout=df_input[sp_idx]["read"],
            distance_threshold=distance_threshold,
            whiteList=whiteList,
            method=denoise_method,
        )
        df_input[target_key][sp_idx] = new_seq_list
        df_input[target_key][~sp_idx] = np.nan
        df_input[target_key][df_input[target_key] == "nan"] = np.nan
        df_HQ = df_input.dropna()

    ## report
    unique_seq = list(set(df_HQ[target_key]))
    print(f"Number of unique elements (after cleaning): {len(unique_seq)}")
    read_fraction_all = df_HQ["read"].sum() / df_raw["read"].sum()
    read_fraction_cutoff = (
        df_HQ["read"].sum() / df_raw[df_raw["read"] >= base_read_cutoff]["read"].sum()
    )
    print(f"Retained read fraction (above cutoff 0): {read_fraction_all:.2f}")
    print(
        f"Retained read fraction (above cutoff {base_read_cutoff}): {read_fraction_cutoff:.2f}"
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    distance = QC_sequence_distance(unique_seq)
    min_dis = plot_seq_distance(distance, ax=axs[0])
    read_coverage(df_HQ, target_key=target_key, ax=axs[1])
    return df_HQ


def read_coverage(df, target_key="clone_id", **kwargs):
    df_out = group_cells(df, group_keys=[target_key])
    ax = sns.histplot(df_out["read"], log_scale=True, cumulative=False, **kwargs)
    ax.set_xlabel(f"Total read of corrected {target_key}")
    ax.set_ylabel(f"Counts")
    return ax


def group_cells(df_HQ, group_keys=["library", "cell_id", "clone_id"]):
    df_out = df_HQ.groupby(group_keys).sum("read")
    df_out["umi_count"] = df_HQ.groupby(group_keys)["umi"].count().values
    df_out = df_out.reset_index()
    return df_out


def denoise_sequence(
    input_seqs,
    read_cout=None,
    distance_threshold=1,
    method="Hamming",
    whiteList=None,
    progress_bar=True,
):
    """
    Take the sequences, make a unique list, and order them by read count. The input_seqs does not need
    to be unique. We will aggregate the read count for the same sequence. From top to bottom, we iteratively find its similar sequences in the rest of the sequence pool, and remove them.

    Note that the output seq list could contain 'nan' if whitelist is used

    Parameters:
    -----------
    method:
        "Hamming", or "UMI_tools"
    seq_list:
        can be a list with duplicate sequences, indicating the read abundance of the read
    """
    seq_list = np.array(input_seqs).astype(bytes)
    if read_cout is None:
        read_count = np.ones(len(seq_list))
    if len(read_cout) != len(input_seqs):
        raise ValueError("read_count does not have the same size as input_seqs")
    df = pd.DataFrame({"seq": seq_list, "read": read_cout})
    df = (
        df.groupby("seq").sum("read").reset_index().sort_values("read", ascending=False)
    )
    if method == "UMI_tools":
        if whiteList is not None:
            raise ValueError("whitelist is not compatible with method=UMI_tools")
        seq_count = {df["seq"].iloc[j]: df["read"].iloc[j] for j in range(len(df))}
        if distance_threshold is None:
            distance_threshold = round(0.1 * len(seq_list[0]))

        if progress_bar:
            print(
                f"Sequences within Hamming distance {distance_threshold} are connected"
            )

        clusterer = UMIClusterer(cluster_method="directional")
        clustered_umis = clusterer(seq_count, threshold=distance_threshold)
        mapping = {}
        for umi_list in clustered_umis:
            for umi in umi_list:
                mapping[umi] = umi_list[0]
    else:  # "Hamming"
        if progress_bar:
            print(
                f"Sequences within Hamming distance {distance_threshold} are connected"
            )
        # quality_seq_list = []
        mapping = {}
        unique_seq_list = list(df["seq"])
        if progress_bar:
            print(f"Processing {len(unique_seq_list)} unique sequences")
        remaining_seq_idx = np.ones(len(unique_seq_list)).astype(bool)
        source_seqs = np.array([list(xx) for xx in unique_seq_list])
        if whiteList is None:
            iter = range(len(unique_seq_list))
            if progress_bar:
                iter = tqdm(iter)
            for __ in iter:
                cur_ids = np.nonzero(remaining_seq_idx)[0]
                id_0 = cur_ids[0]
                cur_seq = unique_seq_list[id_0]
                remain_seq_array = source_seqs[remaining_seq_idx]
                distance_vector = np.sum(remain_seq_array != remain_seq_array[0], 1)
                target_ids = np.nonzero(distance_vector < distance_threshold)[0]
                for k in target_ids:
                    abs_id = cur_ids[k]
                    seq_tmp = unique_seq_list[abs_id]
                    mapping[seq_tmp] = cur_seq
                    remaining_seq_idx[
                        abs_id
                    ] = False  # switch to idx to prevent modifying id list dynamically

                if np.sum(remaining_seq_idx) <= 0:
                    break
        else:
            whiteList_1 = np.array(whiteList).astype(bytes)
            target_seqs = np.array([list(xx) for xx in whiteList_1])
            iter = range(len(whiteList_1))
            if progress_bar:
                iter = tqdm(iter)
            for j in iter:
                cur_seq = whiteList_1[j]
                cur_ids = np.nonzero(remaining_seq_idx)[0]
                remain_seq_array = source_seqs[remaining_seq_idx]
                distance_vector = np.sum(remain_seq_array != target_seqs[j], 1)
                target_ids = np.nonzero(distance_vector < distance_threshold)[0]
                for k in target_ids:
                    abs_id = cur_ids[k]
                    seq_tmp = unique_seq_list[abs_id]
                    mapping[seq_tmp] = cur_seq
                    remaining_seq_idx[
                        abs_id
                    ] = False  # switch to idx to prevent modifying id list dynamically

    if whiteList is None:
        new_seq_list = np.array([mapping[xx] for xx in seq_list]).astype(str)
    else:
        shared_idx = np.in1d(seq_list, list(mapping.keys()))
        new_seq_list = np.array(seq_list).copy()
        new_seq_list[shared_idx] = [mapping[xx] for xx in seq_list[shared_idx]]
        new_seq_list = np.array(new_seq_list).astype(str)
        new_seq_list[~shared_idx] = np.nan

    return mapping, new_seq_list


def QC_sequence_distance(source_seqs_0, target_seqs_0=None, Kmer=1, deduplicate=False):
    """
    First, remove duplicate sequences

    We partition the sequence into Kmers
    eg. 'ABCDEF', Kmers=2 -> ['AB','CD','EF']
    Then, calculate the Hamming distance in the kmer space

    This calculation is exact, and can be slow for large amount of sequences
    """

    if deduplicate:
        source_seqs_0 = list(set(source_seqs_0))
        if target_seqs_0 is not None:
            target_seqs_0 = list(set(target_seqs_0))

    source_seqs = np.array([seq_partition(Kmer, xx) for xx in source_seqs_0])
    if target_seqs_0 is None:
        target_seqs = source_seqs
    else:
        target_seqs = np.array([seq_partition(Kmer, xx) for xx in target_seqs_0])

    seq_N = len(target_seqs)
    ini_N = len(source_seqs)
    distance = np.zeros((ini_N, seq_N))

    if len(source_seqs) > len(target_seqs):
        for j in tqdm(range(len(target_seqs))):
            distance[:, j] = np.sum(source_seqs != target_seqs[j], 1)
    else:
        for j in tqdm(range(len(source_seqs))):
            distance[j, :] = np.sum(target_seqs != source_seqs[j], 1)

    return distance


def plot_seq_distance(distance, **kwargs):
    np.fill_diagonal(distance, np.inf)
    min_distance = distance.min(axis=1)
    ax = sns.histplot(min_distance, **kwargs)
    ax.set_xlabel("Minimum intra-seq hamming distance")
    return min_distance


def seq_partition(n, seq):
    """
    Partition sequence into every n-bp

    eg. 'ABCDEF', n=2 -> [['AB'],['CD'],['EF']]
    """
    if n == 1:
        return list(seq)
    else:
        return ["".join(x) for x in tz.partition(n, seq)]


def QC_clonal_bc_per_cell(df0, read_cutoff=3, plot=True, **kwargs):
    """
    Get Number of clonal bc per cell
    """
    df = df0[df0["read"] >= read_cutoff]
    df_statis = (
        df.groupby(["cell_id"])
        .apply(lambda x: len(set(x["clone_id"])))
        .to_frame(name="clonal_bc_number")
        .reset_index()
    )
    if plot:
        ax = sns.histplot(data=df_statis, x="clonal_bc_number", **kwargs)
        ax.set_xlabel("Number of clonal bc per cell")
        ax.set_ylabel("Count")
    return df_statis


def QC_clonal_reports(df, title=None, **kwargs):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    QC_clone_size(df, read_cutoff=0, ax=axs[0], **kwargs)
    QC_clonal_bc_per_cell(df, read_cutoff=0, ax=axs[1], **kwargs)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    return fig


def QC_clone_size(df0, read_cutoff=3, plot=True, **kwargs):
    df = df0[df0["read"] >= read_cutoff]
    df_statis = (
        df.groupby("clone_id")
        .apply(lambda x: len(set(x["cell_id"])))
        .to_frame(name="clone_size")
        .reset_index()
    )
    if plot:
        ax = sns.histplot(data=df_statis, x="clone_size", **kwargs)
        ax.set_xlabel("Clone size")
        ax.set_ylabel("Count")
    return df_statis


def QC_read_per_molecule(
    df_input_0,
    target_keys=["clone_id", "umi"],
    group_key="cell_id",
    log_scale=True,
    read_cutoff=None,
):
    if read_cutoff is not None:
        df_input = df_input_0[df_input_0.read >= read_cutoff]
    else:
        df_input = df_input_0
    for key in target_keys:
        df_temp = df_input.groupby([group_key, key]).sum("read").reset_index()
        df_plot = pd.DataFrame(
            {
                f"Read per {group_key}": df_temp.groupby(group_key)
                .sum("read")["read"]
                .values,
                f"{key} number": df_temp.groupby(group_key).count()[key].values,
            }
        )
        # This is much faster
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(
            df_plot[f"Read per {group_key}"], df_plot[f"{key} number"], marker=".", s=3
        )
        ax.set_xlabel(f"Number of reads per {group_key}")
        ax.set_ylabel(f"Number of {key} per cell")
        if log_scale:
            plt.xscale("log")
            plt.yscale("log")

        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax = sns.histplot(
            data=df_plot,
            x=f"Read per {group_key}",
            bins=100,
            log_scale=log_scale,
            ax=ax,
        )
        ax.set_xlabel(f"Number of reads per {group_key}")
        ax.set_ylabel("Histogram")
        if log_scale:
            plt.yscale("log")

        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax = sns.histplot(
            data=df_plot, x=f"{key} number", bins=100, log_scale=log_scale, ax=ax
        )
        ax.set_xlabel(f"Number of {key} per cell")
        ax.set_ylabel(f"Histogram")
        if log_scale:
            plt.yscale("log")


def QC_unique_cells(df, target_keys=["cell_id", "clone_id"], base=2, log_scale=True):
    max_read = df["read"].max()
    upper_log2 = np.ceil(np.log(max_read) / np.log(base))
    read_cutoff_list = []
    unique_count = []
    for x in range(int(upper_log2)):
        read_cutoff = base ** x
        read_cutoff_list.append(read_cutoff)
        df_temp = df[df["read"] >= read_cutoff].reset_index()
        temp_list = []
        for key in target_keys:
            temp_list.append(len(set(df_temp[key])))

        unique_count.append(temp_list)

    unique_count = np.array(unique_count)
    for j, key in enumerate(target_keys):
        fig, ax = plt.subplots()
        ax = sns.scatterplot(read_cutoff_list, unique_count[:, j])
        ax.set_xlabel("Read cutoff")
        if log_scale:
            plt.xscale("log")
            plt.yscale("log")
        ax.set_ylabel(f"Unique {key} number")


def generate_LARRY_read_count_table(data_path, sample_list, recompute=False):
    """
    From f"{data_path}/{lib}.LARRY.fastq.gz" --> f"{data_path}/{lib}.LARRY.csv"
    where the read number of each molecular is calculated.

    We use cell barcode + sample id to jointly update the cell_id tag
    We use the cell barcode + umi to jointly define the umi_id tag

    We first load all data into memory before extract the read information. This assumes that
    the data is not too big to fit into the memory (<10G ?)
    """

    df_list = []
    for lib in sample_list:
        csv_file_name = f"{data_path}/{lib}.LARRY.csv"
        if os.path.exists(csv_file_name) and (not recompute):
            data_table = pd.read_csv(csv_file_name, index_col=0)
        else:
            counts = {}
            f = gzip.open(f"{data_path}/{lib}.LARRY.fastq.gz")

            # for other files that starts with a normal line, skip the above two lines and run directly:
            print(f"Reading in library {lib}")
            all_lines = f.readlines()
            current_tag = []
            for x in tqdm(all_lines):
                l = x.decode("utf-8").strip("\n")
                if l == "":
                    current_tag = []
                elif l[0] == ">":
                    current_tag = l[1:].split(",")
                elif l != "" and len(current_tag) == 3:
                    current_tag.append(l)
                    current_tag = tuple(current_tag)
                    if not current_tag in counts:
                        counts[current_tag] = 0
                    counts[current_tag] += 1

            sample_id = [k[0] for k, v in counts.items()]
            cell_bc = [k[1] for k, v in counts.items()]
            umi_id = [k[2] for k, v in counts.items()]
            gfp_bc_id = [k[3] for k, v in counts.items()]
            read_count = [v for k, v in counts.items()]
            library_id = [lib for _ in range(len(sample_id))]

            data_table = pd.DataFrame(
                {"library": library_id, "umi": umi_id, "cell_bc": cell_bc}
            )
            data_table["umi_id"] = data_table["cell_bc"] + "_" + data_table["umi"]
            data_table["cell_id"] = data_table["library"] + "_" + data_table["cell_bc"]
            data_table["clone_id"] = gfp_bc_id
            data_table["read"] = read_count
            data_table.to_csv(f"{data_path}/{lib}.LARRY.csv")

        df_list.append(data_table)
    df_all = pd.concat(df_list)
    return df_all


def print_statistics(df, read_cutoff=None):
    if read_cutoff is not None:
        df_tmp = df[df.read >= read_cutoff]
    else:
        df_tmp = df
    for key in ["library", "cell_id", "clone_id", "umi_id"]:
        if key in df.columns:
            print(f"{key} number: {len(set(df_tmp[key]))}")
    print(f"total reads: {np.sum(df_tmp['read'])/1000:.0f}K")


def remove_cells(
    df,
    read_cutoff=None,
    umi_cutoff=None,
    clone_bc_number_cutoff=None,
    clone_size_cutoff=None,
):
    df_out = df.copy()
    if read_cutoff is not None:
        df_out = df_out[df_out["read"] >= read_cutoff]
    if umi_cutoff is not None:
        df_out = df_out[df_out["umi_count"] >= umi_cutoff]
    if clone_bc_number_cutoff is not None:
        df_bc_N = QC_clonal_bc_per_cell(df_out, read_cutoff=0, plot=False)
        df_bc_N = df_bc_N[df_bc_N["clonal_bc_number"] <= clone_bc_number_cutoff]
        df_out = df_out[df_out.cell_id.isin(df_bc_N["cell_id"])]

    if clone_size_cutoff is not None:
        df_clone_N = QC_clone_size(df_out, read_cutoff=0, plot=False)
        df_clone_N = df_clone_N[df_clone_N["clone_size"] <= clone_size_cutoff]
        df_out = df_out[df_out.clone_id.isin(df_clone_N["clone_id"])]
    return df_out


def rename_library_info(df_all, mapping_dictionary):
    """
    Mapping one library name into another.

    As an example:
    mapping_dictionary={'LARRY_Lime_33':'Lime_RNA_101','LARRY_Lime_34':'Lime_RNA_102', 'LARRY_Lime_35':'Lime_RNA_103',
                   'LARRY_Lime_36':'Lime_RNA_104','LARRY_10X_31':'MPP_10X_A3_1', 'LARRY_10X_32':'MPP_10X_A4_1'}
    """

    for key in tqdm(mapping_dictionary.keys()):
        df_all["library"][df_all.library == key] = mapping_dictionary[key]
    df_all.loc[:, "cell_id"] = df_all["library"] + "_" + df_all["cell_bc"]
    return df_all


def merge_adata_across_times(
    adata_t1, adata_t2, X_shift=12, embed_key="X_umap", data_des="scLimeCat"
):
    adata_t1_ = adata_t1.raw.to_adata()
    adata_t2_ = adata_t2.raw.to_adata()
    adata_t1_.obsm[embed_key] = adata_t1_.obsm[embed_key] + X_shift

    adata_t1_.obs["time_info"] = ["1" for x in range(adata_t1_.shape[0])]
    adata_t2_.obs["time_info"] = ["2" for x in range(adata_t2_.shape[0])]

    adata_t1_.obs["leiden"] = [f"t1_{x}" for x in adata_t1_.obs["leiden"]]
    adata_t2_.obs["leiden"] = [f"t2_{x}" for x in adata_t2_.obs["leiden"]]

    adata = adata_t1_.concatenate(adata_t2_, join="outer")
    adata.obs_names = [
        xx.split("-")[0] for xx in adata.obs_names
    ]  # we assume the names are unique
    adata.obsm["X_emb"] = adata.obsm[embed_key]
    adata.uns["data_des"] = [data_des]
    return adata
