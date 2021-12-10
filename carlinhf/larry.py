import gzip
import os

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import seaborn as sns
import toolz as tz
from matplotlib import pyplot as plt
from tqdm import tqdm
from umi_tools import UMIClusterer


def denoise_clonal_data(
    df_input,
    target_keys=["umi", "clone_id"],
    base_read_cutoff=10,
    method="single_cell",
    read_cutoff_ratio=0.5,
    threshold=None,
):
    """
    Clean umi and LARRY barcode within each cell (especially that the cell barcode is pre-filtered)
    """

    if method == "single_cell":
        cell_id_list = list(set(df_input["cell_id"]))
        df_list = []
        for j in tqdm(range(len(cell_id_list))):
            cell_id_temp = cell_id_list[j]
            df_temp = df_input[df_input["cell_id"] == cell_id_temp]
            read_cutoff = np.max(
                [base_read_cutoff, read_cutoff_ratio * np.max(df_temp["read"])]
            )
            df_temp_1 = df_temp[df_temp["read"] >= read_cutoff]
            if len(df_temp_1) > 0:
                for key_temp in target_keys:
                    clustered_umis, umi_count, new_seq_list = denoise_sequence(
                        df_temp_1[key_temp], threshold=threshold
                    )
                    df_temp_1.loc[:, key_temp] = new_seq_list
                df_list.append(df_temp_1)
        df_HQ = pd.concat(df_list)
    else:
        df_HQ = df_input[(df_input.read >= base_read_cutoff)]
        for key_temp in target_keys:
            print(
                f"Currently cleaning {key_temp}; number of unique elements: {len(set(df_HQ[key_temp]))}"
            )
            clustered_umis, umi_count, new_seq_list = denoise_sequence(
                df_HQ[key_temp], threshold=threshold
            )
            df_HQ[key_temp] = new_seq_list
            print(
                f"Number of unique elements (after cleaning): {len(set(df_HQ[key_temp]))}"
            )

    df_out = df_HQ.groupby(["library", "cell_id", "clone_id"]).sum("read")
    df_out["umi_count"] = (
        df_HQ.groupby(["library", "cell_id", "clone_id"])["umi"].count().values
    )
    df_out = df_out.reset_index()
    return df_out


def denoise_sequence(seq_list, threshold=None):
    seq_list = np.array(seq_list).astype(bytes)
    if threshold is None:
        threshold = round(0.1 * len(seq_list[0]))
    print(f"Sequences within Hamming distance {threshold} are connected")
    umi_count = {xx: np.sum(seq_list == xx) for xx in set(seq_list)}

    clusterer = UMIClusterer(cluster_method="directional")
    clustered_umis = clusterer(umi_count, threshold=threshold)
    correction_dict = {}
    for umi_list in clustered_umis:
        for umi in umi_list:
            correction_dict[umi] = umi_list[0]

    new_seq_list = [correction_dict[xx] for xx in seq_list]
    return correction_dict, umi_count, np.array(new_seq_list).astype(str)


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


def plot_seq_distance(distance, log_scale=True):
    np.fill_diagonal(distance, np.inf)
    fig, ax = plt.subplots()
    min_distance = distance.min(axis=1)
    sns.histplot(min_distance, bins=100, ax=ax)
    if log_scale:
        plt.yscale("log")
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


def QC_clonal_bc_per_cell(df0, read_cutoff=3, plot=True):
    """
    Get Number of clonal bc per cell
    """
    clonal_bc_N = []
    tot_read = []
    bc_list = []
    df = df0[df0["read"] >= read_cutoff]
    for bc in list(set(df["cell_id"])):
        df_temp = df[df.cell_id.isin([bc])]
        if len(df_temp) > 0:
            clonall_bc_N_temp = len(set(df_temp["clone_id"]))
            clonal_bc_N.append(clonall_bc_N_temp)
            bc_list.append(bc)
            tot_read.append(df_temp["read"].sum())

    df_statis = pd.DataFrame(
        {"cell_id": bc_list, "tot_read": tot_read, "clonal_bc_number": clonal_bc_N}
    )
    if plot:
        ax = sns.histplot(data=df_statis, x="clonal_bc_number")
        ax.set_xlabel("Number of clonal bc per cell")
        ax.set_ylabel("Count")
    return df_statis


def QC_clone_size(df0, read_cutoff=3, log_scale=False, plot=True):
    clone_size = []
    bc_list = []
    df = df0[df0["read"] >= read_cutoff]
    for bc in list(set(df["clone_id"])):
        df_temp = df[df.clone_id.isin([bc])]
        clone_size_temp = len(set(df_temp["cell_id"]))
        clone_size.append(clone_size_temp)
        bc_list.append(bc)

    df_statis = pd.DataFrame({"clone_id": bc_list, "clone_size": clone_size})
    if plot:
        ax = sns.histplot(data=df_statis, x="clone_size")
        ax.set_xlabel("Clone size")
        ax.set_ylabel("Count")
        if log_scale:
            plt.xscale("log")
            plt.yscale("log")
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
            l = f.readline().decode("utf-8").strip("\n")

            current_tag = []
            i = 0
            print("Reading in all barcodes")
            while not (l == "" and len(current_tag) == 0):
                i += 1
                if i % (3 * 10 ** 6) == 0:
                    print("Processed " + repr(int(i / 3)) + " reads")
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

                l = f.readline().decode("utf-8").strip("\n")

            sample_id = [k[0] for k, v in counts.items()]
            cell_bc = [k[1] for k, v in counts.items()]
            umi_id = [k[2] for k, v in counts.items()]
            gfp_bc_id = [k[3] for k, v in counts.items()]
            read_count = [v for k, v in counts.items()]
            library_id = [lib for _ in range(len(sample_id))]

            data_table = pd.DataFrame(
                {"library": library_id, "umi": umi_id, "cell_bc": cell_bc}
            )
            data_table["umi_id"] = [
                cell_bc[i] + "_" + umi_id[i] for i in range(len(umi_id))
            ]
            data_table["cell_id"] = [
                library_id[i] + "_" + cell_bc[i] for i in range(len(umi_id))
            ]
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
    print(
        f"Current cell number: {len(set(df_tmp['cell_id']))}\n"
        f"current clone number: {len(set(df_tmp['clone_id']))}\n"
        f"total reads: {np.sum(df_tmp['read'])/1000:.0f}K\n"
    )


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
