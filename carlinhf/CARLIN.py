## This is used to put functions that are so specific that would not be very useful in other context
import gzip
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
import source.help_functions as snakehf
import yaml
from Bio import SeqIO
from matplotlib import pyplot as plt
from scipy.io import loadmat

from . import larry, lineage

#########################################

## We put CARLIN-specific operations here

#########################################


# define seqeunces and primers for QC. For each 5' sequence, we skip the first 2 bp, and for 3' sequence, we skip the last 2 bp, as they are prone to errors
CC_5prime = "AGCTGTACAAGTAAGCGGC"
CC_3prime = "AGAATTCTAACTAGAGCTCGCTGATCAGCCTCGACTGTGCCTTCT"
CC_CARLIN = "CGCCGGACTGCACGACAGTCGACGATGGAGTCGACACGACTCGCGCATACGATGGAGTCGACTACAGTCGCTACGACGATGGAGTCGCGAGCGCTATGAGCGACTATGGAGTCGATACGATACGCGCACGCTATGGAGTCGAGAGCGCGCTCGTCGACTATGGAGTCGCGACTGTACGCACACGCGATGGAGTCGATAGTATGCGTACACGCGATGGAGTCGAGTCGAGACGCTGACGATATGGAGTCGATACGTAGCACGCAGACGATGGGAGCT"

TC_5prime = "TCGGTACCTCGCGAA"
TC_3prime = "GTCTTGTCGGTGCCTTCTAGTT"
TC_CARLIN = "TCGCCGGAGTCGAGACGCTGACGATATGGAGTCGACACGACTCGCGCATACGATGGAGTCGCGAGCGCTATGAGCGACTATGGAGTCGATAGTATGCGTACACGCGATGGAGTCGACTACAGTCGCTACGACGATGGAGTCGATACGATACGCGCACGCTATGGAGTCGCGACTGTACGCACACGCGATGGAGTCGACTGCACGACAGTCGACGATGGAGTCGATACGTAGCACGCAGACGATGGGAGCGAGAGCGCGCTCGTCGACTATGGA"

RC_5prime = "GTACAAGTAAAGCGGCC"
RC_3prime = "GTCTGCTGTGTGCCTTCTAGTT"
RC_CARLIN = "GCGCCGGCGAGCGCTATGAGCGACTATGGAGTCGACACGACTCGCGCATACGATGGAGTCGACTACAGTCGCTACGACGATGGAGTCGATACGATACGCGCACGCTATGGAGTCGACTGCACGACAGTCGACGATGGAGTCGATACGTAGCACGCAGACGATGGGAGCGAGTCGAGACGCTGACGATATGGAGTCGATAGTATGCGTACACGCGATGGAGTCGCGACTGTACGCACACGCGATGGAGTCGAGAGCGCGCTCGTCGACTATGGA"


def consensus_sequence(df):
    X = np.array([np.array(bytearray(x, encoding="utf8")) for x in df])
    return bytes(np.median(X, axis=0).astype("uint8")).decode("utf8")


def obtain_read_dominant_sequences(
    df_input, cell_bc_key="cell_bc", clone_key="clone_id"
):
    """
    Find the candidate sequence with the max read count within each group
    """

    df_input["CARLIN_length"] = df_input[clone_key].apply(lambda x: len(x))
    df_CARLIN_lenth = (
        df_input.groupby([cell_bc_key, clone_key, "CARLIN_length"])
        .agg(read=("read", "sum"))
        .reset_index()
    )

    df_dominant_fraction = (
        df_CARLIN_lenth.groupby([cell_bc_key])
        .agg(
            read=("read", "max"),
            max_read_ratio=("read", lambda x: np.max(x) / np.sum(x)),
        )
        .reset_index()
    )

    df_out = df_CARLIN_lenth.merge(
        df_dominant_fraction, on=[cell_bc_key, "read"], how="inner"
    )
    return df_input.drop(["read", "CARLIN_length"], axis=1).merge(
        df_out, on=[cell_bc_key, clone_key]
    )


def CARLIN_analysis(
    df_input, cell_bc_key="cell_bc", clone_key="clone_id", read_ratio_threshold=0.6
):
    """
    This function is similar to the CARLIN pipeline, that for each tag, we find the dominant CARLIN sequence as the right candidate.
    At the moment, I believe that second part (obtain consensus sequence is not used, as there is only one sequence left after the
    dominant sequence selection.
    """

    # df_dominant_fraction=calculate_read_fraction_for_dominant_sequences(df_input,cell_bc_key=cell_bc_key,clone_key=clone_key)
    df_tmp = obtain_read_dominant_sequences(
        df_input, cell_bc_key=cell_bc_key, clone_key=clone_key
    )
    df_final = df_tmp[df_tmp["max_read_ratio"] > read_ratio_threshold]

    # obtain consensus sequences
    df_final = df_final.groupby(cell_bc_key).agg(
        consensuse_CARLIN=(clone_key, consensus_sequence), read=("read", "sum")
    )
    df_final["CARLIN_length"] = df_final["consensuse_CARLIN"].apply(lambda x: len(x))
    return df_final


def CARLIN_raw_reads(data_path, sample, protocol="scLimeCat"):
    """
    Load raw fastq files. This function will depend on what protocol is used.
    """
    supported_protocol = ["scLimeCat"]
    if not (protocol in supported_protocol):
        raise ValueError(f"Only support protocols: {supported_protocol}")

    if protocol == "scLimeCat":
        seq_list = []
        with gzip.open(f"{data_path}/{sample}_L001_R1_001.fastq.gz", "rt") as handle:
            for record in SeqIO.parse(handle, "fastq"):
                seq_list.append(str(record.seq))

        tag_list = []
        with gzip.open(f"{data_path}/{sample}_L001_R2_001.fastq.gz", "rt") as handle:
            for record in SeqIO.parse(handle, "fastq"):
                tag_list.append(str(record.seq))

        df_seq = pd.DataFrame({"Tag": tag_list, "Seq": seq_list})
        df_seq["cell_bc"] = df_seq["Tag"].apply(lambda x: x[:8])
        df_seq["library"] = sample
        df_seq["cell_id"] = df_seq["library"] + "_" + df_seq["cell_bc"]
        df_seq["umi"] = df_seq["Tag"].apply(lambda x: x[8:16])
        df_seq["umi_id"] = df_seq["cell_bc"] + "_" + df_seq["umi"]
        df_seq["clone_id"] = df_seq["Seq"]

    return df_seq.drop("Tag", axis=1)


def CARLIN_preprocessing(
    df_input,
    template="cCARLIN",
    ref_cell_barcodes=None,
    seq_5prime_upper_N=None,
    seq_3prime_upper_N=None,
):
    """
    Filter the raw reads. This pipeline should be independent of whether this is bulk or single-cell CARLIN

    Parameters
    ----------
    df_input: pd.DataFrame
        input data, from CARLIN_raw_reads
    template: str
        {'cCARLIN','Tigre','Rosa'}
    ref_cell_barcodes:
        Reference cell barcode list, for filtering
    seq_5prime_upper_N:
        Control the number of 5prime bps for QC. Default: use all bps.
    seq_3prime_upper_N:
        Control the number of 5prime bps for QC. Default: use all bps.

    Returns
    -------
    df_output:
        A dataframe of sequences that pass QC
    """

    # seq_5prime:
    #     5' end sequences, for QC. Only reads contain exactly this sequence will pass QC.
    #     The end of the 5' end sequences mark the beginning of CARLIN sequences.
    # seq_3prime:
    # 3' end sequences, for QC. Only reads contain exactly this sequence will pass QC.
    #     The beginning of the 3' end sequences mark the end of CARLIN sequences.

    if template.startswith("cCARLIN"):
        seq_5prime = CC_5prime
        seq_3prime = CC_3prime
        CARLIN_seq = CC_CARLIN
    elif template.startswith("Tigre"):
        seq_5prime = TC_5prime
        seq_3prime = TC_3prime
        CARLIN_seq = TC_CARLIN
    elif template.startswith("Rosa"):
        seq_5prime = RC_5prime
        seq_3prime = RC_3prime
        CARLIN_seq = RC_CARLIN
    else:
        raise ValueError("template must start with {'cCARLIN','Tigre','Rosa'}")

    if seq_5prime_upper_N is not None:
        seq_5prime = seq_5prime[-seq_5prime_upper_N:]
    if seq_3prime_upper_N is not None:
        seq_3prime = seq_3prime[:seq_3prime_upper_N]

    df_output = df_input.copy()
    df_output["Valid"] = df_output["Seq"].apply(
        lambda x: (seq_5prime in x) & (seq_3prime in x)
    )
    tot_fastq_N = len(df_output)
    print("Total fastq:", tot_fastq_N)
    df_output = df_output.query("Valid==True")
    valid_3_5_prime_N = len(df_output)
    print(
        f"Fastq with vaid 3 and 5 prime: {valid_3_5_prime_N} ({valid_3_5_prime_N/tot_fastq_N:.2f})"
    )
    if ref_cell_barcodes is not None:
        df_output = df_output[df_output["cell_bc"].isin(ref_cell_barcodes)]
        valid_BC_N = len(df_output)
        print(f"Fastq with valid barcodes: {valid_BC_N} ({valid_BC_N/tot_fastq_N:.2f})")

    df_output["clone_id"] = df_output["Seq"].apply(
        lambda x: x.split(seq_5prime)[1].split(seq_3prime)[0]
    )
    df_output["unique_id"] = (
        df_output["cell_id"] + "_" + df_output["umi_id"] + "_" + df_output["clone_id"]
    )
    df_tmp = (
        df_output.groupby("unique_id").agg(read=("unique_id", "count")).reset_index()
    )
    return (
        df_output.merge(df_tmp, on="unique_id")
        .drop(["Valid", "Seq", "unique_id"], axis=1)
        .drop_duplicates()
    )


def extract_CARLIN_info(
    data_path,
    SampleList,
):
    """
    Extract CARLIN information, like alleles, colonies, UMI count info

    data_path:
        The root dir to all the samples
    SampleList:
        The list of desired samples to load
    """

    tmp_list = []
    for sample in SampleList:
        base_dir = os.path.join(data_path, sample)
        df_tmp = lineage.load_allele_info(base_dir)
        df_tmp["sample"] = sample.split("_")[0]
        df_tmp["mouse"] = sample.split("-")[0]

        df_allele = pd.read_csv(
            data_path + f"/{sample}/AlleleAnnotations.txt",
            sep="\t",
            header=None,
            names=["allele"],
        )
        df_CB = pd.read_csv(
            data_path + f"/{sample}/AlleleColonies.txt",
            sep="\t",
            header=None,
            names=["CB"],
        )
        df_allele["CB"] = df_CB
        df_allele["CB_N"] = df_allele["CB"].apply(lambda x: len(x.split(",")))

        if os.path.exists(data_path + f"/{sample}/Actaul_CARLIN_seq.txt"):
            df_CARLIN = pd.read_csv(
                data_path + f"/{sample}/Actaul_CARLIN_seq.txt",
                sep="\t",
                header=None,
                names=["CARLIN"],
            )

            df_allele["CARLIN"] = df_CARLIN["CARLIN"].apply(
                lambda x: "".join(x.split("-"))
            )
            df_allele["CARLIN_length"] = df_allele["CARLIN"].apply(lambda x: len(x))

        df_tmp = df_tmp.merge(df_allele, on="allele")
        tmp_list.append(df_tmp)
    df_all = pd.concat(tmp_list)
    return df_all


def CARLIN_output_to_cell_by_barcode_long_table(df_input):
    """
    Convert output from extract_CARLIN_info (a wide table)
    to a two column (cell, and clone_id) long table.
    """
    CB_list = []
    CB_flat = []
    Clone_id_flat = []
    for j in range(len(df_input)):
        df_series = df_input.iloc[j]
        tmp = [x for x in df_series["CB"].split(",")]
        CB_flat += tmp
        Clone_id_flat += [df_series["CARLIN"] for _ in tmp]

    df_ref_flat = pd.DataFrame({"cell_bc": CB_flat, "clone_id": Clone_id_flat})

    return df_ref_flat


def get_SampleList(root_path):
    with open(f"{root_path}/config.yaml", "r") as stream:
        file = yaml.safe_load(stream)
        SampleList = file["SampleList"]
    return SampleList


def load_allele_info(data_path):
    pooled_data = loadmat(os.path.join(data_path, "allele_annotation.mat"))
    allele_freqs = pooled_data["allele_freqs"].flatten()
    alleles = [xx[0][0] for xx in pooled_data["AlleleAnnotation"]]
    return pd.DataFrame({"allele": alleles, "UMI_count": allele_freqs})


def load_allele_frequency_statistics(data_path: str, SampleList: list):
    """
    data_path: should be at the level of samples, e.g., path/to/results_read_cutoff_3

    Return an allele-grouped frequency count table
    """

    df_list = []
    for sample in SampleList:
        df_temp = load_allele_info(os.path.join(data_path, sample))
        df_list.append(df_temp)
        print(f"{sample}: {len(df_temp)}")
    df_raw = pd.concat(df_list).reset_index()
    df_raw["sample_count"] = 1
    df_new = df_raw.groupby("allele", as_index=False).agg(
        {"UMI_count": "sum", "sample_count": "sum"}
    )
    return df_new


def merge_three_locus(
    data_path_CC,
    data_path_RC,
    data_path_TC=None,
    sample_type_1="Col",
    sample_type_2="Rosa",
    sample_type_3="Tigre",
):

    df_CC = pd.read_csv(
        f"{data_path_CC}/merge_all/refined_results.csv", index_col=0
    ).sort_values("sample")
    df_CC = df_CC[df_CC["sample"] != "merge_all"]
    idx = np.argsort(df_CC["sample"])
    df_CC = df_CC.iloc[idx]
    df_CC["sample_id"] = np.arange(len(df_CC))
    df_CC["Type"] = sample_type_1
    df_RC = pd.read_csv(
        f"{data_path_RC}/merge_all/refined_results.csv", index_col=0
    ).sort_values("sample")
    df_RC = df_RC[df_RC["sample"] != "merge_all"]
    idx = np.argsort(df_RC["sample"])
    df_RC = df_RC.iloc[idx]
    df_RC["sample_id"] = np.arange(len(df_RC))
    df_RC["Type"] = sample_type_2

    x = "total_alleles"
    df_CC[f"{x}_norm_fraction"] = df_CC[x] / df_CC[x].sum()
    df_RC[f"{x}_norm_fraction"] = df_RC[x] / df_RC[x].sum()
    x = "singleton"
    df_CC[f"{x}_norm_fraction"] = df_CC[x] / df_CC[x].sum()
    df_RC[f"{x}_norm_fraction"] = df_RC[x] / df_RC[x].sum()

    if data_path_TC is not None:
        df_TC = pd.read_csv(
            f"{data_path_TC}/merge_all/refined_results.csv", index_col=0
        ).sort_values("sample")
        df_TC = df_TC[df_TC["sample"] != "merge_all"]
        idx = np.argsort(df_TC["sample"])
        df_TC = df_TC.iloc[idx]
        df_TC["sample_id"] = np.arange(len(df_TC))
        df_TC["Type"] = sample_type_3

        x = "total_alleles"
        df_TC[f"{x}_norm_fraction"] = df_TC[x] / df_TC[x].sum()
        x = "singleton"
        df_TC[f"{x}_norm_fraction"] = df_TC[x] / df_TC[x].sum()

        df_all = pd.concat([df_CC, df_RC, df_TC])
        df_sample_association = (
            df_CC.filter(["sample_id", "sample"])
            .merge(df_TC.filter(["sample_id", "sample"]), on="sample_id")
            .merge(df_RC.filter(["sample_id", "sample"]), on="sample_id")
        )
    else:
        df_all = pd.concat([df_CC, df_RC])
        df_sample_association = df_CC.filter(["sample_id", "sample"]).merge(
            df_RC.filter(["sample_id", "sample"]), on="sample_id"
        )

    df_sample_association = df_sample_association.rename(
        columns={"sample_x": "CC", "sample_y": "TC", "sample": "RC"}
    )
    return df_all, df_sample_association


def generate_allele_info_across_experiments(
    target_data_list,
    read_cutoff=3,
    root_path="/Users/shouwenwang/Dropbox (HMS)/shared_folder_with_Li/DATA/CARLIN",
):
    """
    Merge a given set of experiments at given read_cutoff.

    This is an operation spanning multiple different experiments

    target_data_list is a list of data name relative to root_dir
    An example:
    target_data_list=[
       '20220306_bulk_tissues/TC_DNA',
        '20220306_bulk_tissues/TC_RNA',
       '20220430_CC_TC_RC/TC']
    """
    df_list = []
    for sample in target_data_list:
        data_path = f"{root_path}/{sample}"
        with open(f"{data_path}/config.yaml", "r") as stream:
            file = yaml.safe_load(stream)
            SampleList = file["SampleList"]
            read_cutoff_override = file["read_cutoff_override"]

        for x in read_cutoff_override:
            if x == read_cutoff:
                print(f"---{sample}: cutoff: {x}")
                for sample_temp in SampleList:

                    input_dir = f"{data_path}/CARLIN/results_cutoff_override_{x}"

                    df_allele = load_allele_frequency_statistics(
                        input_dir, [sample_temp]
                    )
                    df_allele["sample"] = sample_temp
                    df_list.append(df_allele)

    df_merge = pd.concat(df_list, ignore_index=True)
    map_dict = {}
    for x in sorted(set(df_merge["sample"])):
        mouse = "LL" + x.split("LL")[1][:3]
        embryo_id = [f"E{j}" for j in range(20)]
        final_id = mouse
        for y in embryo_id:
            if y in x:
                final_id = final_id + "_" + y
                break
        # print(x, ":", final_id)
        map_dict[x] = final_id

    df_merge["sample_id"] = [map_dict[x] for x in list(df_merge["sample"])]

    df_group = (
        df_merge.groupby("allele")
        .agg(
            UMI_count=("UMI_count", "sum"),
            sample_count=("sample_id", lambda x: len(set(x))),
            sample_id=("sample_id", lambda x: set(x)),
        )
        .reset_index()
    )

    ## plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax = sns.histplot(data=df_group, x="UMI_count", log_scale=True)
    plt.yscale("log")
    singleton_fraction = np.sum(df_group["UMI_count"] == 1) / len(df_group)
    ax.set_title(f"Singleton frac.: {singleton_fraction:.3f}")
    ax.set_ylabel("Allele histogram")
    ax.set_xlabel("Observed frequency (UMI count)")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax = sns.histplot(data=df_group, x="sample_count", log_scale=True)
    plt.yscale("log")
    single_fraction = np.sum(df_group["sample_count"] == 1) / len(df_group)
    ax.set_title(f"Single sample frac.: {single_fraction:.3f}")
    ax.set_ylabel("Allele histogram")
    ax.set_xlabel("Occurence across samples")

    df_ref = df_group.rename(columns={"UMI_count": "expected_count"}).assign(
        expected_frequency=lambda x: x["expected_count"] / x["expected_count"].sum()
    )
    df_ref = df_ref.sort_values(
        "expected_count", ascending=False
    )  # .filter(["allele","expected_frequency",'sample_count'])
    return df_merge, df_ref, map_dict


# def read_quality_checks(
#     path_to_fastq, UMI_length, primer3_length=20, primer5_length=20
# ):

#     f = open(path_to_fastq, "r")
#     data = f.readlines()
#     seq = []
#     for j in range(int(len(data) / 4)):
#         seq.append(data[4 * j + 1].strip("\n"))

#     df = pd.DataFrame({"seq": seq})
#     df["UMI"] = df["seq"].apply(lambda x: x[:UMI_length])
#     df["3primer"] = df["seq"].apply(
#         lambda x: x[UMI_length : (UMI_length + primer3_length)]
#     )
#     df["CARLIN"] = df["seq"].apply(
#         lambda x: x[(UMI_length + primer3_length) : -primer5_length]
#     )
#     df["5primer"] = df["seq"].apply(lambda x: x[-primer5_length:])

#     df_temp = (
#         df.groupby("3primer")
#         .agg({"3primer": "count"})
#         .rename(columns={"3primer": "3primer_count"})
#     )
#     df_3 = df_temp.sort_values("3primer_count").reset_index()
#     primer_3_fraction = df_3["3primer_count"].max() / df_3["3primer_count"].sum()

#     df_temp = (
#         df.groupby("5primer")
#         .agg({"5primer": "count"})
#         .rename(columns={"5primer": "5primer_count"})
#     )
#     df_5 = df_temp.sort_values("5primer_count").reset_index()
#     primer_5_fraction = df_5["5primer_count"].max() / df_5["5primer_count"].sum()

#     print(f"Top_1 primer_3 fraction: {primer_3_fraction}")
#     print(f"Top_1 primer_5 fraction: {primer_5_fraction}")
#     return df, df_3, df_5
