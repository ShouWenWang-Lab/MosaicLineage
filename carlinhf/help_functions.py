## This is used to put functions that are so specific that would not be very useful in other context
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
from matplotlib import pyplot as plt
from scipy.io import loadmat

from . import larry, lineage


def load_all_samples_to_adata(
    SampleList, file_path, df_ref, frequuency_cutoff=10 ** (-4), mode="allele"
):
    """
    mode: allele or mutation
    """
    tmp_list = []
    for sample in sorted(SampleList):
        base_dir = os.path.join(file_path, f"{sample}")
        df_tmp = lineage.load_allele_info(base_dir)
        # print(f"Sample (before removing frequent alleles): {sample}; allele number: {len(df_tmp)}")
        df_tmp["sample"] = sample.split("_")[0]
        df_tmp["mouse"] = sample.split("-")[0]
        tmp_list.append(df_tmp)
    df_all_0 = pd.concat(tmp_list).rename(columns={"UMI_count": "obs_UMI_count"})
    df_all = lineage.query_allele_frequencies(df_ref, df_all_0)

    print("Clone number (before correction): {}".format(len(set(df_all["allele"]))))
    print("Cell number (before correction): {}".format(len(df_all["allele"])))
    df_HQ = df_all[df_all.expected_frequency < frequuency_cutoff]
    print("Clone number (after correction): {}".format(len(set(df_HQ["allele"]))))
    print("Cell number (after correction): {}".format(len(df_HQ["allele"])))

    if mode == "mutation":
        adata_orig = lineage.generate_adata(df_all)
    else:
        adata_orig = lineage.generate_adata_sample_by_allele(df_HQ)
    return adata_orig


def custom_conditional_heatmap_v0(
    coarse_X_clone, fate_names, target_fate, conditional_fates
):
    fate_names = np.array(fate_names)
    sel_id = np.nonzero(fate_names == target_fate)[0][0]
    valid_clone_idx = coarse_X_clone[fate_names == target_fate].sum(0) > 0
    for x_name in conditional_fates:
        valid_clone_idx_tmp = coarse_X_clone[fate_names == x_name].sum(0) > 0
        new_matrix = coarse_X_clone[:, valid_clone_idx & valid_clone_idx_tmp]
        X_count, norm_X_count = lineage.get_fate_count_coupling(new_matrix)
        cs.pl.heatmap(
            new_matrix.T,
            order_map_x=False,
            x_ticks=None,
            log_transform=True,
            fig_width=10,
            fig_height=2,
        )
        plt.title(
            f"{np.sum(valid_clone_idx & valid_clone_idx_tmp)} clones; Fraction: LK {norm_X_count[sel_id,3]:.2f}; B {norm_X_count[sel_id,4]:.1f}; Mo {norm_X_count[sel_id,5]:.2f} G {norm_X_count[sel_id,6]:.2f}; oth {norm_X_count[sel_id,7]:.2f}"
        )
        valid_clone_idx = valid_clone_idx & ~valid_clone_idx_tmp
        # print(f"number {x_name}: {np.sum(valid_clone_idx)}")

    new_matrix = coarse_X_clone[:, valid_clone_idx]
    X_count, norm_X_count = lineage.get_fate_count_coupling(new_matrix)
    cs.pl.heatmap(
        new_matrix.T,
        order_map_x=False,
        x_ticks=fate_names,
        log_transform=True,
        fig_width=10,
        fig_height=2,
    )
    plt.title(
        f"{np.sum(valid_clone_idx)} clones; Fraction: LK {norm_X_count[sel_id,3]:.2f}; B {norm_X_count[sel_id,4]:.1f};  Mo {norm_X_count[sel_id,5]:.2f} G {norm_X_count[sel_id,6]:.2f}; oth {norm_X_count[sel_id,7]:.2f}"
    )


def custom_conditional_heatmap(
    coarse_X_clone,
    fate_names,
    target_fate,
    conditional_fates,
    exclude_fates=None,
    only_LK=False,
    final_x_ticks=True,
):
    """
    only_LK: None, False, True
        None -> do not partition the clone according to the presence of LK
        False -> show both LK+ and LK- clones
        True -> only show LK+ clones
    """

    fate_names = np.array(fate_names)
    partition_fate = [x for x in fate_names if "LK" in x][0]
    sel_id = np.nonzero(fate_names == target_fate)[0][0]
    valid_clone_idx = coarse_X_clone[fate_names == target_fate].sum(0) > 0
    if exclude_fates is not None:
        for x in exclude_fates:
            valid_clone_idx_tmp = ~(coarse_X_clone[fate_names == x].sum(0) > 0)
            valid_clone_idx = valid_clone_idx & valid_clone_idx_tmp

    if only_LK is not None:
        valid_clone_idx_tmp_2 = coarse_X_clone[fate_names == partition_fate].sum(0) > 0
    else:
        valid_clone_idx_tmp_2 = np.ones(coarse_X_clone.shape[1]).astype(bool)
        only_LK = True

    temp_idx_list = []
    for j, x_name in enumerate(conditional_fates):
        valid_clone_idx_tmp = coarse_X_clone[fate_names == x_name].sum(0) > 0
        if (j == 0) and (len(conditional_fates) > 1):
            valid_clone_idx_tmp_1 = (
                coarse_X_clone[fate_names == conditional_fates[1]].sum(0) > 0
            )
            temp_idx = (
                valid_clone_idx
                & valid_clone_idx_tmp
                & valid_clone_idx_tmp_1
                & valid_clone_idx_tmp_2
            )
            temp_idx_list.append(temp_idx)
            if not only_LK:
                temp_idx = (
                    valid_clone_idx
                    & valid_clone_idx_tmp
                    & valid_clone_idx_tmp_1
                    & (~valid_clone_idx_tmp_2)
                )
                temp_idx_list.append(temp_idx)

            new_idx = valid_clone_idx & valid_clone_idx_tmp & (~valid_clone_idx_tmp_1)
        else:
            new_idx = valid_clone_idx & valid_clone_idx_tmp

        temp_idx = new_idx & valid_clone_idx_tmp_2
        temp_idx_list.append(temp_idx)
        if not only_LK:
            temp_idx = new_idx & (~valid_clone_idx_tmp_2)
            temp_idx_list.append(temp_idx)

        valid_clone_idx = valid_clone_idx & (~valid_clone_idx_tmp)

    temp_idx = valid_clone_idx & valid_clone_idx_tmp_2
    temp_idx_list.append(temp_idx)
    if not only_LK:
        temp_idx = valid_clone_idx & (~valid_clone_idx_tmp_2)
        temp_idx_list.append(temp_idx)

    short_fate_names = [x.split("-")[-1] for x in fate_names]
    for j, temp_idx in enumerate(temp_idx_list):
        if np.sum(temp_idx) == 0:
            new_matrix = np.zeros((coarse_X_clone.shape[0], 1))
        else:
            new_matrix = coarse_X_clone[:, temp_idx]
        X_count, norm_X_count = lineage.get_fate_count_coupling(new_matrix)
        if (j == len(temp_idx_list) - 1) and final_x_ticks:
            x_ticks = fate_names
        else:
            x_ticks = None
        cs.pl.heatmap(
            new_matrix.T,
            order_map_x=False,
            order_map_y=True,
            x_ticks=x_ticks,
            log_transform=True,
            fig_width=10,
            fig_height=1,
        )
        des = f"{np.sum(temp_idx)} clones;"
        for j, x in enumerate(short_fate_names[1:]):
            des = des + f" {x} {norm_X_count[sel_id,j+1]:.2f}"
        plt.title(des)
        # plt.title(
        #     f"{np.sum(temp_idx)} clones; {short_fate_names[3]} {norm_X_count[sel_id,3]:.2f}; {short_fate_names[4]} {norm_X_count[sel_id,4]:.2f}; {short_fate_names[5]} {norm_X_count[sel_id,5]:.2f} {short_fate_names[6]} {norm_X_count[sel_id,6]:.2f}; {short_fate_names[7]} {norm_X_count[sel_id,7]:.2f}"
        # )


def custom_fate_bias_heatmap(
    coarse_X_clone,
    fate_names,
    conditional_fates=["LL405-E5-LT-HSC", "LL405-E5-ST-HSC", "LL405-E5-MPP3-4"],
    only_LK=True,
):
    """
    only_LK: None, False, True
        None -> do not partition the clone according to the presence of LK
        False -> show both LK+ and LK- clones
        True -> only show LK+ clones
    """
    custom_conditional_heatmap(
        coarse_X_clone,
        fate_names,
        target_fate=conditional_fates[0],
        conditional_fates=conditional_fates[1:],
        only_LK=only_LK,
        final_x_ticks=False,
        exclude_fates=None,
    )
    custom_conditional_heatmap(
        coarse_X_clone,
        fate_names,
        target_fate=conditional_fates[1],
        conditional_fates=conditional_fates[2:],
        only_LK=only_LK,
        final_x_ticks=False,
        exclude_fates=[conditional_fates[0]],
    )
    custom_conditional_heatmap(
        coarse_X_clone,
        fate_names,
        target_fate=conditional_fates[2],
        conditional_fates=[],
        only_LK=only_LK,
        final_x_ticks=True,
        exclude_fates=conditional_fates[:2],
    )


def reverse_compliment(seq):
    reverse = np.array(list(seq))[::-1]
    map_seq = {"A": "T", "C": "G", "T": "A", "G": "C"}
    complement = "".join([map_seq[x] for x in reverse])
    return complement


def read_quality_checks(
    path_to_fastq, UMI_length, primer3_length=20, primer5_length=20
):

    f = open(path_to_fastq, "r")
    data = f.readlines()
    seq = []
    for j in range(int(len(data) / 4)):
        seq.append(data[4 * j + 1].strip("\n"))

    df = pd.DataFrame({"seq": seq})
    df["UMI"] = df["seq"].apply(lambda x: x[:UMI_length])
    df["3primer"] = df["seq"].apply(
        lambda x: x[UMI_length : (UMI_length + primer3_length)]
    )
    df["CARLIN"] = df["seq"].apply(
        lambda x: x[(UMI_length + primer3_length) : -primer5_length]
    )
    df["5primer"] = df["seq"].apply(lambda x: x[-primer5_length:])

    df_temp = (
        df.groupby("3primer")
        .agg({"3primer": "count"})
        .rename(columns={"3primer": "3primer_count"})
    )
    df_3 = df_temp.sort_values("3primer_count").reset_index()
    primer_3_fraction = df_3["3primer_count"].max() / df_3["3primer_count"].sum()

    df_temp = (
        df.groupby("5primer")
        .agg({"5primer": "count"})
        .rename(columns={"5primer": "5primer_count"})
    )
    df_5 = df_temp.sort_values("5primer_count").reset_index()
    primer_5_fraction = df_5["5primer_count"].max() / df_5["5primer_count"].sum()

    print(f"Top_1 primer_3 fraction: {primer_3_fraction}")
    print(f"Top_1 primer_5 fraction: {primer_5_fraction}")
    return df, df_3, df_5


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


def merge_samples_with_sample_count(
    target_data_list,
    read_cutoff=3,
    root_path="/Users/shouwenwang/Dropbox (HMS)/shared_folder_with_Li/DATA/CARLIN",
):
    """
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

                    df_allele = snakehf.load_allele_frequency_over_multiple_samples(
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


def extract_CARLIN_info(
    data_path,
    SampleList,
    df_ref,
    short_names=None,
    source=None,
):
    """
    Extract CARLIN information

    data_path:
        The root dir to all the samples
    short_names: list
        THe sname length as SampleList, one-to-one correspondence
    df_ref: pd.DataFrame
        The allele bank
    source:
        Used to modify SampleList on the fly, as a supfix
    """
    selected_fates = []
    short_names_mock = []
    Flat_SampleList = []
    for x in SampleList:
        if type(x) is list:
            sub_list = [y.split("_")[0] for y in x]
            selected_fates.append(sub_list)
            short_names_mock.append("_".join(sub_list))
            Flat_SampleList += x
        else:
            selected_fates.append(x.split("_")[0])
            short_names_mock.append(x.split("_")[0])
            Flat_SampleList.append(x)

    if short_names is None:
        short_names = short_names_mock

    tmp_list = []
    for sample in sorted(Flat_SampleList):
        if source is not None:
            sample_file = sample + "." + source
        else:
            sample_file = sample

        base_dir = os.path.join(data_path, sample_file)
        df_tmp = lineage.load_allele_info(base_dir)
        # print(f"Sample (before removing frequent alleles): {sample}; allele number: {len(df_tmp)}")
        df_tmp["sample"] = sample.split("_")[0]
        df_tmp["mouse"] = sample.split("-")[0]

        df_allele = pd.read_csv(
            data_path + f"/{sample_file}/AlleleAnnotations.txt",
            sep="\t",
            header=None,
            names=["allele"],
        )
        df_CB = pd.read_csv(
            data_path + f"/{sample_file}/AlleleColonies.txt",
            sep="\t",
            header=None,
            names=["CB"],
        )
        df_allele["CB"] = df_CB
        df_allele["CB_N"] = df_allele["CB"].apply(lambda x: len(x.split(",")))

        if os.path.exists(data_path + f"/{sample_file}/Actaul_CARLIN_seq.txt"):
            df_CARLIN = pd.read_csv(
                data_path + f"/{sample_file}/Actaul_CARLIN_seq.txt",
                sep="\t",
                header=None,
                names=["CARLIN"],
            )
            df_allele["CARLIN"] = df_CARLIN

        df_tmp = df_tmp.merge(df_allele, on="allele")
        tmp_list.append(df_tmp)
    df_all = (
        pd.concat(tmp_list)
        .rename(columns={"UMI_count": "obs_UMI_count"})
        .merge(df_ref, on="allele", how="left")
    )
    # df_HQ = df_all_0.query("invalid_alleles!=True")
    return df_all
