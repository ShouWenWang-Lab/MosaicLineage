import os
from random import sample
from re import A
from shutil import SameFileError

import cospar as cs
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from matplotlib import rcParams
from nbformat import read

import carlinhf.CARLIN as car
import carlinhf.help_functions as hf
import carlinhf.lineage as lineage

cs.settings.set_figure_params(format="pdf", figsize=[4, 3.5], dpi=150, fontsize=14)
rcParams["legend.handlelength"] = 1.5

###############

# This file contains lousy functions that are often from one-off analysis.
# It differs from plot_scripts that they often return 
# useful object for further analysis,
# while plot_scripts primarily ends with the plotting

################


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
    df_all_0 = pd.concat(tmp_list)
    df_all = lineage.query_allele_frequencies(df_ref, df_all_0)

    print("Clone number (before correction): {}".format(len(set(df_all["allele"]))))
    print("Cell number (before correction): {}".format(len(df_all["allele"])))
    df_HQ = df_all[df_all.expected_frequency < frequuency_cutoff]
    print("Clone number (after correction): {}".format(len(set(df_HQ["allele"]))))
    print("Cell number (after correction): {}".format(len(df_HQ["allele"])))

    if mode == "mutation":
        adata_orig = lineage.generate_adata_allele_by_mutation(df_all)
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





def load_and_annotate_sc_CARLIN_data(
    SC_CARLIN_dir,
    Bulk_CARLIN_dir,
    target_sample,
    plate_map=None,
    locus="CC",
    ref_dir="/Users/shouwen/Dropbox (HMS)/shared_folder_with_Li/Analysis/CARLIN/data",
    BC_max_sample_count=6,
    BC_max_freq=10 ** (-4),
    read_cutoff=10,
):
    """
    A lazy function(scipt) to load, filter, and annotate the single-cell CARLIN data generated by SW pipeline

    1, load the single-cell CARLIN analysis results from SW method (with SC_CARLIN_dir + target_sample)
    2, convert CARLIN sequence to allele annotation using the bulk data (need the Bulk_CARLIN_dir + target_sample)
    3, annotate sample information, including plate_ID, mouse ID etc. (need plate_map)
    4, load allele bank and merge with it to add expected frequency and co-occurence across independent samples
    """
    print(
        f"locus: {locus}; read cutoff: {read_cutoff}; BC_max_sample_count: {BC_max_sample_count}; BC_max_freq: {BC_max_freq}"
    )

    # load all CARLIN data across samples
    SampleList = car.get_SampleList(SC_CARLIN_dir + f"/{locus}")
    df_list = []
    for sample in SampleList:
        df_sc_tmp = pd.read_csv(
            f"{SC_CARLIN_dir}/{locus}/CARLIN/Shouwen_Method/{sample}/called_barcodes_by_SW_method.csv"
        )
        df_list.append(df_sc_tmp)
    df_sc_data = pd.concat(df_list, ignore_index=True)

    # convert CARLIN sequence to allele annotation using the bulk data
    df_tmp = (
        pd.read_csv(f"{Bulk_CARLIN_dir}/{locus}_CARLIN_{target_sample}_all.csv")
        .filter(["allele", "CARLIN"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    CARLIN_to_allel_map = dict(zip(df_tmp["CARLIN"], df_tmp["allele"]))
    df_sc_data["allele"] = (
        df_sc_data["clone_id"].map(CARLIN_to_allel_map).fillna("unmapped")
    )

    # annotate sample information
    if "library" in df_sc_data.columns:
        df_sc_data["sample"] = df_sc_data["library"].apply(lambda x: x.split("_")[0])

    df_sc_data["locus"] = df_sc_data["sample"].apply(lambda x: x[-2:])
    df_sc_data["mouse"] = df_sc_data["sample"].apply(lambda x: x.split("-")[0])
    df_sc_data["plate_ID"] = df_sc_data["sample"].apply(lambda x: x[:-3])
    if plate_map is not None:
        df_sc_data["plate_ID"] = df_sc_data["plate_ID"].map(plate_map)

    df_sc_data["RNA_id"] = df_sc_data["plate_ID"] + "_RNA_" + df_sc_data["cell_bc"]
    df_sc_data["clone_id"] = df_sc_data["locus"] + "_" + df_sc_data["clone_id"]
    df_sc_data["allele"] = df_sc_data["locus"] + "_" + df_sc_data["allele"]
    df_sc_data = df_sc_data[df_sc_data.read >= read_cutoff]

    # add expected allele frequency, and filter out promiscuous clones
    df_ref = pd.read_csv(f"{ref_dir}/reference_merged_alleles_{locus}.csv").filter(
        ["allele", "expected_frequency", "sample_count"]
    )
    df_sc_data = df_sc_data.merge(df_ref, on="allele", how="left").fillna(0)

    df_sc_data = df_sc_data.assign(
        HQ=lambda x: (x["sample_count"] <= BC_max_sample_count)
        & (x["expected_frequency"] <= BC_max_freq)
    ).query("HQ==True")

    return df_sc_data

    # add clonal fate outcome
    df_clone_fate = pd.read_csv(
        f"/Users/shouwen/Dropbox (HMS)/shared_folder_with_Li/Analysis/multi-omic-analysis/202205_in_vivo_bone_marrow/data/bulk_CARLIN_clonal_fate_{locus}.csv"
    )
    df_sc_data = df_sc_data.merge(df_clone_fate, on="allele", how="left")
    df_sc_data["fate"] = df_sc_data["fate"].fillna("no_fates")

    return df_sc_data, df_sc_data.filter(
        ["RNA_id", "clone_id", "fate", "CARLIN_length", "read", "fate_N"]
    )
