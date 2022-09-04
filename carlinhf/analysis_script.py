import os

import cospar as cs
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from matplotlib import rcParams

import carlinhf.CARLIN as car
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


def generate_allele_info_across_experiments(
    target_data_list,
    read_cutoff=3,
    root_path="/Users/shouwenwang/Dropbox (HMS)/shared_folder_with_Li/DATA/CARLIN",
    mouse_label="LL",
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
            if "read_cutoff_UMI_override" in file.keys():
                read_cutoff_override = file["read_cutoff_UMI_override"]
            elif "read_cutoff_override" in file.keys():
                read_cutoff_override = file["read_cutoff_override"]

        for x in read_cutoff_override:
            if x == read_cutoff:
                print(f"---{sample}: cutoff: {x}")
                for sample_temp in SampleList:

                    input_dir = f"{data_path}/CARLIN/results_cutoff_override_{x}"

                    df_allele = car.load_allele_frequency_statistics(
                        input_dir, [sample_temp]
                    )
                    df_allele["sample"] = sample_temp
                    df_list.append(df_allele)

    df_merge = pd.concat(df_list, ignore_index=True)
    map_dict = {}
    for x in sorted(set(df_merge["sample"])):
        mouse = mouse_label + x.split(mouse_label)[1][:3]
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
            sample_id=("sample_id", lambda x: list(set(x))),
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
        expected_frequency=lambda x: x["expected_count"]
        / x["expected_count"].sum()
        * len(df_group)
    )
    df_ref = df_ref.sort_values(
        "expected_count", ascending=False
    )  # .filter(["allele","expected_frequency",'sample_count'])
    return df_merge, df_ref, map_dict


def load_and_annotate_sc_CARLIN_data(
    sc_root_path: str,
    bulk_data_path: str,
    sample_map: dict = None,
    plate_map: dict = None,
    locus: str = "CC",
    ref_dir: str = "/Users/shouwen/Dropbox (HMS)/shared_folder_with_Li/Analysis/CARLIN/data",
    sc_data_source: str = "SW",
):
    """
    A lazy function(scipt) to load, filter, and annotate the single-cell CARLIN data generated by SW pipeline

    1, load the single-cell CARLIN analysis results from SW method (with SC_CARLIN_dir + target_sample)
    2, convert CARLIN sequence to allele annotation using the bulk data (need the Bulk_CARLIN_dir + target_sample)
    3, annotate sample information, including plate_ID, mouse ID etc. (need plate_map)
    4, load allele bank and merge with it to add expected frequency and co-occurence across independent samples

    The bulk data is preprocess with `plot_script.analyze_cell_coupling`, where cell number normalization within each cluster
    is performed when possible. The final df_fate_matrix is therefore first cell-type-abundance-wise, then clone-wise normalized.

    Parameters
    ----------
    sc_root_path:
        root_path to point to the sample folder, e.g. path/CARLIN/Shouwen_Method
    bulk_data_path:
        path to sample folder, e.g. path/to/read_cutoff_override_3
    sample_map: dict
        A dictionary to rename samples
    plate_map: dict
        A dictionary to rename plates
    locus: str, {'CC','TC','RC'}
        CARLIN locus.
    ref_dir: str
        Allele bank reference directory. set to "/Users/shouwen/Dropbox (HMS)/shared_folder_with_Li/Analysis/CARLIN/data",
    sc_data_source:
        Data source of the bulk fate data.

    Returns
    -------
    df_sc_data:
        All single-cell allele annotation data. Note that it has been filtered according to {BC_max_sample_count,C_max_freq,read_cutoff}, but not yet intersected with the bulk fate outcome data
    df_clone_fate:
        The text anotated fate outcome for each clone using the corresponding bulk CARLIN. All the alleles detected in the bulk
        experiment are shown, i.e., it has not been intersected with the df_sc_data yet.
     df_fate_matrix:
        The clonal fate matrix from the bulk data, corresponding to df_clone_fate. It is first cell-type-abundance-wise, then clone-wise normalized.

    """

    if locus not in ["CC", "TC", "RC"]:
        raise ValueError("locus should be in {'CC','TC','RC'}")
    if sc_data_source not in ["SW", "joint"]:
        raise ValueError("sc_data_source be in {'SW','joint'}")

    # load all CARLIN data across samples
    SampleList = car.get_SampleList(f"{sc_root_path}/../..")
    df_list = []
    for sample in SampleList:
        if sc_data_source == "SW":
            file_path=f"{sc_root_path}/{sample}/called_barcodes_by_SW_method.csv"
            if os.path.exists(file_path):
                df_sc_tmp = pd.read_csv(
                    file_path
                )
            else:
                print(f'{file_path} does not exist. Skip!')
        else:
            print(
                "load allele data identified with either original and new (SW) method"
            )
            file_path=f"{sc_root_path}/{sample}/df_outer_joint.csv"
            if os.path.exists(file_path):
                df_sc_tmp = pd.read_csv(
                    file_path
                )
            else:
                print(f'{file_path} does not exist. Skip!')
        df_list.append(df_sc_tmp)
    df_sc_data = pd.concat(df_list, ignore_index=True)

    # convert CARLIN sequence to allele annotation using the bulk data
    df_all_fate = pd.read_csv(f"{bulk_data_path}/merge_all/df_allele_all.csv")
    CARLIN_to_allel_map = dict(zip(df_all_fate["CARLIN"], df_all_fate["allele"]))
    df_sc_data["allele"] = (
        df_sc_data["clone_id"].map(CARLIN_to_allel_map).fillna("unmapped")
    )

    # add expected allele frequency, and filter out promiscuous clones
    df_ref = pd.read_csv(f"{ref_dir}/reference_merged_alleles_{locus}.csv").filter(
        ["allele", "expected_frequency", "sample_count"]
    )
    df_sc_data = df_sc_data.merge(df_ref, on="allele", how="left").fillna(0)

    # annotate single-cell sample information
    df_sc_data["locus"] = locus
    df_sc_data = car.add_metadata(df_sc_data, plate_map=plate_map)

    ## convert the bulk fate information to clone-fate table, and then merge with the sc clonal data
    ## This step needs to happen after *annotate single-cell sample information* so that
    ## both alleles are annotated with locus information
    df_all_fate["locus"] = locus
    df_all_fate["allele"] = df_all_fate["locus"] + "_" + df_all_fate["allele"]
    if sample_map is not None:
        df_all_fate["sample"] = df_all_fate["sample"].map(sample_map).astype("category")
    (df_clone_fate, df_fate_matrix) = lineage.generate_clonal_fate_table(
        df_all_fate, thresh=0.1
    )
    # df_sc_data = df_sc_data.merge(df_clone_fate, on="allele", how="left")
    # df_sc_data["fate"] = df_sc_data["fate"].fillna("no_fates")
    # print("------unique fates---------", set(df_sc_data["fate"]))
    print("------expected frequency---------", set(df_sc_data["expected_frequency"]))

    return df_sc_data, df_clone_fate, df_fate_matrix


def merge_scCARLIN_to_bulk_CARLIN(
    df_sc_data,
    bulk_data_path,
    locus="CC",
    BC_max_sample_count: int = 6,
    BC_max_freq: float = 10 ** (-4),
    min_clone_size: int = 3,
):
    """
    Convert the single-cell CARLIN data to bulk like data, and
    merge it with the corresponding bulk dataset to perform bulk-like
    cell coupling analysis
    """
    sp_idx = df_sc_data["allele"].apply(lambda x: "_unmapped" in x)
    df_sc_data.loc[sp_idx, "allele"] = df_sc_data[sp_idx]["clone_id"]
    df_sc_bulk = (
        df_sc_data.groupby(["allele", "sample"])
        .agg(
            UMI_count=("cell_id", lambda x: len(set(x))),
            expected_frequency=("expected_frequency", "mean"),
            sample_count=("sample_count", "mean"),
        )
        .reset_index()
    )
    df_sc_bulk["source"] = "single-cell"

    df_merge_tmp = (
        pd.read_csv(f"{bulk_data_path}/merge_all/df_allele_all.csv")
        .filter(
            [
                "allele",
                "sample",
                "UMI_count",
                "expected_frequency",
                "sample_count",
                "clone_size",
            ]
        )
        .fillna(0)
    )
    df_merge_tmp["allele"] = f"{locus}_" + df_merge_tmp["allele"]
    df_merge_tmp["source"] = "bulk"
    df_merge_tmp = df_merge_tmp[df_merge_tmp["clone_size"] >= min_clone_size]

    df_merge = pd.concat([df_sc_bulk, df_merge_tmp], ignore_index=True)
    df_merge = df_merge[
        (df_merge["expected_frequency"] <= BC_max_freq)
        & (df_merge["sample_count"] <= BC_max_sample_count)
    ]
    return df_merge


#############################

# The rest are no longer used?

##############################


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


def load_single_cell_CARLIN(root_path, plate_map, read_cutoff=2, locus=None):
    SampleList = car.get_SampleList(root_path)
    data_path = os.path.join(
        root_path, "CARLIN", f"results_cutoff_override_{read_cutoff}"
    )
    df_out = car.extract_CARLIN_info(data_path, SampleList)

    df_out["locus"] = df_out["sample"].apply(lambda x: x[-2:])

    df_ref = pd.read_csv(
        f"/Users/shouwen/Dropbox (HMS)/shared_folder_with_Li/Analysis/CARLIN/data/reference_merged_alleles_{locus}.csv"
    ).filter(["allele", "expected_frequency", "sample_count"])
    df_out = df_out.merge(df_ref, on="allele", how="left")
    df_out["plate_ID"] = (
        df_out["sample"].apply(lambda x: x[:-3]).map(plate_map)
    )  # .astype('category')

    return df_out


def integrate_early_clone_and_fate(
    df_clone_fate_tmp,
    SC_CARLIN_dir,
    plate_map,
    locus="CC",
    BC_max_sample_count=5,
    BC_max_freq=10 ** (-4),
    read_cutoff=2,
):
    root_path = SC_CARLIN_dir + f"/{locus}"
    df_sc_tmp = load_single_cell_CARLIN(
        root_path, plate_map, locus=locus, read_cutoff=read_cutoff
    )
    df_sc_tmp = df_sc_tmp.merge(df_clone_fate_tmp, on="allele")
    df_sc_tmp["clone_id"] = f"{locus}_" + df_sc_tmp["allele"]

    df_final_tmp = (
        df_sc_tmp[df_sc_tmp["mouse"] == df_sc_tmp["fate_mouse"]]
        .fillna(0)
        .assign(
            HQ=lambda x: (x["sample_count"] <= BC_max_sample_count)
            & (x["expected_frequency"] <= BC_max_freq)
        )
        .query("HQ==True")
    )

    CB_list = []
    CB_flat = []
    Clone_id_flat = []
    for j in range(len(df_final_tmp)):
        df_series = df_final_tmp.iloc[j]
        plate_id = df_series["plate_ID"]
        tmp = [plate_id + "_RNA" + "_" + x for x in df_series["CB"].split(",")]
        CB_list.append(",".join(tmp))
        CB_flat += tmp
        Clone_id_flat += [df_series["clone_id"] for _ in tmp]

    df_final_tmp["RNA_id"] = CB_list
    df_cell_to_BC = pd.DataFrame({"RNA_id": CB_flat, "clone_id": Clone_id_flat}).merge(
        df_final_tmp.filter(["clone_id", "fate"]), on="clone_id", how="left"
    )
    return df_final_tmp, df_cell_to_BC


def estimate_error_rate(df):
    df_plot = df.groupby(["cell_id"]).agg(
        clone_measure_N=("clone_id", lambda x: len(x)),
).reset_index()
    df_sub=df[df['cell_id'].isin(df_plot[df_plot['clone_measure_N']>1]['cell_id'].to_list())
               ].sort_values(['RNA_id','locus','read'],ascending=False).set_index(['RNA_id','locus'])
    
    error_rate=(df_sub.groupby(["cell_id"]).agg(
        unique_allele_N=("allele", lambda x: len(set(x))),
    ).reset_index()['unique_allele_N']>1).mean()
    return pd.DataFrame({'>1_observation':[len(df_sub['cell_id'].unique())],'error_rate':[error_rate]}).set_index('error_rate')