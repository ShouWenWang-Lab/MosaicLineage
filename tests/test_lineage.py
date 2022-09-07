import os
import sys

from carlinhf import lineage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import cospar as cs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat

from tests.context import hf


def config(shared_datadir):
    cs.settings.data_path = os.path.join(shared_datadir, "..", "output")
    cs.settings.figure_path = os.path.join(shared_datadir, "..", "output")
    cs.settings.verbosity = 0  # range: 0 (error),1 (warning),2 (info),3 (hint).
    cs.settings.set_figure_params(
        format="png", figsize=[4, 3.5], dpi=25, fontsize=14, pointsize=3, dpi_save=25
    )
    cs.hf.set_up_folders()  # setup the data_path and figure_path


def test_all(shared_datadir):
    config(shared_datadir)
    data = loadmat(
        os.path.join(shared_datadir, "merge_all", "allele_breakdown_by_sample.mat")
    )
    SampleList = [xx[0][0] for xx in data["sample_names"]]

    tmp_list = []
    for sample in SampleList:
        print(f"Sample: {sample}")
        base_dir = os.path.join(shared_datadir, f"{sample}")
        df_tmp = hf.load_allele_info(base_dir)
        df_tmp["sample"] = sample.split("_")[0]
        df_tmp["mouse"] = sample.split("-")[0]
        tmp_list.append(df_tmp)
    df_all_0 = pd.concat(tmp_list)
    # df_all = hf.query_allele_frequencies(df_all_0, df_all_0)
    df_all_0["normalized_count"] = df_all_0["UMI_count"]
    adata_orig = hf.generate_adata_allele_by_mutation(df_all_0)

    adata_sub = adata_orig
    adata_sub.obsm["X_clone"] = adata_sub.X

    cell_N_temp = []
    clone_N_temp = []
    mutation_N_temp = []
    for xx in set(adata_sub.obs["sample"]):
        cell_N_temp.append(np.sum(adata_sub.obs["sample"] == xx))
        clone_N_temp.append(
            len(set(adata_sub[adata_sub.obs["sample"] == xx].obs["allele"]))
        )
        mutation_N_temp.append(
            np.sum(
                adata_sub[adata_sub.obs["sample"] == xx]
                .obsm["X_clone"]
                .sum(0)
                .A.flatten()
                > 0
            )
        )
    df_info = pd.DataFrame(
        {
            "Sample": SampleList,
            "Cell number": cell_N_temp,
            "Clone number": clone_N_temp,
            "mutation number": mutation_N_temp,
        }
    )
    df_info.to_csv(os.path.join(shared_datadir, "..", "output", "X_clone_info.csv"))

    adata_sub.obs["state_info"] = adata_sub.obs["sample"]
    adata_sub.uns["data_des"] = ["coarse"]
    cs.settings.set_figure_params(
        format="png", figsize=[4, 4], dpi=75, fontsize=15, pointsize=2
    )
    cs.pl.barcode_heatmap(
        adata_sub,
        color_bar=True,
        fig_height=10,
        fig_width=10,
        y_ticks=None,  # adata_sub.var_names,
        x_label="Allele",
        y_label="Mutation",
    )

    cs.tl.fate_hierarchy(adata_sub, source="X_clone")
    cs.pl.fate_hierarchy(adata_sub, source="X_clone")
    my_tree_coarse = adata_sub.uns["fate_hierarchy_X_clone"]["tree"]
    hf.visualize_tree(
        my_tree_coarse,
        color_coding=None,
        mode="r",
        data_des="coarse",
        figure_path=cs.settings.data_path,
    )

    input_dict = {
        "LL607-MPP3-4": 1,
        "LL607-MPP2": 1,
        "LL605-MPP3-4": 0,
        "LL605-LK": 0,
        "LL607-HSC": 1,
        "LL605-HSC": 0,
        "LL605-MPP2": 0,
        "LL607-LK": 1,
    }

    origin_score = hf.onehot(input_dict)
    parent_map = adata_sub.uns["fate_hierarchy_X_clone"]["parent_map"]
    node_mapping = adata_sub.uns["fate_hierarchy_X_clone"]["node_mapping"]
    corr, __ = hf.tree_reconstruction_accuracy(
        parent_map, node_mapping, origin_score=origin_score, weight_factor=1
    )
    print(f"Tree reconstruction accuracy: {corr}")

    cs.tl.fate_coupling(adata_sub, source="X_clone", method="SW")
    cs.pl.fate_coupling(adata_sub, source="X_clone")
    X_coupling = adata_sub.uns["fate_coupling_X_clone"]["X_coupling"]
    fate_names = adata_sub.uns["fate_coupling_X_clone"]["fate_names"]

    cs.settings.set_figure_params(
        format="png", figsize=[4, 3.5], dpi=75, fontsize=15, pointsize=4
    )
    scores = hf.evaluate_coupling_matrix(
        X_coupling, fate_names, origin_score=origin_score, decay_factor=1, plot=True
    )
    print(f"Coupling matrix scores: {scores}")


os.chdir(os.path.dirname(__file__))
cs.settings.verbosity = 3  # range: 0 (error),1 (warning),2 (info),3 (hint).
# test_load_dataset("data")
# test_preprocessing("data")
# test_clonal_analysis("data")
# test_Tmap_inference("data")
# test_Tmap_plotting("data")
print("current directory", str(os.getcwd()))
test_all("data")
