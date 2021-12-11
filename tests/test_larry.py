import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import cospar as cs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat

from tests.context import larry


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
    df_all = pd.read_csv(
        os.path.join(shared_datadir, "LARRY", "Lime", "test.csv"), index_col=0
    )
    # seq_list = df_all["clone_id"]
    # whiteList = [
    #     "AAAATGATGTAATTTTGGGCTCGTCTTA",
    #     "AAACATGACGTTCAACTGGAGGACAATA",
    #     "AAACGAAGTTGCTTTATTAGAGATCCCA",
    #     "AAACTTTGGGTTAAGGCCAAAAAATCGT",
    #     "AAAGTATAAAGTAGATGTGTGTCGGCGC",
    #     "AAATGTAACCTAGAGTACAATATATAAC",
    #     "AAATGTTCTAGTTCCTTCTCAAATCTGA",
    #     "AAATTCTAACAGTCGACTATAAGACCAC",
    #     "AACATATAGACCACACGTGCTTGCTATA",
    #     "AACCCCTATTTATGTATTTCGGCCTGTA",
    #     "AACTAATTCAACCCAACGCAGAGCGCAA",
    # ]
    # mapping, new_seq_list = larry.denoise_sequence(
    #     seq_list, method="distance", distance_threshold=6, whiteList=whiteList
    # )
    # new_seq_list = set(new_seq_list)
    # new_seq_list.remove("nan")
    # distance = larry.QC_sequence_distance(list(set(new_seq_list)))
    # larry.plot_seq_distance(distance)

    # whiteList=pd.read_csv('data/actual_bc.csv',index_col=0)['whitelist']
    import cospar as cs

    adata = cs.hf.read(
        "/Users/shouwenwang/Dropbox (HMS)/shared_folder_with_Li/Analysis/notebooks/data/scLimeCat_adata_preprocessed.h5ad"
    )
    whiteList = list(adata[adata.obs["time_info"] == "2"].obs_names)

    mapping_dictionary = {
        "LARRY_Lime_33": "Lime_RNA_101",
        "LARRY_Lime_34": "Lime_RNA_102",
        "LARRY_Lime_35": "Lime_RNA_103",
        "LARRY_Lime_36": "Lime_RNA_104",
        "LARRY_10X_31": "MPP_10X_A3_1",
        "LARRY_10X_32": "MPP_10X_A4_1",
    }

    df_all = larry.rename_library_info(df_all, mapping_dictionary)
    df_temp = df_all[df_all["read"] >= 3]
    mapping, new_seq_list = larry.denoise_sequence(
        df_temp["cell_id"],
        method="distance",
        distance_threshold=2,
        whiteList=list(whiteList),
    )


os.chdir(os.path.dirname(__file__))
cs.settings.verbosity = 3  # range: 0 (error),1 (warning),2 (info),3 (hint).
# test_load_dataset("data")
# test_preprocessing("data")
# test_clonal_analysis("data")
# test_Tmap_inference("data")
# test_Tmap_plotting("data")
print("current directory", str(os.getcwd()))
test_all("data")
