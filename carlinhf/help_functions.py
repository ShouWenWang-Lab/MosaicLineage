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
from matplotlib import pyplot as plt
from scipy.io import loadmat

from . import larry, lineage


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
    fate_names = np.array(fate_names)
    partition_fate = [x for x in fate_names if "LK" in x][0]
    sel_id = np.nonzero(fate_names == target_fate)[0][0]
    valid_clone_idx = coarse_X_clone[fate_names == target_fate].sum(0) > 0
    if exclude_fates is not None:
        for x in exclude_fates:
            valid_clone_idx_tmp = ~(coarse_X_clone[fate_names == x].sum(0) > 0)
            valid_clone_idx = valid_clone_idx & valid_clone_idx_tmp

    valid_clone_idx_tmp_2 = coarse_X_clone[fate_names == partition_fate].sum(0) > 0

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
        plt.title(
            f"{np.sum(temp_idx)} clones; {short_fate_names[3]} {norm_X_count[sel_id,3]:.2f}; {short_fate_names[4]} {norm_X_count[sel_id,4]:.2f}; {short_fate_names[5]} {norm_X_count[sel_id,5]:.2f} {short_fate_names[6]} {norm_X_count[sel_id,6]:.2f}; {short_fate_names[7]} {norm_X_count[sel_id,7]:.2f}"
        )


def custom_fate_bias_heatmap(
    coarse_X_clone,
    fate_names,
    conditional_fates=["LL405-E5-LT-HSC", "LL405-E5-ST-HSC", "LL405-E5-MPP3-4"],
):
    custom_conditional_heatmap(
        coarse_X_clone,
        fate_names,
        target_fate=conditional_fates[0],
        conditional_fates=conditional_fates[1:],
        only_LK=True,
        final_x_ticks=False,
        exclude_fates=None,
    )
    custom_conditional_heatmap(
        coarse_X_clone,
        fate_names,
        target_fate=conditional_fates[1],
        conditional_fates=conditional_fates[2:],
        only_LK=True,
        final_x_ticks=False,
        exclude_fates=[conditional_fates[0]],
    )
    custom_conditional_heatmap(
        coarse_X_clone,
        fate_names,
        target_fate=conditional_fates[2],
        conditional_fates=[],
        only_LK=True,
        final_x_ticks=True,
        exclude_fates=conditional_fates[:2],
    )
