import os

import cospar as cs
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.spatial import distance

import carlinhf.CARLIN as car
import carlinhf.lineage as lineage
import carlinhf.plotting as plotting

cs.settings.set_figure_params(format="pdf", figsize=[4, 3.5], dpi=150, fontsize=14)
rcParams["legend.handlelength"] = 1.5


# This file contains lousy functions that are often from one-off analysis.
# It primarily focuses with plotting and ends with the plotting


####################################

# Comparing bulk Cas9 and DNTT data

####################################


def remove_samples(df, removed_sample=None):
    """
    remove some samples in a dataframe
    """

    if removed_sample is not None:
        df_new = df[~df["sample"].isin(removed_sample)]
    else:
        df_new = df
    return df_new


def mutation_statistics_box_plot(
    df,
    sample_key,
    removed_sample=None,
    keys=["ave_del_len", "ave_insert_len", "ins_del_ratio_ratio_by_eventful_UMI"],
    y_labels=[
        "Average deletion length",
        "Average insertion length",
        "(Insertion #)/(deletion #): per UMI",
    ],
):
    """
    df_noMerge: a
    """

    ## remove some negative controls
    if removed_sample is None:
        removed_sample = ["merge_all"]
    else:
        removed_sample.append("merge_all")
    df_noM_new = remove_samples(df, removed_sample)

    for j, key in enumerate(keys):
        fig, ax = plt.subplots(figsize=(3, 4))
        ax = sns.boxplot(data=df_noM_new, x="Design", y=key, width=0.5)
        ax = sns.stripplot(
            data=df_noM_new,
            x="Design",
            y=key,
            size=5,
            edgecolor="black",
            linewidth=1,
            jitter=1,
        )
        ax.set_ylabel(y_labels[j])
        ax.set_xlabel("")
        plt.xticks(rotation=90)
        plt.tight_layout()
        fig.savefig(f"figure/{sample_key}/{key}_202105.pdf")


def mutation_statistics_distribution_per_allele(df_LL, df_SB, sample_key):
    """
    Compare mutation statistics per allele between two data source

    df_LL: allele count dataframe from Cas9-TdT mouse
    df_SB: allele count dataframe from Cas9 mouse
    sample_key: for making a separate folder and save the data
    """
    os.makedirs(f"figure/{sample_key}", exist_ok=True)

    ### Mutation event number per allele
    mut_per_allele_LL = lineage.mutations_per_allele(df_LL)
    mut_per_allele_SB = lineage.mutations_per_allele(df_SB)
    mut_LL_hist_y, mut_LL_hist_x = np.histogram(mut_per_allele_LL, bins=np.arange(15))
    mut_LL_hist_y = mut_LL_hist_y / np.sum(mut_LL_hist_y)

    mut_SB_hist_y, mut_SB_hist_x = np.histogram(mut_per_allele_SB, bins=np.arange(15))
    mut_SB_hist_y = mut_SB_hist_y / np.sum(mut_SB_hist_y)

    print(
        f"Mean mut for Cas9: {np.mean(mut_per_allele_SB)}; for Cas9-TdT: {np.mean(mut_per_allele_LL)}"
    )

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=mut_SB_hist_x[:-1], y=mut_SB_hist_y, label="Cas9", marker="o", ax=ax
    )
    ax = sns.lineplot(
        x=mut_LL_hist_x[:-1], y=mut_LL_hist_y, label="Cas9-TdT", ax=ax, marker="o"
    )

    ax.set_xlim([-0.1, 10])
    ax.set_xlabel("Mutation event # per allele")
    ax.set_ylabel("Distribution")
    plt.tight_layout()
    plotting.add_shade(ax)
    plt.savefig(f"figure/{sample_key}/mutation_per_allele_compare_cas9_Dntt.pdf")

    ### Insertion event number per allele
    ins_per_allele_LL, del_per_allele_LL = lineage.mutations_per_allele_ins_del(df_LL)
    ins_per_allele_SB, del_per_allele_SB = lineage.mutations_per_allele_ins_del(df_SB)

    ins_LL_hist_y, ins_LL_hist_x = np.histogram(ins_per_allele_LL, bins=np.arange(15))
    ins_LL_hist_y = ins_LL_hist_y / np.sum(ins_LL_hist_y)

    ins_SB_hist_y, ins_SB_hist_x = np.histogram(ins_per_allele_SB, bins=np.arange(15))
    ins_SB_hist_y = ins_SB_hist_y / np.sum(ins_SB_hist_y)

    print(
        f"Mean ins for Cas9: {np.mean(ins_per_allele_SB)}; for Cas9-TdT: {np.mean(ins_per_allele_LL)}"
    )

    fig, ax = plt.subplots()
    ax = sns.lineplot(x=ins_SB_hist_x[:-1], y=ins_SB_hist_y, label="Cas9", marker="o")
    ax = sns.lineplot(
        x=ins_LL_hist_x[:-1], y=ins_LL_hist_y, label="Cas9-TdT", ax=ax, marker="o"
    )

    ax.set_xlim([-0.1, 10])
    ax.set_xlabel("Insertion event # per allele")
    ax.set_ylabel("Distribution")
    plt.tight_layout()
    plotting.add_shade(ax)
    plt.savefig(f"figure/{sample_key}/insertion_per_allele_compare_cas9_Dntt.pdf")

    ### Deletion event number per allele
    del_LL_hist_y, del_LL_hist_x = np.histogram(del_per_allele_LL, bins=np.arange(15))
    del_LL_hist_y = del_LL_hist_y / np.sum(del_LL_hist_y)

    del_SB_hist_y, del_SB_hist_x = np.histogram(del_per_allele_SB, bins=np.arange(15))
    del_SB_hist_y = del_SB_hist_y / np.sum(del_SB_hist_y)

    print(
        f"Mean del number for Cas9: {np.mean(del_per_allele_SB)}; for Cas9-TdT: {np.mean(del_per_allele_LL)}"
    )

    fig, ax = plt.subplots()
    ax = sns.lineplot(x=del_SB_hist_x[:-1], y=del_SB_hist_y, label="Cas9", marker="o")
    ax = sns.lineplot(
        x=del_LL_hist_x[:-1], y=del_LL_hist_y, label="Cas9-TdT", ax=ax, marker="o"
    )

    ax.set_xlim([-0.1, 10])
    ax.set_xlabel("Deletion event # per allele")
    ax.set_ylabel("Distribution")
    plt.tight_layout()
    plotting.add_shade(ax)
    plt.savefig(f"figure/{sample_key}/deletion_per_allele_compare_cas9_Dntt.pdf")

    ## Total insertion length per allele
    ins_per_allele_LL, del_per_allele_LL = lineage.mutations_length_per_allele_ins_del(
        df_LL
    )
    ins_per_allele_SB, del_per_allele_SB = lineage.mutations_length_per_allele_ins_del(
        df_SB
    )
    ins_length_LL = [np.sum(x) for x in ins_per_allele_LL]
    ins_length_SB = [np.sum(x) for x in ins_per_allele_SB]
    ins_LL_hist_y, ins_LL_hist_x = np.histogram(ins_length_LL, bins=np.arange(100))
    ins_LL_hist_y = ins_LL_hist_y / np.sum(ins_LL_hist_y)

    ins_SB_hist_y, ins_SB_hist_x = np.histogram(ins_length_SB, bins=np.arange(100))
    ins_SB_hist_y = ins_SB_hist_y / np.sum(ins_SB_hist_y)

    print(
        f"Mean total insertion length for Cas9: {np.mean(ins_length_SB)}; for Cas9-TdT: {np.mean(ins_length_LL)}"
    )
    print(
        f"Mediam total insertion length for Cas9: {np.median(ins_length_SB)}; for Cas9-TdT: {np.median(ins_length_LL)}"
    )

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=ins_SB_hist_x[:-1], y=ins_SB_hist_y, label="Cas9"
    )  # ,marker='o')
    ax = sns.lineplot(
        x=ins_LL_hist_x[:-1], y=ins_LL_hist_y, label="Cas9-TdT"
    )  # ,ax=ax,marker='o')

    ax.set_xlim([-0.1, 30])
    ax.set_xlabel("Total insertion length per allele")
    ax.set_ylabel("Distribution")
    # plt.xscale('log')
    plt.tight_layout()
    plotting.add_shade(ax)
    plt.savefig(f"figure/{sample_key}/tot_ins_length_per_allele_compare_cas9_Dntt.pdf")

    ## Single insertion length per allele
    ins_length_LL = []
    for x in ins_per_allele_LL:
        ins_length_LL += list(x)
    ins_length_SB = []
    for x in ins_per_allele_SB:
        ins_length_SB += list(x)
    ins_LL_hist_y, ins_LL_hist_x = np.histogram(ins_length_LL, bins=np.arange(100))
    ins_LL_hist_y = ins_LL_hist_y / np.sum(ins_LL_hist_y)

    ins_SB_hist_y, ins_SB_hist_x = np.histogram(ins_length_SB, bins=np.arange(100))
    ins_SB_hist_y = ins_SB_hist_y / np.sum(ins_SB_hist_y)

    print(
        f"Mean single insertion length for Cas9: {np.mean(ins_length_SB)}; for Cas9-TdT: {np.mean(ins_length_LL)}"
    )
    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=ins_SB_hist_x[:-1], y=ins_SB_hist_y, label="Cas9"
    )  # ,marker='o')
    ax = sns.lineplot(
        x=ins_LL_hist_x[:-1], y=ins_LL_hist_y, label="Cas9-TdT"
    )  # ,ax=ax,marker='o')

    ax.set_xlim([-0.1, 30])
    ax.set_xlabel("Single insertion length per allele")
    ax.set_ylabel("Distribution")
    # plt.xscale('log')
    plt.tight_layout()
    plotting.add_shade(ax)
    plt.savefig(
        f"figure/{sample_key}/single_ins_length_per_allele_compare_cas9_Dntt.pdf"
    )

    ## Total deletion length per allele
    del_length_LL = [np.sum(x) for x in del_per_allele_LL]
    del_length_SB = [np.sum(x) for x in del_per_allele_SB]
    del_LL_hist_y, del_LL_hist_x = np.histogram(del_length_LL, bins=np.arange(300))
    del_LL_hist_y = del_LL_hist_y / np.sum(del_LL_hist_y)

    del_SB_hist_y, del_SB_hist_x = np.histogram(del_length_SB, bins=np.arange(300))
    del_SB_hist_y = del_SB_hist_y / np.sum(del_SB_hist_y)

    print(
        f"Mean total deletion length for Cas9: {np.mean(del_length_SB)}; for Cas9-TdT: {np.mean(del_length_LL)}"
    )
    print(
        f"Mediam total deletion length for Cas9: {np.median(del_length_SB)}; for Cas9-TdT: {np.median(del_length_LL)}"
    )

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=del_SB_hist_x[:-1], y=del_SB_hist_y, label="Cas9"
    )  # ,marker='o')
    ax = sns.lineplot(
        x=del_LL_hist_x[:-1], y=del_LL_hist_y, label="Cas9-TdT"
    )  # ,ax=ax,marker='o')

    # ax.set_xlim([-0.1,30])
    ax.set_xlabel("Total deletion length per allele")
    ax.set_ylabel("Distribution")
    # plt.xscale('log')
    plt.tight_layout()
    plotting.add_shade(ax)
    plt.savefig(f"figure/{sample_key}/tot_del_length_per_allele_compare_cas9_Dntt.pdf")

    ## Single deletion length per allele
    del_length_LL = []
    for x in del_per_allele_LL:
        del_length_LL += list(x)
    del_length_SB = []
    for x in del_per_allele_SB:
        del_length_SB += list(x)
    del_LL_hist_y, del_LL_hist_x = np.histogram(del_length_LL, bins=np.arange(300))
    del_LL_hist_y = del_LL_hist_y / np.sum(del_LL_hist_y)

    del_SB_hist_y, del_SB_hist_x = np.histogram(del_length_SB, bins=np.arange(300))
    del_SB_hist_y = del_SB_hist_y / np.sum(del_SB_hist_y)

    print(
        f"Mean single deletion length for Cas9: {np.mean(del_length_SB)}; for Cas9-TdT: {np.mean(del_length_LL)}"
    )
    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=del_SB_hist_x[:-1], y=del_SB_hist_y, label="Cas9"
    )  # ,marker='o')
    ax = sns.lineplot(
        x=del_LL_hist_x[:-1], y=del_LL_hist_y, label="Cas9-TdT"
    )  # ,ax=ax,marker='o')

    # ax.set_xlim([-0.1,30])
    ax.set_xlabel("Single deletion length per allele")
    ax.set_ylabel("Distribution")
    # plt.xscale('log')
    plotting.add_shade(ax)
    plt.tight_layout()
    plt.savefig(
        f"figure/{sample_key}/single_del_length_per_allele_compare_cas9_Dntt.pdf"
    )

    ## Single deletion length
    y_SB = np.cumsum(del_SB_hist_y)
    y_LL = np.cumsum(del_LL_hist_y)
    fig, ax = plt.subplots()
    sns.lineplot(x=[0] + list(del_SB_hist_x[:-1]), y=[0] + list(y_SB), label="Cas9")
    ax = sns.lineplot(
        x=[0] + list(del_LL_hist_x[:-1]), y=[0] + list(y_LL), label="Cas9-TdT"
    )
    ax.set_xlabel("Single deletion length")
    ax.set_ylabel("Cumulative deletion frequency")
    plt.tight_layout()
    plt.savefig(
        f"figure/{sample_key}/cumu_single_del_length_per_allele_compare_cas9_Dntt.pdf"
    )


def mutation_statistics_distribution_per_allele_single_input(
    df_SB, sample_key, label="Cas9"
):
    """
    Check mutation statistics per allele from a single data source

    df_SB: allele count dataframe from Cas9 mouse
    sample_key: for making a separate folder and save the data
    """
    os.makedirs(f"figure/{sample_key}", exist_ok=True)

    ### Mutation event number per allele
    mut_per_allele_SB = lineage.mutations_per_allele(df_SB)

    mut_SB_hist_y, mut_SB_hist_x = np.histogram(mut_per_allele_SB, bins=np.arange(15))
    mut_SB_hist_y = mut_SB_hist_y / np.sum(mut_SB_hist_y)

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=mut_SB_hist_x[:-1], y=mut_SB_hist_y, label=label, marker="o", ax=ax
    )

    ax.set_xlim([-0.1, 10])
    ax.set_xlabel("Mutation event # per allele")
    ax.set_ylabel("Distribution")
    plt.tight_layout()
    plotting.add_shade_1(ax)
    plt.savefig(f"figure/{sample_key}/mutation_per_allele_compare_cas9_Dntt.pdf")

    ### Insertion event number per allele
    ins_per_allele_SB, del_per_allele_SB = lineage.mutations_per_allele_ins_del(df_SB)

    ins_SB_hist_y, ins_SB_hist_x = np.histogram(ins_per_allele_SB, bins=np.arange(15))
    ins_SB_hist_y = ins_SB_hist_y / np.sum(ins_SB_hist_y)

    fig, ax = plt.subplots()
    ax = sns.lineplot(x=ins_SB_hist_x[:-1], y=ins_SB_hist_y, label=label, marker="o")
    ax.set_xlim([-0.1, 10])
    ax.set_xlabel("Insertion event # per allele")
    ax.set_ylabel("Distribution")
    plt.tight_layout()
    plotting.add_shade_1(ax)
    plt.savefig(f"figure/{sample_key}/insertion_per_allele_compare_cas9_Dntt.pdf")

    ### Deletion event number per allele
    del_SB_hist_y, del_SB_hist_x = np.histogram(del_per_allele_SB, bins=np.arange(15))
    del_SB_hist_y = del_SB_hist_y / np.sum(del_SB_hist_y)

    fig, ax = plt.subplots()
    ax = sns.lineplot(x=del_SB_hist_x[:-1], y=del_SB_hist_y, label=label, marker="o")
    ax.set_xlim([-0.1, 10])
    ax.set_xlabel("Deletion event # per allele")
    ax.set_ylabel("Distribution")
    plt.tight_layout()
    plotting.add_shade_1(ax)
    plt.savefig(f"figure/{sample_key}/deletion_per_allele_compare_cas9_Dntt.pdf")

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=ins_SB_hist_x[:-1],
        y=ins_SB_hist_y,
        label="Insertion",
        marker="o",
        color="#225ea8",
    )
    ax = sns.lineplot(
        x=del_SB_hist_x[:-1],
        y=del_SB_hist_y,
        label="Deletion",
        marker="o",
        color="#d7301f",
    )
    ax.set_xlim([-0.1, 10])
    ax.set_xlabel("Event # per allele")
    ax.set_ylabel("Distribution")
    plt.tight_layout()
    plotting.add_shade(ax, color=["#225ea8", "#d7301f"])
    plt.savefig(f"figure/{sample_key}/ins_deletion_per_allele.pdf")

    ## Total insertion length per allele
    ins_per_allele_SB, del_per_allele_SB = lineage.mutations_length_per_allele_ins_del(
        df_SB
    )
    ins_length_SB = [np.sum(x) for x in ins_per_allele_SB]
    ins_SB_hist_y, ins_SB_hist_x = np.histogram(ins_length_SB, bins=np.arange(100))
    ins_SB_hist_y = ins_SB_hist_y / np.sum(ins_SB_hist_y)
    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=ins_SB_hist_x[:-1], y=ins_SB_hist_y, label=label
    )  # ,marker='o')
    ax.set_xlim([-0.1, 30])
    ax.set_xlabel("Total insertion length per allele")
    ax.set_ylabel("Distribution")
    # plt.xscale('log')
    plt.tight_layout()
    plotting.add_shade_1(ax)
    plt.savefig(f"figure/{sample_key}/tot_ins_length_per_allele_compare_cas9_Dntt.pdf")

    ## Single insertion length per allele
    ins_length_SB = []
    for x in ins_per_allele_SB:
        ins_length_SB += list(x)

    ins_SB_hist_y, ins_SB_hist_x = np.histogram(ins_length_SB, bins=np.arange(100))
    ins_SB_hist_y = ins_SB_hist_y / np.sum(ins_SB_hist_y)

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=ins_SB_hist_x[:-1], y=ins_SB_hist_y, label=label
    )  # ,marker='o')

    ax.set_xlim([-0.1, 30])
    ax.set_xlabel("Single insertion length per allele")
    ax.set_ylabel("Distribution")
    # plt.xscale('log')
    plt.tight_layout()
    plotting.add_shade_1(ax)
    plt.savefig(
        f"figure/{sample_key}/single_ins_length_per_allele_compare_cas9_Dntt.pdf"
    )

    ## Total deletion length per allele
    del_length_SB = [np.sum(x) for x in del_per_allele_SB]

    del_SB_hist_y, del_SB_hist_x = np.histogram(del_length_SB, bins=np.arange(300))
    del_SB_hist_y = del_SB_hist_y / np.sum(del_SB_hist_y)

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=del_SB_hist_x[:-1], y=del_SB_hist_y, label=label
    )  # ,marker='o')

    # ax.set_xlim([-0.1,30])
    ax.set_xlabel("Total deletion length per allele")
    ax.set_ylabel("Distribution")
    # plt.xscale('log')
    plt.tight_layout()
    plotting.add_shade_1(ax)
    plt.savefig(f"figure/{sample_key}/tot_del_length_per_allele_compare_cas9_Dntt.pdf")

    ins_length_SB = [np.sum(x) for x in ins_per_allele_SB]
    ins_SB_hist_y, ins_SB_hist_x = np.histogram(ins_length_SB, bins=np.arange(100))
    ins_SB_hist_y = ins_SB_hist_y / np.sum(ins_SB_hist_y)
    rcParams["axes.spines.right"] = True
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        ins_SB_hist_x[:-1],
        ins_SB_hist_y,
        label="Ins",
        color="#225ea8",
    )  # ,marker='o')
    ax2 = plt.twinx()
    ax2.plot(
        del_SB_hist_x[:-1],
        del_SB_hist_y,
        label="Del",
        color="#d7301f",
    )  # ,marker='o')
    ax.set_xlabel("Total mutation length per allele")
    ax.set_ylabel("Distribution")
    # ax2.set_ylabel("")
    ax.figure.legend(loc="upper center")
    # plt.xscale('log')
    plotting.add_shade_1(ax, color="#225ea8")
    plotting.add_shade_1(ax2, color="#d7301f")
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/tot_ins_del_length_per_allele.pdf")
    rcParams["axes.spines.right"] = False

    ## Single deletion length per allele
    del_length_SB = []
    for x in del_per_allele_SB:
        del_length_SB += list(x)

    del_SB_hist_y, del_SB_hist_x = np.histogram(del_length_SB, bins=np.arange(300))
    del_SB_hist_y = del_SB_hist_y / np.sum(del_SB_hist_y)

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x=del_SB_hist_x[:-1], y=del_SB_hist_y, label=label
    )  # ,marker='o')

    # ax.set_xlim([-0.1,30])
    ax.set_xlabel("Single deletion length per allele")
    ax.set_ylabel("Distribution")
    # plt.xscale('log')
    plotting.add_shade_1(ax)
    plt.tight_layout()
    plt.savefig(
        f"figure/{sample_key}/single_del_length_per_allele_compare_cas9_Dntt.pdf"
    )

    ## Single deletion length
    y_SB = np.cumsum(del_SB_hist_y)
    fig, ax = plt.subplots()
    sns.lineplot(x=[0] + list(del_SB_hist_x[:-1]), y=[0] + list(y_SB), label=label)
    ax.set_xlabel("Single deletion length")
    ax.set_ylabel("Cumulative deletion frequency")
    plt.tight_layout()
    plt.savefig(
        f"figure/{sample_key}/cumu_single_del_length_per_allele_compare_cas9_Dntt.pdf"
    )


def mutation_statistics_distribution_UMI(df_LL, df_SB, sample_key):
    """
    Mutation statistics per UMI between two data source

    df_LL: allele count dataframe from Cas9-TdT mouse
    df_SB: allele count dataframe from Cas9 mouse
    sample_key: for making a separate folder and save the data
    """
    os.makedirs(f"figure/{sample_key}", exist_ok=True)

    df_SB = lineage.correct_null_allele_frequency(df_SB, editing_efficiency=0.3)
    ins_per_allele_LL, del_per_allele_LL = lineage.mutations_length_per_allele_ins_del(
        df_LL
    )
    ins_per_allele_SB, del_per_allele_SB = lineage.mutations_length_per_allele_ins_del(
        df_SB
    )

    freq_LL = df_LL["UMI_count"].to_numpy()
    freq_SB = df_SB["UMI_count"].to_numpy()
    ins_length_LL_tot = [
        np.repeat(int(np.sum(x)), freq_LL[i]) for i, x in enumerate(ins_per_allele_LL)
    ]
    ins_length_SB_tot = [
        np.repeat(int(np.sum(x)), freq_SB[i]) for i, x in enumerate(ins_per_allele_SB)
    ]
    ins_length_LL = []
    for x in ins_length_LL_tot:
        ins_length_LL += list(x)

    ins_length_SB = []
    for x in ins_length_SB_tot:
        ins_length_SB += list(x)

    ins_LL_hist_y, ins_LL_hist_x = np.histogram(ins_length_LL, bins=np.arange(100))
    ins_LL_hist_y = ins_LL_hist_y / np.sum(ins_LL_hist_y)

    ins_SB_hist_y, ins_SB_hist_x = np.histogram(ins_length_SB, bins=np.arange(100))
    ins_SB_hist_y = ins_SB_hist_y / np.sum(ins_SB_hist_y)

    print(
        f"Mean insertion length for Cas9: {np.mean(ins_length_SB)}; for Cas9-TdT: {np.mean(ins_length_LL)}"
    )

    ax = sns.lineplot(x=ins_SB_hist_x[:-1], y=ins_SB_hist_y, label="Cas9", marker="o")
    ax = sns.lineplot(
        x=ins_LL_hist_x[:-1], y=ins_LL_hist_y, label="Cas9-TdT", ax=ax, marker="o"
    )
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlim([-0.1, 10.2])
    ax.set_xlabel("Total insertion length per UMI")
    ax.set_ylabel("Distribution")
    # plt.xscale('log')
    plt.tight_layout()
    plotting.add_shade(ax)
    plt.savefig(f"figure/{sample_key}/tot_ins_length_per_UMI_compare_cas9_Dntt.pdf")


def allele_statistics_at_given_sampling_depth(
    df_Merge, sample_key, removed_sample=None, allele_cutoff=0
):

    # df_noMerge=df_Merge[df_Merge['sample']!='merge_all']
    x_label_1 = "Observed cell # (called UMI)"
    x_label_2 = "Edited cell # (edited UMI)"
    os.makedirs("figure/" + sample_key, exist_ok=True)

    ## Singleton fraction
    df_Merge_1 = df_Merge[df_Merge.total_alleles > allele_cutoff]
    df_Merge_1["singleton_ratio"] = (
        df_Merge_1["singleton"] / df_Merge_1["total_alleles"]
    )
    df_Merge_1["total_alleles"] = np.log10(df_Merge_1["total_alleles"])
    g = sns.lmplot(
        data=df_Merge_1,
        x="total_alleles",
        y="singleton_ratio",
        hue="Design",
        ci=None,
        palette="muted",
        height=4,
        aspect=1.2,
        robust=True,
        scatter_kws={"s": 50, "alpha": 1, "edgecolor": "k"},
        line_kws={"linewidth": 2},
    )
    g.ax.set_xlabel("Total observed alleles")
    g.ax.set_ylabel("Singleton fraction")
    g.ax.set_ylim([0, 1])
    g.ax.set_xlim([2, 6])
    plt.xticks(
        ticks=[2, 3, 4, 5, 6],
        labels=[r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"],
    )
    plt.savefig(f"figure/{sample_key}/Singleton_fraction.pdf")

    ## remove some negative controls
    df_Merge_2 = remove_samples(df_Merge, removed_sample)

    ## Observed allele number
    g = sns.lmplot(
        data=df_Merge_2,
        x="UMI_eventful",
        y="total_alleles",
        hue="Design",
        ci=None,
        palette="muted",
        height=4,
        aspect=1.2,
        scatter_kws={"s": 50, "alpha": 1, "edgecolor": "k"},
    )
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    g.ax.set_xlabel(x_label_2)
    g.ax.set_ylabel("Observed allele number")
    # plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/total_allele_vs_UMI_eventful.pdf")

    g = sns.lmplot(
        data=df_Merge_2,
        x="UMI_called",
        y="total_alleles",
        hue="Design",
        ci=None,
        palette="muted",
        height=4,
        aspect=1.2,
        scatter_kws={"s": 50, "alpha": 1, "edgecolor": "k"},
    )
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    g.ax.set_xlabel(x_label_1)
    g.ax.set_ylabel("Observed allele number")
    # plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/total_allele_vs_UMI_called.pdf")

    ## Observed singleton
    g = sns.lmplot(
        data=df_Merge_2,
        x="UMI_eventful",
        y="singleton",
        hue="Design",
        ci=None,
        palette="muted",
        height=4,
        aspect=1.2,
        scatter_kws={"s": 50, "alpha": 1, "edgecolor": "k"},
    )
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    g.ax.set_xlabel(x_label_2)
    g.ax.set_ylabel("# of alleles observed only once")
    # plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/singleton_vs_UMI_eventful.pdf")

    g = sns.lmplot(
        data=df_Merge_2,
        x="UMI_called",
        y="singleton",
        hue="Design",
        ci=None,
        palette="muted",
        height=4,
        aspect=1.2,
        scatter_kws={"s": 50, "alpha": 1, "edgecolor": "k"},
    )
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    g.ax.set_xlabel(x_label_1)
    g.ax.set_ylabel("# of alleles observed only once")
    # plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/singleton_vs_UMI_called.pdf")

    ## Effective allele number
    df_Merge_3 = df_Merge_2[df_Merge_2.total_alleles > allele_cutoff]
    g = sns.lmplot(
        data=df_Merge_3,
        x="UMI_eventful",
        y="effective_allele_N",
        hue="Design",
        ci=None,
        palette="muted",
        height=4,
        aspect=1.2,
        scatter_kws={"s": 50, "alpha": 1, "edgecolor": "k"},
        line_kws={"linewidth": 2},
    )
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    g.ax.set_xlabel(x_label_2)
    g.ax.set_ylabel("Effective allele number")
    # plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/effective_allele_vs_UMI_eventful.pdf")

    g = sns.lmplot(
        data=df_Merge_3,
        x="UMI_called",
        y="effective_allele_N",
        hue="Design",
        ci=None,
        palette="muted",
        height=4,
        aspect=1.2,
        robust=False,
        lowess=False,
        scatter_kws={"s": 50, "alpha": 1, "edgecolor": "k"},
        line_kws={"linewidth": 2},
    )
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    g.ax.set_xlabel(x_label_1)
    g.ax.set_ylabel("Effective allele number")
    # plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/effective_allele_vs_UMI_called.pdf")

    ## Average deletion length over CARLIN potential
    x = "ave_del_len"
    y = "CARLIN_potential_by_UMI"
    df_temp = df_Merge
    g = sns.lmplot(
        data=df_temp,
        x=x,
        y=y,
        hue="Design",
        ci=None,
        palette="muted",
        height=4,
        aspect=1.2,
        scatter_kws={"s": 50, "alpha": 1, "edgecolor": "k"},
        line_kws={"linewidth": 2},
    )
    # for i in range(len(df_temp)):
    #     plt.text(df_temp.iloc[i][x]+0.2, df_temp.iloc[i][y]+0.2, df_temp.iloc[i]['Tissue'])
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    g.ax.set_xlabel("Average deletion length")
    g.ax.set_ylabel("CARLIN potential by UMI")
    # plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/del_length_vs_CARLIN_potential.pdf")


def insertion_del_freq_histogram(df, sample_key):
    """
    This is to test whether insertion contributes to more rare alleles than deletion
    """

    os.makedirs("figure/" + sample_key, exist_ok=True)
    df_mutation = lineage.mutation_frequency(df, plot=False)
    x_var, y_var = plotting.plot_loghist(
        list(df_mutation["UMI_count"]), cutoff_y=1, bins=20
    )
    x_label = "Occurence # per allele (UMI count)"

    # x_var=[0,5,10,100,500,1000,10000,1000000]
    df_mutation_1 = df_mutation.reset_index()
    # df_mutation_1=df_mutation_1[
    sel_idx = df_mutation_1["mutation"].apply(
        lambda x: (("del" in x) and ("ins" not in x))
    )
    df_mutation_del = df_mutation_1[sel_idx]
    y_var_del, x_var_del = np.histogram(list(df_mutation_del["UMI_count"]), bins=x_var)

    sel_idx = df_mutation_1["mutation"].apply(
        lambda x: (("del" not in x) and ("ins" in x))
    )
    df_mutation_ins = df_mutation_1[sel_idx]
    y_var_ins, x_var_ins = np.histogram(list(df_mutation_ins["UMI_count"]), bins=x_var)

    sel_idx = df_mutation_1["mutation"].apply(lambda x: (("del" in x) and ("ins" in x)))
    df_mutation_indel = df_mutation_1[sel_idx]
    y_var_indel, x_var_indel = np.histogram(
        list(df_mutation_indel["UMI_count"]), bins=x_var
    )

    fig, ax = plt.subplots()
    plt.loglog(x_var_del[:-1], y_var_del, "-o", label="del")
    plt.loglog(x_var_ins[:-1], y_var_ins, "-o", label="ins")
    plt.loglog(x_var_indel[:-1], y_var_indel, "-o", label="indel")
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel("Histogram")
    plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/in_del_freq_histogram_log.pdf")

    fig, ax = plt.subplots()
    plt.loglog(x_var_del[:-1], y_var_del, "-o", label="del")
    plt.loglog(x_var_ins[:-1], y_var_ins + y_var_indel, "-o", label="ins+indel")
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel("Histogram")
    plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/indel_del_freq_histogram_log.pdf")

    fig, ax = plt.subplots()
    ax = sns.lineplot(x=x_var_del[:-1], y=y_var_del, label="del", marker="o")
    ax = sns.lineplot(
        x=x_var_del[:-1],
        y=y_var_ins + y_var_indel,
        label="ins+indel",
        ax=ax,
        marker="o",
    )
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlim([0, 50])
    plt.xlabel(x_label)
    ax.set_ylabel("Histogram")
    # plt.xscale('log')
    plt.tight_layout()
    plotting.add_shade(ax)
    plt.tight_layout()
    plt.savefig(f"figure/{sample_key}/ins_del_freq_histogram_normal_scale.pdf")


def plot_deletion_statistics(df):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="del_initial", y="del_end", s=0.5, alpha=0.1)

    count_data, location = np.histogram(df["del_initial"], bins=np.arange(270))
    max_info = []
    for x in range(10):
        idx = np.arange(x * 27, (x + 1) * 27 - 1)
        cur = np.argmax(count_data[idx])
        max_info.append([count_data[idx][cur], location[idx][cur]])

    fig, ax = plt.subplots()
    ax = sns.histplot(data=df, x="del_initial", bins=200)
    for i in range(len(max_info)):
        ax.text(max_info[i][1], max_info[i][0], max_info[i][1], fontsize=8)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.xlabel("Initial location of a deletion mutant")

    count_data, location = np.histogram(df["del_end"], bins=np.arange(270))
    max_info = []
    for x in range(10):
        idx = np.arange(x * 27, (x + 1) * 27 - 1)
        cur = np.argmax(count_data[idx])
        max_info.append([count_data[idx][cur], location[idx][cur]])

    fig, ax = plt.subplots()
    ax = sns.histplot(data=df, x="del_end", bins=200)
    for i in range(len(max_info)):
        ax.text(max_info[i][1], max_info[i][0], max_info[i][1], fontsize=8)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.xlabel("Terminal location of a deletion mutant")

    fig, ax = plt.subplots()
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax = sns.histplot(data=df, x="ins_length", bins=200)
    ax.set_xlim([0, 50])


####################

# Miscellaneous

####################


def three_locus_comparison_plots(
    df_all,
    sample_key,
    QC_metric=[
        "tot_fastq_N",
        "valid_5_primer (read_frac)",
        "valid_3_primer (read_frac)",
        "valid_2_seq (read_frac)",
        "valid_read_structure (read_frac)",
        "valid_lines (read_frac)",
        "common_UMIs (read_frac)",
        "consensus_calling_fraction",
        "UMI_per_cell",
        "cell_number",
        "Mean_read_per_edited_UMI",
        "UMI_per_clone",
    ],
    QC_x_label=[
        "Total fastq reads",
        "valid_5_primer (read_frac)",
        "valid_3_primer (read_frac)",
        "valid_2_seq (read_frac)",
        "Read fraction (valid structure)",
        "Read fraction (valid reads)",
        "Read fraction (common UMI)",
        "Read frac. (allele calling)",
        "UMI per cell",
        "Cell number",
        "Mean reads per edited UMI",
        "UMI per clone",
    ],
    performance_metric=[
        "edit_UMI_fraction",
        "total_alleles",
        "singleton",
        "singleton_fraction",
        "total_alleles_norm_fraction",
        "singleton_norm_fraction",
        "Allele output per reads (normalized)",
    ],
    performance_x_label=[
        "Edited cell fraction (edited UMI fraction)",
        "Total allele number",
        "Singleton number",
        "Singleton fraction",
        "Percent of alleles within a locus",
        "Percent of singleton within a locus",
        "Allele output per reads (normalized)",
    ],
):
    """
    Comparing CC,TC,RC profiling, for both QC, and
    evaluating the allele diversity in each locus.

    QC_metric and QC_x_label has one-to-one correspondence
    performance_metric and performance_x_label has one-to-one correspondence
    """

    df_all["singleton_fraction"] = df_all["singleton"] / df_all["total_alleles"]
    df_all["consensus_calling_fraction"] = (
        df_all["called_UMIs_total (read_frac)"] / df_all["common_UMIs (read_frac)"]
    )

    temp = df_all["total_alleles"] / (
        df_all["tot_fastq_N"] * df_all["valid_lines (read_frac)"]
    )
    temp = temp / np.sum(temp)
    df_all["Allele output per reads (normalized)"] = temp

    df_all["UMI_per_clone"] = df_all["UMI_called"] / df_all["total_alleles"]

    for j, qc in enumerate(QC_metric):
        if qc in df_all.columns:
            g = sns.catplot(
                data=df_all,
                x="sample_id",
                y=qc,
                hue="Type",
                kind="bar",
                edgecolor=".6",
                aspect=1.2,
                hue_order=["Col", "Tigre", "Rosa"],
            )
            g.ax.set_ylabel(QC_x_label[j])
            g.ax.set_xlabel("")
            g.ax.set_title("QC")
            plt.xticks(rotation=90)
            # plt.xticks(rotation='vertical');
            plt.savefig(f"figure/{sample_key}/{qc}.pdf")
        else:
            print(f"{qc} not found in df_all")

    for j, y in enumerate(performance_metric):
        if y in df_all.columns:
            g = sns.catplot(
                data=df_all,
                x="sample_id",
                y=y,
                hue="Type",
                kind="bar",
                edgecolor=".6",
                aspect=1.2,
                hue_order=["Col", "Tigre", "Rosa"],
            )
            g.ax.set_ylabel(performance_x_label[j])
            g.ax.set_xlabel("")
            g.ax.set_title("Performance")
            plt.xticks(rotation=90)
            # plt.xticks(rotation='vertical');
            plt.savefig(f"figure/{sample_key}/{y}.pdf")
        else:
            print(f"{y} not found in df_all")


def analyze_cell_coupling_core(
    df_allele,
    selected_fates: list = None,
    short_names: list = None,
    remove_single_lineage_clone=False,
    plot_sample_number=True,
    plot_barcodes_binary=True,
    plot_barcodes_normalize=True,
    plot_cell_count=True,
    plot_hierarchy=True,
    plot_pie=False,
    plot_correlation=True,
    order_map=False,
    included_fates_N=2,
    time_info=None,
):
    """
    Given adata, analyze cell coupling in full.
    """

    adata_orig = lineage.generate_adata_sample_by_allele(
        df_allele, count_value_key="UMI_count", use_UMI=True
    )

    # a temporary fix for time_info
    if (short_names is not None) & (time_info is None):
        time_info = np.array(["HSC" in x for x in short_names]).astype(int).astype(str)

    if plot_sample_number:
        # sample number histogram
        sample_num_per_clone = (adata_orig.obsm["X_clone"] > 0).sum(0).A.flatten()
        fig, ax = plt.subplots()
        plt.hist(sample_num_per_clone)
        # plt.yscale('log')
        plt.xlabel("Number of samples per clone")
        plt.ylabel("Histogram")

    # barcode heatmap
    adata_orig.uns["data_des"] = ["coarse"]
    # cs.settings.set_figure_params(format="png", figsize=[4, 4], dpi=75, fontsize=15)
    coarse_X_clone, selected_fates = cs.tl.coarse_grain_clone_over_cell_clusters(
        adata_orig, selected_fates=selected_fates
    )

    if remove_single_lineage_clone:
        print("Warning: Remove single lineage clones")
        print("coarse_X_clone shape:", coarse_X_clone.shape)
        coarse_X_clone = coarse_X_clone[:, (coarse_X_clone > 0).sum(0) > 1]

    adata = lineage.generate_adata_from_X_clone(
        ssp.csr_matrix(coarse_X_clone), state_info=short_names, time_info=time_info
    )

    if plot_barcodes_normalize:
        cs.pl.barcode_heatmap(
            adata,
            normalize=True,
            selected_fates=short_names,
            order_map_x=False,
            order_map_y=False,
            fig_height=1.3 * plt.rcParams["figure.figsize"][0],
            fig_width=plt.rcParams["figure.figsize"][0],
        )

        if type(included_fates_N) is int:
            lineage.conditional_heatmap(
                coarse_X_clone,
                short_names,
                time_info=time_info,
                mode="or",
                normalize=True,
                included_fates=short_names[:included_fates_N],
                fig_height=1 * plt.rcParams["figure.figsize"][0],
                fig_width=plt.rcParams["figure.figsize"][0],
            )
        elif type(included_fates_N) is list:
            for j in range(len(included_fates_N)):
                lineage.conditional_heatmap(
                    coarse_X_clone,
                    short_names,
                    normalize=True,
                    time_info=time_info,
                    mode="or",
                    included_fates=short_names[
                        included_fates_N[j] : included_fates_N[j] + 1
                    ],
                    fig_height=1 * plt.rcParams["figure.figsize"][0],
                    fig_width=plt.rcParams["figure.figsize"][0],
                )

    if plot_barcodes_binary:
        cs.pl.barcode_heatmap(
            adata,
            binarize=True,
            selected_fates=short_names,
            order_map_x=False,
            order_map_y=False,
            fig_height=1.3 * plt.rcParams["figure.figsize"][0],
            fig_width=plt.rcParams["figure.figsize"][0],
        )

        if type(included_fates_N) is int:
            lineage.conditional_heatmap(
                coarse_X_clone,
                short_names,
                time_info=time_info,
                mode="or",
                binarize=True,
                included_fates=short_names[:included_fates_N],
                fig_height=1 * plt.rcParams["figure.figsize"][0],
                fig_width=plt.rcParams["figure.figsize"][0],
            )
        elif type(included_fates_N) is list:
            for j in range(len(included_fates_N)):
                lineage.conditional_heatmap(
                    coarse_X_clone,
                    short_names,
                    time_info=time_info,
                    binarize=True,
                    mode="or",
                    included_fates=short_names[
                        included_fates_N[j] : included_fates_N[j] + 1
                    ],
                    fig_height=1 * plt.rcParams["figure.figsize"][0],
                    fig_width=plt.rcParams["figure.figsize"][0],
                )

    adata.obs_names = short_names
    adata.var_names = adata_orig.var_names
    fate_names = short_names

    if plot_cell_count:
        # cell count
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.bar(
            np.arange(coarse_X_clone.shape[0]),
            (coarse_X_clone > 0).sum(1),
            tick_label=fate_names,
        )
        plt.xticks(rotation="vertical")
        plt.ylabel("Clone number")

    if plot_hierarchy:
        # hierarchy
        # cs.tl.fate_hierarchy(adata, source="X_clone", method="SW")
        # cs.pl.fate_hierarchy(adata, source="X_clone")

        # fate coupling
        cs.tl.fate_coupling(
            adata, method="SW", source="X_clone", selected_fates=short_names
        )
        # cs.settings.set_figure_params(dpi=100, figsize=(5.5, 5))
        cs.pl.fate_coupling(
            adata,
            source="X_clone",
            vmin=0,
            order_map_x=order_map,
            order_map_y=order_map,
            color_bar_label="Fate coupling (SW)",
        )

        # fate coupling
        cs.tl.fate_coupling(
            adata, method="Jaccard", source="X_clone", selected_fates=short_names
        )
        # cs.settings.set_figure_params(dpi=100, figsize=(5.5, 5))
        cs.pl.fate_coupling(
            adata,
            source="X_clone",
            vmin=0,
            color_bar_label="Fate coupling (Jaccard)",
            order_map_x=order_map,
            order_map_y=order_map,
        )

    if plot_correlation:
        coarse_X_clone = cs.tl.get_normalized_coarse_X_clone(
            adata, short_names, fate_normalize_source="X_clone"
        ).to_numpy()

        ax = cs.pl.heatmap(
            np.corrcoef(coarse_X_clone),
            order_map_x=order_map,
            order_map_y=order_map,
            x_ticks=short_names,
            y_ticks=short_names,
            color_bar_label="Pearson correlation",
            color_map=plt.cm.coolwarm,
            vmax=0.2,
            vmin=-0.2,
            fig_height=plt.rcParams["figure.figsize"][0],
            fig_width=1.2 * plt.rcParams["figure.figsize"][0],
        )
        ax.set_title("Pan-celltype correlation")

    if plot_pie:
        fig, ax = plt.subplots(figsize=(5, 5))
        plotting.plot_pie_chart(
            coarse_X_clone, fate_names=short_names, include_fate=short_names[0]
        )

    return adata


def analyze_cell_coupling(
    data_path,
    SampleList,
    df_ref,
    short_names=None,
    source=None,
    remove_single_lineage_clone=False,
    plot_sample_number=True,
    plot_barcodes_binary=True,
    plot_barcodes_normalize=True,
    plot_cell_count=True,
    plot_hierarchy=True,
    plot_pie=False,
    plot_correlation=True,
    order_map=False,
):
    """
    Analyze CARLIN clonal data, show the fate coupling etc.

    You can group different samples together through list nesting: e.g.
    SampleList=[['a','b'],'c'], and short_names=['1','2']
    Then, 'a','b' will be grouped under name '1', and 'c' will under name '2'
    Downstream clustering analysis will focus on mega cluster '1' and '2'.
    The order of SampleList will also affect the organization of the barcode heatmap

    This analysis also considers the actual cell number when the information is available
    under the root foder for the experiment, i.e., the file f"{data_path}/../../sample_info.csv"
    exists.

    Parameters
    ----------
    data_path:
        A path/to/sample_folders
    SampleList:
        A list of samples to use under this folder. Allows to nesting to group samples.
    df_ref:
        An allele bank as a reference for the expected frequency and
        concurrence across samples of an allele
    source:
        Supfix to the same name, typically {'cCARLIN','Tigre','Rosa'}.
        Needed for joint profiling
    short_names:
        A list of short names for SampleList. no nesting.
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

    if source is not None:
        Flat_SampleList = [x + f".{source}" for x in Flat_SampleList]

    df_all = car.extract_CARLIN_info(data_path, Flat_SampleList)
    df_all = df_all.merge(df_ref, on="allele", how="left")

    ignore = True
    if os.path.exists(f"{data_path}/../../sample_info.csv"):
        df_sample_info = (
            pd.read_csv(f"{data_path}/../../sample_info.csv")
            .dropna()
            .filter(["cell_number", "sample_id"])
            .rename(columns={"sample_id": "sample"})
        )
        df_all = df_all.merge(df_sample_info, on="sample", how="left")
        if not ignore:
            print("Correct UMI count information by cell number information")
            df_all["orig_UMI_count"] = df_all["UMI_count"]
            df_all["UMI_count"] = (
                df_all["UMI_count"] / df_all["cell_number"]
            )  # normalize by cell_number
            df_all["UMI_count"] = (
                df_all["UMI_count"]
                / np.max(df_all["UMI_count"])
                * np.max(df_all["orig_UMI_count"])
            )  # normalize within the column by max value
    else:
        print(
            "cell number sample_info.csv does not exist or ignore cell number information. Do not perform cell number correction for the obs_UMI_count"
        )

    df_HQ = df_all.query("invalid_alleles!=True")

    print("Clone number (before correction): {}".format(len(set(df_all["allele"]))))
    print("Cell number (before correction): {}".format(len(df_all["allele"])))
    print("Clone number (after correction): {}".format(len(set(df_HQ["allele"]))))
    print("Cell number (after correction): {}".format(len(df_HQ["allele"])))

    adata = analyze_cell_coupling_core(
        df_HQ,
        selected_fates,
        short_names,
        remove_single_lineage_clone=remove_single_lineage_clone,
        plot_sample_number=plot_sample_number,
        plot_barcodes_binary=plot_barcodes_binary,
        plot_barcodes_normalize=plot_barcodes_normalize,
        plot_cell_count=plot_cell_count,
        plot_hierarchy=plot_hierarchy,
        plot_pie=plot_pie,
        plot_correlation=plot_correlation,
        order_map=order_map,
    )
    return adata, df_all


def clonal_analysis(
    adata_sub, data_des="all", scenario="coarse", data_path=".", color_coding=None
):
    """
    Perform basic clonal analysis, including clonal heatmap, and tree generation and visualization

    This is tailored for the allele-by-mutation adata object
    """

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

        plotting.visualize_tree(
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

        plotting.visualize_tree(
            my_tree_refined,
            color_coding=color_coding,
            mode="c",
            data_des=f"{data_des}_refined",
            figure_path=cs.settings.figure_path,
            dpi=300,
        )


def visualize_sc_CARLIN_data(df_sc_data):
    df_plot = (
        df_sc_data.groupby(["locus", "library"])
        .agg(
            cell_number=("cell_bc", lambda x: len(set(x))),
            clone_number=("clone_id", lambda x: len(set(x))),
        )
        .reset_index()
    )
    df_plot["library"] = df_plot["library"].apply(lambda x: x.split("_")[0][:-3])
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_plot, x="cell_number", y="clone_number", hue="locus")

    fig, ax = plt.subplots()
    sns.barplot(data=df_plot, x="library", y="clone_number", hue="locus")
    plt.xticks(rotation=90)
    plt.ylim([0, 50])

    fig, ax = plt.subplots()
    sns.barplot(data=df_plot, x="library", y="cell_number", hue="locus")
    plt.xticks(rotation=90)
    plt.ylim([0, 90])

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_sc_data.filter(
            ["clone_id", "read", "CARLIN_length", "locus"], axis=1
        ).drop_duplicates(),
        x="read",
        y="CARLIN_length",
        hue="locus",
    )
    plt.xscale("log")

    # filter to count only unique alleles
    g = sns.FacetGrid(
        df_sc_data.filter(
            ["clone_id", "sample_count", "locus"], axis=1
        ).drop_duplicates(),
        col="locus",
    )
    g.map(sns.histplot, "sample_count")

    # filter to count only unique alleles
    g = sns.FacetGrid(
        df_sc_data.filter(
            ["clone_id", "expected_frequency", "locus"], axis=1
        ).drop_duplicates(),
        col="locus",
    )
    g.map(sns.histplot, "expected_frequency")

    g = sns.FacetGrid(
        df_sc_data.filter(
            ["clone_id", "CARLIN_length", "locus"], axis=1
        ).drop_duplicates(),
        col="locus",
    )
    g.map(sns.histplot, "CARLIN_length")

    # sns.histplot(data=df_sc_data,x='CARLIN_length',bins=50,hue='locus',multiple='fill',element='poly')
    # #plt.xscale('log')

    fig, ax = plt.subplots()
    plotting.plot_venn3(
        df_sc_data[df_sc_data.locus == "CC"]["RNA_id"],
        df_sc_data[df_sc_data.locus == "TC"]["RNA_id"],
        df_sc_data[df_sc_data.locus == "RC"]["RNA_id"],
        labels=["CC", "TC", "RC"],
    )
    plt.title("Cell number")
