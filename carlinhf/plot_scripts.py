import os

import cospar as cs
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams

import carlinhf.LINE1 as line1
import carlinhf.lineage as lineage

cs.settings.set_figure_params(
    format="pdf", figsize=[4, 3.5], dpi=150, fontsize=14, pointsize=5
)
rcParams["legend.handlelength"] = 1.5


def add_shade(ax, color=["#2b83ba", "#d7191c"]):
    l1 = ax.lines[0]
    l2 = ax.lines[1]
    # Get the xy data from the lines so that we can shade
    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]
    x2 = l2.get_xydata()[:, 0]
    y2 = l2.get_xydata()[:, 1]
    ax.fill_between(x1, y1, color=color[0], alpha=0.5)
    ax.fill_between(x2, y2, color=color[1], alpha=0.1)
    return ax


def add_shade_1(ax, color="#2b83ba"):
    l1 = ax.lines[0]
    # Get the xy data from the lines so that we can shade
    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]
    ax.fill_between(x1, y1, color=color, alpha=0.5)
    return ax


def remove_samples(df, removed_sample):
    ## remove some negative controls
    if removed_sample is not None:
        del_samples = []
        for x in df["sample"]:
            if np.sum([y in x for y in removed_sample]) > 0:
                del_samples.append(x)

        df_new = df[~df["sample"].isin(del_samples)]
    else:
        df_new = df
    return df_new


def mutation_statistics_box_plot(df, sample_key, removed_sample=None):
    """
    df_noMerge: a
    """

    ## remove some negative controls
    if removed_sample is None:
        removed_sample = ["merge_all"]
    else:
        removed_sample.append("merge_all")
    df_noM_new = remove_samples(df, removed_sample)

    keys = ["ave_del_len", "ave_insert_len", "ins_del_ratio_ratio_by_eventful_UMI"]
    y_labels = [
        "Average deletion length",
        "Average insertion length",
        "(Insertion #)/(deletion #): per UMI",
    ]

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


def mutation_statistics_distribution(df_LL, df_SB, sample_key):
    """
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
    add_shade(ax)
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
    add_shade(ax)
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
    add_shade(ax)
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
    add_shade(ax)
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
    add_shade(ax)
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
    add_shade(ax)
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
    add_shade(ax)
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


def mutation_statistics_distribution_single_input(df_SB, sample_key, label="Cas9"):
    """
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
    add_shade_1(ax)
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
    add_shade_1(ax)
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
    add_shade_1(ax)
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
    add_shade(ax, color=["#225ea8", "#d7301f"])
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
    add_shade_1(ax)
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
    add_shade_1(ax)
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
    add_shade_1(ax)
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
    add_shade_1(ax, color="#225ea8")
    add_shade_1(ax2, color="#d7301f")
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
    add_shade_1(ax)
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
    add_shade(ax)
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
    x_var, y_var = line1.plot_loghist(
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
    add_shade(ax)
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


def three_locus_comparison_plots(df_all, sample_key):

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

    QC_metric = [
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
    ]

    qc_x_label = [
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
    ]

    performance_metric = [
        "edit_UMI_fraction",
        "total_alleles",
        "singleton",
        "singleton_fraction",
        "total_alleles_norm_fraction",
        "singleton_norm_fraction",
        "Allele output per reads (normalized)",
    ]

    performance_x_label = [
        "Edited cell fraction (edited UMI fraction)",
        "Total allele number",
        "Singleton number",
        "Singleton fraction",
        "Percent of alleles within a locus",
        "Percent of singleton within a locus",
        "Allele output per reads (normalized)",
    ]

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
            g.ax.set_ylabel(qc_x_label[j])
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


def analyze_cell_coupling(data_path, SampleList, df_ref, short_names=None, source=None):
    """
    Analyze CARLIN clonal data, show the fate coupling etc.
    """
    pseudo_count = 0.0001  # for heatmap plot
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
        tmp_list.append(df_tmp)
    df_all_0 = pd.concat(tmp_list).rename(columns={"UMI_count": "obs_UMI_count"})
    df_HQ = df_all_0[~df_all_0.allele.isin(df_ref[df_ref["invalid_alleles"]].allele)]

    print("Clone number (before correction): {}".format(len(set(df_all_0["allele"]))))
    print("Cell number (before correction): {}".format(len(df_all_0["allele"])))
    print("Clone number (after correction): {}".format(len(set(df_HQ["allele"]))))
    print("Cell number (after correction): {}".format(len(df_HQ["allele"])))

    adata_0 = lineage.generate_adata_sample_by_allele(df_HQ)

    # sample number histogram
    sample_num_per_clone = (adata_0.obsm["X_clone"] > 0).sum(0).A.flatten()
    fig, ax = plt.subplots()
    plt.hist(sample_num_per_clone)
    # plt.yscale('log')
    plt.xlabel("Number of samples per clone")
    plt.ylabel("Histogram")

    # barcode heatmap
    adata_0.uns["data_des"] = ["coarse"]
    cs.settings.set_figure_params(
        format="png", figsize=[4, 4], dpi=75, fontsize=15, pointsize=2
    )
    cs.pl.barcode_heatmap(
        adata_0,
        selected_fates=selected_fates,
        log_transform=True,
        order_map_x=False,
        plot=False,
    )
    coarse_X_clone = adata_0.uns["barcode_heatmap"]["coarse_X_clone"]
    adata = lineage.generate_adata_v0(
        ssp.csr_matrix(coarse_X_clone), state_info=short_names
    )

    fate_names = short_names

    final_matrix = lineage.custom_hierachical_ordering(
        np.arange(coarse_X_clone.shape[0]), coarse_X_clone
    )
    cs.pl.heatmap(
        (final_matrix > 0).T + pseudo_count,
        order_map_x=False,
        order_map_y=False,
        x_ticks=fate_names,
        color_bar_label="Barcode count",
        fig_height=10,
        fig_width=8,
    )

    # cell count
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.bar(
        np.arange(coarse_X_clone.shape[0]),
        (coarse_X_clone > 0).sum(1),
        tick_label=fate_names,
    )
    plt.xticks(rotation="vertical")
    plt.ylabel("Clone number")

    # hierarchy
    cs.tl.fate_hierarchy(adata, source="X_clone")
    cs.plotting.fate_hierarchy(adata, source="X_clone")

    # fate coupling
    cs.tl.fate_coupling(adata, method="SW", source="X_clone")
    cs.settings.set_figure_params(dpi=100, figsize=(5.5, 5))
    cs.plotting.fate_coupling(adata, source="X_clone", vmin=0)

    # # correlation
    # adata.obs["time_info"] = adata.obs["state_info"].apply(lambda x: x.split("-")[1])
    # df = cs.tl.get_normalized_coarse_X_clone(adata, short_names)

    # df_t = df  # .iloc[:7]
    # coarse_X_clone = df_t.to_numpy()

    # color_map = plt.cm.coolwarm
    # # ax=cs.pl.heatmap(coarse_X_clone,order_map_x=True,order_map_y=False,
    # #                  y_ticks=df.index,fig_width=10,
    # #                  color_bar_label='Normalized fraction')
    # # ax.set_xlabel('Clone ID')
    # # ax.set_title('Intra cell-type & clone normalization')

    # ax = cs.pl.heatmap(
    #     np.corrcoef(coarse_X_clone),
    #     order_map_x=True,
    #     order_map_y=True,
    #     x_ticks=short_names,
    #     y_ticks=short_names,
    #     color_bar_label="Pearson correlation",
    #     fig_height=6,
    #     fig_width=8,
    #     color_map=color_map,
    #     # vmax=0.3,
    #     # vmin=-0.3,
    # )
    # ax.set_title("Pan-celltype correlation")
    return adata
