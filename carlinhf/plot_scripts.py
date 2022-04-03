import os

import cospar as cs
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams

import carlinhf.lineage as lineage

cs.settings.set_figure_params(
    format="pdf", figsize=[4, 3.5], dpi=150, fontsize=14, pointsize=5
)
rcParams["legend.handlelength"] = 1.5


def add_shade(ax):
    l1 = ax.lines[0]
    l2 = ax.lines[1]
    # Get the xy data from the lines so that we can shade
    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]
    x2 = l2.get_xydata()[:, 0]
    y2 = l2.get_xydata()[:, 1]
    ax.fill_between(x1, y1, color="#2b83ba", alpha=0.5)
    ax.fill_between(x2, y2, color="#d7191c", alpha=0.1)
    return ax


def mutation_statistics(df_LL, df_SB, sample_key):
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
