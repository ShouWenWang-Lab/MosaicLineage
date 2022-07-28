import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

######################################

# Concise and basic plotting functions

######################################


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


def plot_pie_chart(
    matrix,
    fate_names,
    include_fate=None,
    labeldistance=1.1,
    rotatelabels=True,
    counterclock=False,
    textprops={"fontsize": 12},
    **kwargs,
):
    """
    Plot the pie chart for cell numbers overlapped between different fates. The input matrix should be a fate-by-clone matrix.

    matrix.shape[0]=len(fate_names)

    In the first step, we transform the matrix to a boelean matrix
    """

    matrix = matrix > 0
    fate_names = np.array(fate_names)
    if include_fate is None:
        matrix_sub = matrix
    else:
        assert include_fate in fate_names
        clone_idx = matrix[fate_names == include_fate, :].sum(0) > 0
        matrix_sub = matrix[:, clone_idx]

    cell_type_dict = {}
    for i in range(matrix_sub.shape[1]):
        id_tmp = tuple(sorted(np.array(fate_names)[matrix_sub[:, i]]))
        if id_tmp not in cell_type_dict.keys():
            cell_type_dict[id_tmp] = 1  # initial cell number
        else:
            cell_type_dict[id_tmp] += 1

    your_data = dict(sorted(cell_type_dict.items()))
    labels = []
    sizes = []

    for x, y in your_data.items():
        tmp = list(x)
        tmp.append(y)
        labels.append(tmp)
        sizes.append(y)

    # Plot
    plt.pie(
        sizes,
        labels=labels,
        labeldistance=labeldistance,
        rotatelabels=rotatelabels,
        counterclock=counterclock,
        textprops=textprops,
        **kwargs,
    )

    plt.axis("equal")


def plot_venn3(data_1, data_2, data_3, labels=["1", "2", "3"]):

    set_1 = set(data_1)
    set_2 = set(data_2)
    set_3 = set(data_3)

    from matplotlib import pyplot as plt
    from matplotlib_venn import (
        venn2,
        venn2_circles,
        venn2_unweighted,
        venn3,
        venn3_circles,
    )

    vd3 = venn3(
        [set_1, set_2, set_3],
        set_labels=labels,
        set_colors=("#c4e6ff", "#F4ACB7", "#9D8189"),
        alpha=0.8,
    )
    venn3_circles([set_1, set_2, set_3], linestyle="-", linewidth=0.5, color="grey")
    for text in vd3.set_labels:
        text.set_fontsize(16)
    for text in vd3.subset_labels:
        text.set_fontsize(16)


def plot_venn2(data_1, data_2, labels=["1", "2"]):

    set_1 = set(data_1)
    set_2 = set(data_2)

    from matplotlib import pyplot as plt
    from matplotlib_venn import (
        venn2,
        venn2_circles,
        venn2_unweighted,
        venn3,
        venn3_circles,
    )

    vd3 = venn2(
        [set_1, set_2],
        set_labels=labels,
        set_colors=("#c4e6ff", "#F4ACB7"),
        alpha=0.8,
    )
    venn2_circles([set_1, set_2], linestyle="-", linewidth=0.5, color="grey")
    for text in vd3.set_labels:
        text.set_fontsize(16)
    for text in vd3.subset_labels:
        text.set_fontsize(16)


def visualize_tree(
    input_tree,
    color_coding=None,
    mode="r",
    width=60,
    height=60,
    dpi=300,
    data_des="tree",
    figure_path=".",
):
    """
    mode: r or c
    """

    from ete3 import AttrFace, NodeStyle, Tree, TreeStyle, faces
    from IPython.display import Image, display

    def layout(node):
        if node.is_leaf():
            N = AttrFace("name", fsize=5)
            faces.add_face_to_node(N, node, 100, position="aligned")
            # pass

    if color_coding is not None:
        print("coding")
        for n in input_tree.traverse():
            nst1 = NodeStyle(size=1, fgcolor="#f0f0f0")
            n.set_style(nst1)

        for n in input_tree:
            for key, value in color_coding.items():
                if n.name.startswith(key):
                    nst1 = NodeStyle(size=1)
                    nst1["bgcolor"] = value
                    n.set_style(nst1)

    ts = TreeStyle()
    ts.layout_fn = layout
    ts.show_leaf_name = False
    ts.mode = mode
    # ts.extra_branch_line_color = "red"
    # ts.extra_branch_line_type = 0
    input_tree.render(
        os.path.join(figure_path, f"{data_des}.pdf"),
        tree_style=ts,
        w=width,
        h=height,
        units="mm",
    )
    input_tree.render(
        os.path.join(figure_path, f"{data_des}.png"),
        tree_style=ts,
        w=width,
        h=height,
        dpi=dpi,
        units="mm",
    )

    display(Image(filename=os.path.join(figure_path, f"{data_des}.png")))


#######################
### Power law related
#######################


def plot_density(x, bins=50, cutoff_y=5, cutoff_x=None, data_des=None):
    """
    The result is indepent on bins, not very accurate
    """
    fig, ax = plt.subplots()
    hist_, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    hist, bins = np.histogram(x, bins=logbins)

    valid_idx = hist >= cutoff_y
    x_var = bins[1:][valid_idx]
    # y_var = data[0][valid_idx]
    y_var = hist / (logbins[1:] - logbins[0])
    y_var = y_var[valid_idx]
    if cutoff_x is not None:
        valid_idx = x_var <= cutoff_x
        x_var = x_var[valid_idx]
        y_var = y_var[valid_idx]

    plt.loglog(x_var, y_var)
    line_resu = linregress(np.log(x_var), np.log(y_var))
    if data_des is None:
        plt.title(f"Slope: {line_resu.slope:.2f}")
    else:
        plt.title(f"Slope ({data_des}): {line_resu.slope:.2f}")
    plt.xlabel("Clone size")
    plt.ylabel("Count")
    return x_var, y_var


def plot_loghist(x, bins=50, cutoff_y=5, cutoff_x=None, data_des=None):
    """
    The result is depent on bins, not very accurate
    """
    fig, ax = plt.subplots()
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    data = plt.hist(x, bins=logbins, log=True)
    plt.xscale("log")

    valid_idx = data[0] >= cutoff_y
    x_var = data[1][1:][valid_idx]
    y_var = data[0][valid_idx]
    if cutoff_x is not None:
        valid_idx = x_var <= cutoff_x
        x_var = x_var[valid_idx]
        y_var = y_var[valid_idx]

    plt.loglog(x_var, y_var)
    line_resu = linregress(np.log(x_var), np.log(y_var))
    if data_des is None:
        plt.title(f"Slope: {line_resu.slope:.2f}")
    else:
        plt.title(f"Slope ({data_des}): {line_resu.slope:.2f}")
    plt.xlabel("Clone size")
    plt.ylabel("Count")
    return x_var, y_var


def plot_cumu(X, data_des=None, cutoff_x_up=None, cutoff_x_down=1):
    """
    The result is indepent on bins, accurate
    """
    fig, ax = plt.subplots()
    sorted_X_sp = np.array(list(sorted(set(X))))
    cumu_data = np.zeros(len(sorted_X_sp))
    for j, x in enumerate(sorted_X_sp):
        cumu_data[j] = np.sum(X >= x)
    cumu_data = cumu_data

    valid_idx = sorted_X_sp >= cutoff_x_down
    if cutoff_x_up is not None:
        valid_idx = valid_idx & (sorted_X_sp <= cutoff_x_up)

    x_var = sorted_X_sp[valid_idx]
    y_var = cumu_data[valid_idx]
    plt.loglog(x_var, y_var)
    line_resu = linregress(np.log(x_var), np.log(y_var))
    if data_des is None:
        plt.title(f"Slope: {line_resu.slope:.2f}")
    else:
        plt.title(f"Slope ({data_des}): {line_resu.slope:.2f}")
    plt.xlabel("Clone size (x)")
    plt.ylabel("Cumulative count (>=x)")
    return x_var, y_var
