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
    """
    Add shaded regions between two lines on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object containing two lines to shade between.
    color : list of str, optional
        Colors for the shaded regions. Default is ["#2b83ba", "#d7191c"].

    Returns
    -------
    matplotlib.axes.Axes
        The modified axis object.
    """
    l1 = ax.lines[0]
    l2 = ax.lines[1]
    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]
    x2 = l2.get_xydata()[:, 0]
    y2 = l2.get_xydata()[:, 1]
    ax.fill_between(x1, y1, color=color[0], alpha=0.5)
    ax.fill_between(x2, y2, color=color[1], alpha=0.1)
    return ax


def add_shade_1(ax, color="#2b83ba"):
    """
    Add a shaded region under a single line on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object containing the line to shade under.
    color : str, optional
        Color for the shaded region. Default is "#2b83ba".

    Returns
    -------
    matplotlib.axes.Axes
        The modified axis object.
    """
    l1 = ax.lines[0]
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
    Plot a pie chart showing cell number overlap between different fates.

    Parameters
    ----------
    matrix : ndarray
        A fate-by-clone matrix where matrix.shape[0] should equal len(fate_names).
    fate_names : array-like
        Names of the fate types corresponding to matrix rows.
    include_fate : str, optional
        If provided, only include clones that appear in this fate.
    labeldistance : float, optional
        Distance of labels from center. Default is 1.1.
    rotatelabels : bool, optional
        Whether to rotate labels. Default is True.
    counterclock : bool, optional
        Direction of pie slices. Default is False (clockwise).
    textprops : dict, optional
        Properties passed to matplotlib text. Default is {"fontsize": 12}.
    **kwargs
        Additional arguments passed to plt.pie.

    Returns
    -------
    None
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


def plot_venn3(
    data_1,
    data_2,
    data_3,
    labels=["1", "2", "3"],
    set_colors=("#3274A1", "#E1812C", "#3B923B"),
    alpha=0.5,
    text_font_size=16,
):
    """
    Plot a three-set Venn diagram.

    Parameters
    ----------
    data_1 : iterable
        First set of elements.
    data_2 : iterable
        Second set of elements.
    data_3 : iterable
        Third set of elements.
    labels : list of str, optional
        Labels for each set. Default is ["1", "2", "3"].
    set_colors : tuple of str, optional
        Colors for each set. Default is ("#3274A1", "#E1812C", "#3B923B").
    alpha : float, optional
        Transparency of the sets. Default is 0.5.
    text_font_size : int, optional
        Font size for labels. Default is 16.

    Returns
    -------
    None
    """
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
        set_colors=set_colors,
        alpha=alpha,
    )
    venn3_circles([set_1, set_2, set_3], linestyle="-", linewidth=0.5, color="grey")
    for text in vd3.set_labels:
        if text != None:
            text.set_fontsize(text_font_size)
    for text in vd3.subset_labels:
        if text != None:
            text.set_fontsize(text_font_size)


def plot_venn2(data_1, data_2, labels=["1", "2"], set_colors=("#c4e6ff", "#F4ACB7")):
    """
    Plot a two-set Venn diagram.

    Parameters
    ----------
    data_1 : iterable
        First set of elements.
    data_2 : iterable
        Second set of elements.
    labels : list of str, optional
        Labels for each set. Default is ["1", "2"].
    set_colors : tuple of str, optional
        Colors for each set. Default is ("#c4e6ff", "#F4ACB7").

    Returns
    -------
    None
    """
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
        set_colors=set_colors,
        alpha=0.8,
    )
    venn2_circles([set_1, set_2], linestyle="-", linewidth=0.5, color="grey")
    for text in vd3.set_labels:
        text.set_fontsize(16)
    for text in vd3.subset_labels:
        text.set_fontsize(16)


def visualize_tree(
    input_tree,
    color_coding: dict = None,
    mode="r",
    width=60,
    height=60,
    dpi=300,
    data_des="tree",
    figure_path=".",
    line_width=0,
    marker_size_internal=5,
    marker_size_leaf=5,
):
    """
    Visualize a tree structured in ete3 style.

    This function provides options to color code the leaves of the tree. For example,
    if leaf nodes represent single cells and some are more similar than others, imposing
    the same color for similar cells allows visualization of how the tree structure
    recapitulates the similarity of these cells.

    Requires a full installation of ete3 packages (not part of default cospar installation):
    ete3, ete_toolchain, PyQt5, QtPy.

    Parameters
    ----------
    input_tree : ete3.Tree
        A tree stored in ete3 style. Can be the output from running
        `cs.tl.fate_hierarchy(adata, source="X_clone")`, where the resulting tree
        will be stored at `my_tree = adata.uns["fate_hierarchy_X_clone"]["tree"]`.
    color_coding : dict, optional
        Dictionary mapping leaf names to specific colors.
        Example: `{'node_1':"#e5f5f9", 'node_2':"#99d8c9", ...}`.
    mode : str, optional
        Plotting mode: 'r' for rectangular, 'c' for circular. Default is 'r'.
    width : int, optional
        Width of the tree plot. Default is 60.
    height : int, optional
        Height of the tree plot. Default is 60.
    dpi : int, optional
        Resolution of the tree plot. Default is 300.
    data_des : str, optional
        Label for saving the figure (figure name). Default is "tree".
    figure_path : str, optional
        Directory to save the figure. Default is ".".
    line_width : int, optional
        Width of branch lines. Default is 0.
    marker_size_internal : int, optional
        Size of internal node markers. Default is 5.
    marker_size_leaf : int, optional
        Size of leaf node markers. Default is 5.

    Returns
    -------
    None
    """

    from ete3 import AttrFace, NodeStyle, Tree, TreeStyle, faces
    from IPython.display import Image, display

    def layout(node):
        if node.is_leaf(): # this is the part showing the leaf
            N = AttrFace("name", fsize=5)
            faces.add_face_to_node(N, node, 100, position="aligned")
            # pass

    if color_coding is not None:
        print("coding")
        for n in input_tree.traverse(): # internal node
            nst1 = NodeStyle(size=marker_size_internal, fgcolor="#f0f0f0",vt_line_width=line_width,hz_line_width=line_width)
            n.set_style(nst1)

        for n in input_tree:
            for key, value in color_coding.items():
                if n.name==key:
                    nst1 = NodeStyle(size=marker_size_leaf,hz_line_width=line_width, fgcolor="#000000")
                    nst1["bgcolor"] = value
                    n.set_style(nst1)

    ts = TreeStyle()
    #ts.layout_fn = layout # layout not used. It will add faces to each node, and each fates is the leaf name
    ts.mode = mode
    ts.show_leaf_name = False
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
    Plot a density histogram with log-log scale.

    The result is independent of bins but not very accurate.

    Parameters
    ----------
    x : array-like
        Data to plot.
    bins : int, optional
        Number of bins. Default is 50.
    cutoff_y : int, optional
        Minimum count threshold for bins to be displayed. Default is 5.
    cutoff_x : float, optional
        Maximum value threshold for x to be displayed. Default is None.
    data_des : str, optional
        Description for plot title. Default is None.

    Returns
    -------
    x_var : ndarray
        X values used for plotting.
    y_var : ndarray
        Y values (density) used for plotting.
    """
    fig, ax = plt.subplots()
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
    Plot a histogram with log scale on both axes.

    The result depends on bins and is not very accurate.

    Parameters
    ----------
    x : array-like
        Data to plot.
    bins : int, optional
        Number of bins. Default is 50.
    cutoff_y : int, optional
        Minimum count threshold for bins to be displayed. Default is 5.
    cutoff_x : float, optional
        Maximum value threshold for x to be displayed. Default is None.
    data_des : str, optional
        Description for plot title. Default is None.

    Returns
    -------
    x_var : ndarray
        X values used for plotting.
    y_var : ndarray
        Y values (density) used for plotting.
    """
    fig, ax = plt.subplots()
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
    Plot a cumulative distribution with log-log scale.

    The result is independent of bins and is accurate.

    Parameters
    ----------
    X : array-like
        Data to plot.
    data_des : str, optional
        Description for plot title. Default is None.
    cutoff_x_up : float, optional
        Upper bound for x values. Default is None.
    cutoff_x_down : float, optional
        Lower bound for x values. Default is 1.

    Returns
    -------
    x_var : ndarray
        X values used for plotting.
    y_var : ndarray
        Y values (cumulative counts) used for plotting.
    """
    fig, ax = plt.subplots()
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
