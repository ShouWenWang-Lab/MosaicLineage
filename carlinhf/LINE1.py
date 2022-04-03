import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress

# plt.rcParams["text.usetex"] = True

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


def estimate_exponent(X, xmin=None):
    X = np.array(X)
    if xmin is None:
        xmin = np.min(X)
    X_new = X[X >= xmin]
    return 1 + len(X_new) / np.sum(np.log(X_new / xmin))


def simulate_LINE1_dynamics(
    insertion_rate=0.05,
    max_clone_N=10000,
    replication_N=10,
):
    cell_lists_new = []
    common_clone_N = 0
    current_clone_id = 0
    cell_N = 2 ** (replication_N + 1)
    temp_vector = np.zeros(max_clone_N)
    cell_lists_new.append(temp_vector)

    for j0 in range(1, replication_N + 1):
        print(f"Current generation: {j0}")
        cell_lists_old = cell_lists_new.copy()
        cell_lists_new = []
        for j, temp_vector in enumerate(cell_lists_old):
            # print("before:", temp_vector)
            for __ in range(2):  # generate two daughters
                temp_vector_1, current_clone_id = simulate_insertion(
                    temp_vector, current_clone_id, insertion_rate=insertion_rate
                )
                cell_lists_new.append(temp_vector_1)

    simu_clone_annot_0 = np.array(cell_lists_new)
    valid_clone_idx = simu_clone_annot_0.sum(0) > 0
    simu_clone_annot = simu_clone_annot_0[:, valid_clone_idx]

    return simu_clone_annot


def power_law_from_double_exp(
    a=np.log(2), b=np.log(2), insertion_rate=0.5, generation=12, sp_ratio=1
):
    clone_size_ideal = []
    generation_all = []
    for j in range(1, generation + 1):
        print(f"generation: {j}")
        cell_N = int(np.exp(a * j))
        eventual_size = int(
            np.exp(b * (generation - j))
        )  # due to cell death etc, the eventual size of a clone is not 2**generation
        # clone_size_ideal += list(np.random.exponential(eventual_size, size=cell_N))
        clone_size_ideal += [eventual_size] * cell_N
        generation_all += [generation - j] * cell_N
    clone_size_ideal = np.array(clone_size_ideal).astype(int)
    clone_size_ideal = clone_size_ideal[clone_size_ideal > 0]
    total_N = len(clone_size_ideal)
    insertion_events = np.random.rand(total_N) < insertion_rate
    clone_size_actual = clone_size_ideal[insertion_events]

    if sp_ratio == 1:
        clone_size_actual_sp = clone_size_actual
    else:
        clone_size_actual_sp = []
        for x in clone_size_actual:
            y = np.random.binomial(x, sp_ratio)
            clone_size_actual_sp.append(y)
        clone_size_actual_sp = np.array(clone_size_actual_sp)
        clone_size_actual_sp = clone_size_actual_sp[clone_size_actual_sp > 0]

    return clone_size_actual_sp


def simulate_insertion(vector, current_clone_id, insertion_rate=0.05):
    new_vector = vector.copy()
    if np.random.rand() < insertion_rate:
        current_clone_id = current_clone_id + 1
        new_vector[current_clone_id] = 1
    return new_vector, current_clone_id


def load_data(df0, read_depth_thresh=1):

    df1 = df0[df0["read_depth"] >= read_depth_thresh]

    ### assemble cell type info
    cell_id_array = np.array(df1["cell"])
    cell_type_array = np.array(df1["cell_type"])
    clone_id_array = np.array(df1["cluster"])
    unique_cell_id_array = np.array(list(set(cell_id_array)))
    unique_clone_array = np.array(list(set(clone_id_array)))

    cell_type_info_temp = []
    ## Annotate the cell type
    for cell_id in unique_cell_id_array:
        cell_type = list(set(cell_type_array[cell_id_array == cell_id]))[0]
        # print(cell_type)
        cell_type_info_temp.append(cell_type)

    ### sort the cell idx according to cell type
    cell_type_info_temp = np.array(cell_type_info_temp)
    sort_idx = np.argsort(cell_type_info_temp)
    cell_type_info_sort = cell_type_info_temp[sort_idx]
    unique_cell_id_array_sort = unique_cell_id_array[sort_idx]

    ## generate clone_annot where the cell idx have been sorted
    clone_annot = np.zeros((len(unique_cell_id_array_sort), len(unique_clone_array)))
    for j in range(len(cell_id_array)):
        cell_id_1 = np.nonzero(unique_cell_id_array_sort == cell_id_array[j])[0]
        clone_id_1 = np.nonzero(unique_clone_array == clone_id_array[j])[0]
        # clone_annot[cell_id_1,clone_id_1] += 1
        clone_annot[cell_id_1, clone_id_1] = 1

    return clone_annot, cell_type_info_sort
