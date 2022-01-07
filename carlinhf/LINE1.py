import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress

#######################
### Power law related
#######################


def plot_loghist(x, bins):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    data = plt.hist(x, bins=logbins, log=True)
    plt.xscale("log")

    valid_idx = data[0] > 0
    x_var = data[1][1:][valid_idx]
    y_var = data[0][valid_idx]
    plt.loglog(x_var, y_var)
    line_resu = linregress(np.log(x_var), np.log(y_var))
    plt.title(f"Slope): {line_resu.slope:.2f}")
    plt.xlabel("Clone size")
    plt.ylabel("Count")


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


def power_law_from_double_exp(a=2, b=2, insertion_rate=0.5, generation=12, sp_ratio=1):
    clone_size_ideal = []
    generation_all = []
    for j in range(1, generation + 1):
        cell_N = a ** j
        eventual_size = b ** (
            generation - j
        )  # due to cell death etc, the eventual size of a clone is not 2**generation
        # clone_size_ideal += list(np.random.exponential(eventual_size,size=cell_N))
        clone_size_ideal += [eventual_size] * cell_N
        generation_all += [generation - j] * cell_N
    clone_size_ideal = np.array(clone_size_ideal)
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
