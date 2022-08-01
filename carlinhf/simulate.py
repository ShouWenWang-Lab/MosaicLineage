import numpy as np
import pandas as pd

import carlinhf.lineage as lineage

rng = np.random.default_rng()

##########################

# functions for simulation

##########################

def power_law_from_double_exp(
    a=np.log(2), b=np.log(2), insertion_rate=0.5, generation=12, sp_ratio=1
):
    """ "
    Simulate power law distribution from a double exponential model
    """
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


def generate_synthetic_alleles(
    df_allele: pd.DataFrame,
    target_sample_N: int = 10**4,
    max_mutation_N: int = 5,
    random_seed=123,
):
    """
    Parameters
    ----------
    df_allele:
        pandas dataframe for allele count
    target_sample_N:
        target sequence number to sample
    max_mutation_N:
        Maximum number of mutations in a synthetic allele. This is usefual because if the mutation_N is too big,
        like 10, then it is really hard to find 5 mutations that would satisfy the constraints of an valid allele.
    random_seed:
        random seed of the generation process. Using the same seed, the result is deterministic.

    Returns
    -------
    df_synthesis:
        pandas dataframe for synthetic allele count
    """

    np.random.seed(random_seed)

    ## extract the mutation data from the allele data
    df_mutation = lineage.mutation_frequency(df_allele, plot=False)
    norm_factor = df_mutation["UMI_count"].sum()
    df_mutation["Frequency"] = df_mutation["UMI_count"] / norm_factor

    ## extract the start and end position of a mutation
    start_L = np.zeros(len(df_mutation)) - 10
    end_L = np.zeros(len(df_mutation)) - 10
    for j, x in enumerate(df_mutation["mutation"]):
        if "del" in x:  # del or delins
            temp = x.split("del")[0].split("_")
            start_L[j] = temp[0]
            end_L[j] = temp[1]
        elif "ins" in x:  # ins
            temp = x.split("ins")[0].split("_")
            start_L[j] = temp[0]
            end_L[j] = temp[1]
        elif ">" in x:
            temp = x.split(">")[0][:-1]
            start_L[j] = temp
            end_L[j] = temp
    df_mutation["start_position"] = start_L.astype(int)
    df_mutation["end_position"] = end_L.astype(int)

    ## extract mutation number histogram
    mut_per_allele = lineage.mutations_per_allele(df_allele)
    mut_per_UMI = np.concatenate(
        [[mut_per_allele[i]] * int(x) for i, x in enumerate(df_allele["UMI_count"])]
    )
    mut_hist_y, mut_hist_x = np.histogram(mut_per_UMI, bins=np.arange(17))
    mut_hist_UMI = mut_hist_y / np.sum(mut_hist_y)

    ## generate data for different types of mutations, within each type, we normalize the sampling frequency
    df_mutation["delins"] = df_mutation["mutation"].apply(lambda x: "delins" in x)
    df_mutation["del"] = df_mutation["mutation"].apply(
        lambda x: ("del" in x) and ("ins" not in x)
    )
    df_mutation["ins"] = df_mutation["mutation"].apply(
        lambda x: ("del" not in x) and ("ins" in x)
    )
    df_mutation["others"] = df_mutation["mutation"].apply(
        lambda x: ("del" not in x) and ("ins" not in x)
    )

    df_delins = df_mutation[df_mutation["delins"]].filter(
        ["mutation", "UMI_count", "Frequency", "start_position", "end_position"]
    )
    df_delins["Frequency"] = df_delins["Frequency"] / df_delins["Frequency"].sum()
    df_del = df_mutation[df_mutation["del"]].filter(
        ["mutation", "UMI_count", "Frequency", "start_position", "end_position"]
    )
    df_del["Frequency"] = df_del["Frequency"] / df_del["Frequency"].sum()
    df_ins = df_mutation[df_mutation["ins"]].filter(
        ["mutation", "UMI_count", "Frequency", "start_position", "end_position"]
    )
    df_ins["Frequency"] = df_ins["Frequency"] / df_ins["Frequency"].sum()
    df_others = df_mutation[df_mutation["others"]].filter(
        ["mutation", "UMI_count", "Frequency", "start_position", "end_position"]
    )
    df_others["Frequency"] = df_others["Frequency"] / df_others["Frequency"].sum()
    # df_others=df_others[df_others.mutation!='[]']
    df_list = [df_delins, df_del, df_ins, df_others]

    ## estimate probability for different types of mutations
    # mutation_type_prob=[len(df_delins), len(df_del), len(df_ins), len(df_others)] # by allele
    mutation_type_prob = [
        df_delins["UMI_count"].sum(),
        df_del["UMI_count"].sum(),
        df_ins["UMI_count"].sum(),
        df_others["UMI_count"].sum(),
    ]  # by UMI
    mutation_type_prob = np.array(mutation_type_prob) / np.sum(mutation_type_prob)

    ## generate the random number before the actual computation
    prob = mut_hist_UMI[: (max_mutation_N + 1)]
    mutation_N_array = np.random.choice(
        mut_hist_x[: (max_mutation_N + 1)], size=target_sample_N, p=prob / np.sum(prob)
    )
    type_id_array = np.random.choice(
        np.arange(4),
        size=int(target_sample_N * np.mean(mutation_N_array)),
        p=mutation_type_prob,
    )
    mutation_id_list = []
    for cur_id in range(len(df_list)):
        print(f"Current id: {cur_id}")
        size = 500 * target_sample_N * mutation_type_prob[cur_id]
        type_id = np.random.choice(
            np.arange(len(df_list[cur_id])),
            size=int(size),
            p=df_list[cur_id]["Frequency"],
        )
        mutation_id_list.append(type_id)

    from tqdm import tqdm

    ## initialize the simulation
    type_id_cur = 0
    mutation_type_start = np.array([0, 0, 0, 0])
    mutation_type_start_max = np.array([len(x) - 2 for x in mutation_id_list])
    new_allele_array = []
    predicted_frequency_array = []
    for j in tqdm(range(target_sample_N)):
        mutation_N = mutation_N_array[j]
        # print(f'round {j}; current mutation number {mutation_N}')

        # select number of mutations in this allele
        type_id_end = type_id_cur + mutation_N
        type_id_temp = type_id_array[type_id_cur:type_id_end]
        type_id_cur = type_id_end

        ## select mutations from different types
        success = False
        while (success is False) and (
            mutation_type_start < mutation_type_start_max
        ).all():
            sel_mutations = []
            start_position_array = []
            end_position_array = []
            frequency = []
            for x in type_id_temp:
                start_temp = mutation_type_start[x]
                mutation_type_start[x] += 1
                mutation_index = mutation_id_list[x][start_temp]
                mutation_temp = df_list[x].iloc[mutation_index]
                sel_mutations.append(mutation_temp)
                start_position_array.append(mutation_temp["start_position"])
                end_position_array.append(mutation_temp["end_position"])
                frequency.append(mutation_temp["Frequency"])

            ## check if the selection is reasonable
            reorder_idx_start = np.argsort(start_position_array).flatten()
            reorder_idx_end = np.argsort(end_position_array).flatten()
            unique_start = len(set(start_position_array)) == len(start_position_array)
            unique_end = len(set(start_position_array)) == len(start_position_array)
            if (
                (reorder_idx_start == reorder_idx_end).all()
                and unique_start
                and unique_end
            ):  # they should satisfy the same ordering
                new_alleles = ",".join(
                    [
                        sel_mutations[i0]["mutation"]
                        for i0 in reorder_idx_start.flatten()
                    ]
                )
                new_allele_array.append(new_alleles)
                predicted_frequency_array.append(np.prod(frequency))
                success = True
            # otherwise, go for the next while loop

        if (mutation_type_start >= mutation_type_start_max).any():
            print("mutation type data insufficient. Break")
            break

    df_synthesis = pd.DataFrame(
        {"allele": new_allele_array, "Predicted_frequency": predicted_frequency_array}
    )
    df_synthesis["UMI_count"] = 1
    return (
        df_synthesis.groupby("allele")
        .agg({"UMI_count": "sum", "Predicted_frequency": "mean"})
        .reset_index()
    )
