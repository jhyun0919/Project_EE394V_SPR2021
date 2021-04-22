import matlab.engine
import os
import numpy as np
from tqdm import trange
import pickle


def standardize_data_type(data, d_type="float64"):
    for key_i in data.keys():
        for key_j in data[key_i].keys():
            data[key_i][key_j] = np.asarray(data[key_i][key_j], dtype=d_type)

    return data


def get_active_gen_constraints(data):
    p_g = data["gen_info"]["p_g"]
    p_g_lim = data["gen_info"]["p_g_lim"]

    active_constraints = np.zeros_like(p_g_lim)

    for g_idx in range(p_g_lim.shape[0]):
        if p_g_lim[g_idx][0] != p_g_lim[g_idx][1]:
            # max constraint
            if p_g[g_idx] >= p_g_lim[g_idx][0]:
                active_constraints[g_idx][0] = 1
            # min constraint
            if p_g[g_idx] <= p_g_lim[g_idx][1]:
                active_constraints[g_idx][1] = 1

    return active_constraints


def get_active_flow_constraints(data):
    p_f = data["flow_info"]["p_f"]
    p_f_lim = data["flow_info"]["p_f_lim"]

    active_constraints = np.zeros_like(p_f_lim)

    for f_idx in range(p_f_lim.shape[0]):
        if p_f_lim[f_idx][0] != p_f_lim[f_idx][1]:
            # max constraint
            if p_f[f_idx] >= p_f_lim[f_idx][0]:
                active_constraints[f_idx][0] = 1
            # min constraint
            if p_f[f_idx] <= p_f_lim[f_idx][1]:
                active_constraints[f_idx][1] = 1

    return active_constraints


def merge_active_constraints(gen_active, flow_active):
    gen_active = gen_active.reshape(gen_active.shape[0] * 2)
    flow_active = flow_active.reshape(flow_active.shape[0] * 2)
    return np.hstack((gen_active, flow_active))


def create_dataset(test_case, dataset_size, std_scaler=0.03, d_type="float32"):
    """
    create a dataset by mat engine (matpower),
    and return it.

    Args:
        test_case (str): a name of test case.
        dataset_size (int): the size of the dataset.
        std_scaler (float, optional): coefficient to scale the uncertainty. Defaults to 0.03.

    Returns:
        dict: the created dataset.
    """
    # create a dataset
    print("> creating dataset with {}".format(test_case))

    x = []
    y = []

    org_dir = os.getcwd()
    os.chdir("./matpower7.1/")

    # start mat-engine
    eng = matlab.engine.start_matlab()

    for _ in trange(dataset_size):
        # do opf solver
        data = eng.dc_opf_solver(test_case, std_scaler)
        data = standardize_data_type(data, d_type)

        # assing x data
        x.append(data["w_info"]["w"].reshape(data["w_info"]["w"].shape[1]))
        # assgin y data
        gen_active = get_active_gen_constraints(data)
        flow_active = get_active_flow_constraints(data)
        active_constraints = merge_active_constraints(gen_active, flow_active)
        y.append(active_constraints)

    g = {
        "bus_idx": data["bus_info"]["bus_idx"].squeeze(),
        "fbus2tbus": data["flow_info"]["bus2bus"],
    }

    eng.quit()
    os.chdir(org_dir)

    dataset = {"x": np.array(x), "y": np.array(y), "g": g}

    return dataset


def save_dataset(test_case, dataset, dataset_size):
    """
    save the dataset to the designated directory.

    Args:
        test_case (str): a name of test case
        dataset ([type]): a dataset
        dataset_size (int): the size of the dataset
    """
    file_name = test_case.split(".")[0]
    file_path = "./../../data/" + str(dataset_size)

    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_dir = os.path.join(file_path, file_name + ".pickle")
    outfile = open(file_dir, "wb")
    pickle.dump(dataset, outfile)
    outfile.close()


def build_datasets(test_cases, dataset_size):
    """
    build datasets for model training with test cases.

    Args:
        test_cases (list): names of test cases
        dataset_size (int): dataset size
    """
    for test_case in test_cases:
        # create a dataset
        dataset = create_dataset(test_case, dataset_size)
        # save the dataset
        save_dataset(test_case, dataset, dataset_size)
