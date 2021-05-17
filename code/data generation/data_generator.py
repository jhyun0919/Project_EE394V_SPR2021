import matlab.engine
import os
import numpy as np
from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
import pandas as pd
from tqdm import trange
import pickle
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["figure.figsize"] = [8, 6]


def standardize_data_type(data, d_type="float32"):
    for key_i in data.keys():
        for key_j in data[key_i].keys():
            data[key_i][key_j] = np.asarray(data[key_i][key_j], dtype=d_type)

    return data


# def get_active_gen_constraints(data):
#     p_g = data["gen_info"]["p_g"]
#     p_g_lim = data["gen_info"]["p_g_lim"]

#     active_constraints = np.zeros_like(p_g_lim)

#     for g_idx in range(p_g_lim.shape[0]):
#         if p_g_lim[g_idx][0] != p_g_lim[g_idx][1]:
#             # max constraint
#             if p_g[g_idx] >= p_g_lim[g_idx][0]:
#                 active_constraints[g_idx][0] = 1
#             # min constraint
#             if p_g[g_idx] <= p_g_lim[g_idx][1]:
#                 active_constraints[g_idx][1] = 1

#     return active_constraints


# def get_active_flow_constraints(data):
#     p_f = data["flow_info"]["p_f"]
#     p_f_lim = data["flow_info"]["p_f_lim"]

#     active_constraints = np.zeros_like(p_f_lim)

#     for f_idx in range(p_f_lim.shape[0]):
#         if p_f_lim[f_idx][0] != p_f_lim[f_idx][1]:
#             # max constraint
#             if p_f[f_idx] >= p_f_lim[f_idx][0]:
#                 active_constraints[f_idx][0] = 1
#             # min constraint
#             if p_f[f_idx] <= p_f_lim[f_idx][1]:
#                 active_constraints[f_idx][1] = 1

#     return active_constraints


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

    _, counts = np.unique(data["gen_info"]["gen2bus"], return_counts=True)
    max_num_colocated_gen = np.max(counts)
    num_buses = len(data["bus_info"]["bus_idx"])

    gen2bus_idx = list(np.squeeze(data["gen_info"]["gen2bus"]))
    gen2bus_idx_ = list(np.squeeze(data["gen_info"]["gen2bus"]))

    # 1: upper limit, 2: lower limit, 0: o/w
    active_gen_constraints = np.zeros((num_buses, max_num_colocated_gen))
    for bus_idx in range(num_buses):
        gen_appear = 0
        for _ in range(max_num_colocated_gen):
            if (bus_idx + 1) in gen2bus_idx_:
                g_idx = gen2bus_idx.index(bus_idx + 1)
                if active_constraints[g_idx][0] == 1:
                    active_gen_constraints[bus_idx][gen_appear] = 1
                elif active_constraints[g_idx][1] == 1:
                    active_gen_constraints[bus_idx][gen_appear] = -1
                gen2bus_idx_.remove(bus_idx + 1)

                gen_appear += 1

    return active_gen_constraints


def merge_active_constraints(gen_active, flow_active):
    gen_active = gen_active.reshape(gen_active.shape[0] * 2)
    flow_active = flow_active.reshape(flow_active.shape[0] * 2)
    return np.hstack((gen_active, flow_active))


def create_dataset(
    test_case, dataset_size, std_scaler=0.03, d_type="float32", gen_only=False
):
    # create a dataset
    print("> creating dataset with {}".format(test_case))

    X = []
    Y = []

    org_dir = os.getcwd()
    os.chdir("./matpower7.1/")

    # start mat-engine
    eng = matlab.engine.start_matlab()

    for _ in trange(dataset_size):
        # do opf solver
        while True:
            data = eng.dc_opf_solver(test_case, std_scaler)
            data = standardize_data_type(data, d_type)
            if data["success_info"]["success"] == 1:
                break

        # assing x data
        x = data["w_info"]["w"].reshape(data["w_info"]["w"].shape[1], 1)
        # assgin y data
        # gen_active = get_active_gen_constraints(data)

        # gen_only opt
        if not gen_only:
            # flow_active = get_active_flow_constraints(data)
            # active_constraints = merge_active_constraints(gen_active, flow_active)
            # y = active_constraints
            pass
        else:
            y = get_active_gen_constraints(data)

        assert x.shape[0] == y.shape[0]

        # assign g data
        g = {
            "bus_idx": data["bus_info"]["bus_idx"].squeeze(),
            "fbus2tbus": data["flow_info"]["bus2bus"],
            "gen_bus_idx": data["gen_info"]["gen2bus"],
        }

        X.append(np.expand_dims(x, axis=0))
        Y.append(np.expand_dims(y, axis=0))

    X = np.vstack(X)
    Y = np.vstack(Y)
    save_dataset(X, Y, g, test_case, dataset_size, std_scaler, gen_only)

    eng.quit()
    os.chdir(org_dir)


def get_file_name(test_case, dataset_size, std_scaler):
    file_name = test_case.split(".")[0]

    # file dir: data size
    file_dir = "./../../../data/size_" + str(dataset_size)
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    # file dir: std
    file_dir = file_dir + "/std_" + str(std_scaler)
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    file_name = os.path.join(file_dir, file_name)

    return file_name


def save_dataset(x, y, g, test_case, dataset_size, std_scaler, gen_only=False):
    file_name = get_file_name(test_case, dataset_size, std_scaler)
    if gen_only:
        file_name = file_name + "_gen_only"

    # df_X = pd.DataFrame({"X": list(x)}).T
    # df_Y = pd.DataFrame({"Y": list(y)}).T

    # if os.path.isfile(file_name + "_feautre.csv"):
    #     df_X.to_csv(file_name + "_feautre.csv", mode="a", header=False)
    #     df_Y.to_csv(file_name + "_label.csv", mode="a", header=False)
    # else:
    #     df_X.to_csv(file_name + "_feautre.csv", mode="w", header=False)
    #     df_Y.to_csv(file_name + "_label.csv", mode="w", header=False)

    outfile = open(file_name + "_feature.pickle", "wb")
    pickle.dump(x, outfile)
    outfile.close()

    outfile = open(file_name + "_label.pickle", "wb")
    pickle.dump(y, outfile)
    outfile.close()

    outfile = open(file_name + "_graph.pickle", "wb")
    pickle.dump(g, outfile)
    outfile.close()


def load_dataset(
    test_case,
    dataset_size,
    std_scaler,
    test_type="default",
    relative_dir="./../../",
    gen_only=False,
):
    if test_type != "default":
        test_case = test_case.split(".")[0] + "__" + test_type + ".m"
    case_name = test_case.split(".")[0]

    file_name = (
        relative_dir
        + "data/size_"
        + str(dataset_size)
        + "/std_"
        + str(std_scaler)
        + "/"
        + case_name
    )

    if gen_only:
        file_name = file_name + "_gen_only"

    infile = open(file_name + "_feature.pickle", "rb",)
    x = pickle.load(infile)
    infile.close()

    infile = open(file_name + "_label.pickle", "rb",)
    y = pickle.load(infile)
    infile.close()

    infile = open(file_name + "_graph.pickle", "rb",)
    g = pickle.load(infile)
    infile.close()

    # graph bus indes -1
    for key in g.keys():
        g[key] = g[key] - 1

    dataset = {"x": x, "y": y, "g": g}

    return dataset


def build_datasets(
    test_cases,
    dataset_size,
    std_scaler=0.03,
    d_type="float32",
    test_type="default",
    gen_only=False,
):
    for test_case in test_cases:
        if test_type != "default":
            test_case = test_case.split(".")[0] + "__" + test_type + ".m"
        # create a dataset
        create_dataset(test_case, dataset_size, std_scaler, d_type, gen_only)
        # save the dataset
        # save_dataset(test_case, dataset, dataset_size, std_scaler)

