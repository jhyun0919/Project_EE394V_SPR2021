import matplotlib.pyplot as plt
from networkx.algorithms.tree.recognition import is_forest
import seaborn as sns
import networkx as nx
import pickle
import pandas as pd

# plt.rcParams["figure.figsize"] = [12, 8]


def get_dataset(test_case, dataset_size, std_scaler, test_type="default"):
    if test_type != "default":
        test_case = test_case.split(".")[0] + "__" + test_type + ".m"
    case_name = test_case.split(".")[0]
    infile = open(
        "./size_"
        + str(dataset_size)
        + "/std_"
        + str(std_scaler)
        + "/"
        + case_name
        + ".pickle",
        "rb",
    )
    dataset = pickle.load(infile)
    infile.close()
    return dataset


def plot_graph(g):
    node_list = []
    color_map = []
    G = nx.Graph()
    for bus_i in g["bus_idx"]:
        node_list.append(bus_i)
        if bus_i in g["gen_bus_idx"]:
            color_map.append("red")
        else:
            color_map.append("green")
    G.add_nodes_from(node_list)

    for idx, f2t in enumerate(g["fbus2tbus"]):
        G.add_edge(f2t[0], f2t[1], key=idx)

    plt.figure(figsize=(12, 8))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G, pos=pos, node_color=color_map)
    plt.show()  # display


def plot_dist_X(df):
    plt.figure(figsize=(8, 20))
    plt.title("feature distribution")
    sns.violinplot(data=df, orient="h")
    plt.show()


def plot_dist_Y(df, plot_title=None):
    plt.figure(figsize=(12, 8))
    plt.plot(df.sum(axis=0))
    if plot_title is not None:
        plt.title(plot_title)
    plt.show()


def plot_two_dist_Ys(df_1, df_2, label_1, label_2, plot_title=None):
    plt.figure(figsize=(12, 8))
    plt.plot(df_1.sum(axis=0), label=str(label_1))
    plt.plot(df_2.sum(axis=0), label=str(label_2))
    plt.legend()
    if plot_title is not None:
        plt.title(plot_title)
    plt.show()


def show_info(g=None, df_X=None, df_Y=None):
    if g is not None:
        plot_graph(g)
    if df_X is not None:
        plot_dist_X(df_X)
    if df_Y is not None:
        plot_dist_Y(df_Y)


def compare_dist_Ys_diff_std_scalers(
    test_cases, dataset_size, std_scaler_1, std_scaler_2
):
    for test_case in test_cases:
        dataset_1 = get_dataset(test_case, dataset_size, std_scaler_1)
        dataset_2 = get_dataset(test_case, dataset_size, std_scaler_2)
        df_Y_1 = pd.DataFrame(dataset_1["y"])
        df_Y_2 = pd.DataFrame(dataset_2["y"])
        plot_two_dist_Ys(
            df_Y_1,
            df_Y_2,
            std_scaler_1,
            std_scaler_2,
            plot_title=test_case.split(".")[0],
        )


def compare_dist_Ys_diff_test_types(
    test_cases, dataset_size, std_scaler, test_type_1, test_type_2
):
    for test_case in test_cases:
        dataset_1 = get_dataset(test_case, dataset_size, std_scaler)
        dataset_2 = get_dataset(test_case, dataset_size, std_scaler)
        df_Y_1 = pd.DataFrame(dataset_1["y"])
        df_Y_2 = pd.DataFrame(dataset_2["y"])
        plot_two_dist_Ys(
            df_Y_1,
            df_Y_2,
            test_type_1,
            test_type_2,
            plot_title=test_case.split(".")[0],
        )
