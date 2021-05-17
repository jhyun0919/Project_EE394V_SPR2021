import torch
import torch.nn.functional as F
import dgl
import numpy as np
from os import path, mkdir
from statistics import mean
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


###################################################
# Load Graph data
###################################################


def get_graph_structure(g_data, add_self_loop=True):
    # create a graph
    u = torch.tensor(
        np.hstack((g_data["fbus2tbus"][:, 0], g_data["fbus2tbus"][:, 1]))
    ).type(torch.int32)
    v = torch.tensor(
        np.hstack((g_data["fbus2tbus"][:, 1], g_data["fbus2tbus"][:, 0]))
    ).type(torch.int32)
    g = dgl.graph((u, v))

    if add_self_loop:
        g = dgl.add_self_loop(g)

    return g


###################################################
# Load feature & label data
###################################################


def data_prep(dataset, batch_size):
    # normalize data
    x_normed = np.nan_to_num(dataset["x"] / dataset["x"].max(axis=0), nan=0.0)

    # split data
    train_X, val_X, test_X = split_data(x_normed, verbose=False)
    train_Y, val_Y, test_Y = split_data(dataset["y"], verbose=False)

    # set data feeders
    train_loader = get_dataloader(train_X, train_Y, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_X, val_Y, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_X, test_Y, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def standardize_data_type(data, d_type="float32"):
    return np.array(data).astype(d_type)


def split_data(data, train_ratio=0.7, val_ratio=0.2, shuffle=False, verbose=False):
    if shuffle:
        np.random.shuffle(data)

    n = len(data)
    train_data = data[0 : int(n * train_ratio)]
    val_data = data[int(n * train_ratio) : int(n * (train_ratio + val_ratio))]
    test_data = data[int(n * (train_ratio + val_ratio)) :]

    if verbose:
        print("> check data shape")
        print("\ttrain shape: {}".format(train_data.shape))
        print("\tval shape: {}".format(val_data.shape))
        print("\ttest shape: {}".format(test_data.shape))

    return train_data, val_data, test_data


def get_dataloader(data_X, data_Y, batch_size, shuffle=True, drop_last=True):

    if batch_size is None:
        drop_last = False
    dataloader = DataLoader(
        dataset=TensorDataset(Tensor(data_X), Tensor(data_Y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return dataloader


###################################################
# Train & Test
###################################################


def train(
    net,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    model_name,
    dataset_name,
    num_epochs=1,
    print_step=100,
):

    print("> case: {}".format(dataset_name.split(".")[0]))
    print("> model: {}".format(model_name))
    print("- num of params: {}".format(count_parameters(net)))
    print("- training")

    train_losses = []
    val_losses = []
    min_val_loss = np.inf

    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # transfer Data to GPU if available
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # clear the parameter gradients
            optimizer.zero_grad()
            # forward propa
            preds = net(inputs)

            # calc the loss
            loss = loss_func(preds, labels)
            # back propa
            loss.backward()
            # update the wieghts
            optimizer.step()
            # update running loss
            running_loss += loss.item()
            # print statistics
            if i % print_step == (print_step - 1):
                # train loss
                train_loss = running_loss / print_step
                train_losses.append(train_loss)
                running_loss = 0.0

                # val loss
                val_loss = 0.0
                net.eval()
                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data

                        # transfer Data to GPU if available
                        if torch.cuda.is_available():
                            inputs, labels = inputs.cuda(), labels.cuda()

                        # forward propa
                        preds = net(inputs)
                        # calc the loss
                        loss = loss_func(preds, labels)
                        # update val loss
                        val_loss += loss.item()
                val_loss = val_loss / len(val_loader)
                val_losses.append(val_loss)

                # print
                print(
                    "\t[epoch: {:d}, iter: {:5d}]\tloss: {:.5f}\tval loss: {:.5f}".format(
                        epoch + 1, i + 1, train_loss, val_loss
                    )
                )

        # save the best model's weights
        if min_val_loss > val_loss:
            print(
                "\t- validation loss decreased ({:.5f}->{:.5f}): the best model was updated.".format(
                    min_val_loss, val_loss
                )
            )
            min_val_loss = val_loss
            save_model(net, model_name, dataset_name.split(".")[0])

    return train_losses, val_losses


def test(net, test_loader, model_name, dataset_name, threshold=0.4):
    print("- testing")

    # load the best model
    load_model(net, model_name, dataset_name)

    accuracy = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            preds = net(inputs)

            for i in range(preds.shape[0]):
                for j in range(preds.shape[1]):
                    if preds[i][j] > threshold:
                        preds[i][j] = 1
                    elif preds[i][j] < threshold:
                        preds[i][j] = -1
                    else:
                        preds[i][j] = 0

            correct = (preds == labels).sum().item()
            total = labels.shape[0] * labels.shape[1]

            acc = correct / total
            accuracy.append(acc)

    print("\taccuracy: {:.4f}%".format(mean(accuracy) * 100))


###################################################
# Save & Load model
###################################################


def save_model(model, model_name, dataset_name):
    # set a path
    if not path.exists("./weights"):
        mkdir("./weights")
    weights_dir = path.join("./weights", dataset_name)
    if not path.exists(weights_dir):
        mkdir(weights_dir)
    weights_path = path.join(weights_dir, model_name)

    # save the weights
    torch.save(model.state_dict(), weights_path + ".pt")


def load_model(model, model_name, dataset_name):
    # set a path
    dataset_name = dataset_name.split(".")[0]
    weights_dir = path.join("./weights", dataset_name)
    weights_path = path.join(weights_dir, model_name)

    # load the weights
    model.load_state_dict(torch.load(weights_path + ".pt"))


###################################################
# Check model's information
###################################################


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Ref: https://discuss.dgl.ai/t/how-to-plot-the-attention-weights/206
def plot(
    g,
    attention,
    ax,
    nodes_to_plot=None,
    nodes_labels=None,
    edges_to_plot=None,
    nodes_pos=None,
    nodes_colors=None,
    edge_colormap=plt.cm.Reds,
):
    """
    Visualize edge attentions by coloring edges on the graph.
    g: nx.DiGraph
        Directed networkx graph
    attention: list
        Attention values corresponding to the order of sorted(g.edges())
    ax: matplotlib.axes._subplots.AxesSubplot
        ax to be used for plot
    nodes_to_plot: list
        List of node ids specifying which nodes to plot. Default to
        be None. If None, all nodes will be plot.
    nodes_labels: list, numpy.array
        nodes_labels[i] specifies the label of the ith node, which will
        decide the node color on the plot. Default to be None. If None,
        all nodes will have the same canonical label. The nodes_labels
        should contain labels for all nodes to be plot.
    edges_to_plot: list of 2-tuples (i, j)
        List of edges represented as (source, destination). Default to
        be None. If None, all edges will be plot.
    nodes_pos: dictionary mapping int to numpy.array of size 2
        Default to be None. Specifies the layout of nodes on the plot.
    nodes_colors: list
        Specifies node color for each node class. Its length should be
        bigger than number of node classes in nodes_labels.
    edge_colormap: plt.cm
        Specifies the colormap to be used for coloring edges.
    """
    if nodes_to_plot is None:
        nodes_to_plot = sorted(g.nodes())
    if edges_to_plot is None:
        assert isinstance(
            g, nx.DiGraph
        ), "Expected g to be an networkx.DiGraph" "object, got {}.".format(type(g))
        edges_to_plot = sorted(g.edges())
    nx.draw_networkx_edges(
        g,
        nodes_pos,
        edgelist=edges_to_plot,
        edge_color=attention,
        edge_cmap=edge_colormap,
        width=2,
        alpha=0.5,
        ax=ax,
        edge_vmin=0,
        edge_vmax=1,
    )

    if nodes_colors is None:
        nodes_colors = sns.color_palette("deep", max(nodes_labels) + 1)

    nx.draw_networkx_nodes(
        g,
        nodes_pos,
        nodelist=nodes_to_plot,
        ax=ax,
        node_size=30,
        node_color=[nodes_colors[nodes_labels[v - 1]] for v in nodes_to_plot],
        with_labels=False,
        alpha=0.9,
    )
