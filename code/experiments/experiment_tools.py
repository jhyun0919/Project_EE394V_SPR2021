import torch
import dgl
import numpy as np
from os import path, mkdir
from statistics import mean
import sys

sys.path.insert(0, "../data generation")

from data_preprocess_tools import *


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


def data_prep(dataset, batch_size):
    # normalize data
    x_normed = np.nan_to_num(dataset["x"] / dataset["x"].max(axis=0), nan=0.0)

    # split data
    train_X, val_X, test_X = split_data(x_normed, verbose=False)
    train_Y, val_Y, test_Y = split_data(dataset["y"], verbose=False)

    # set data feeders
    train_loader = get_dataloader(train_X, train_Y, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_X, val_Y, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_X, test_Y, batch_size=1, shuffle=True)

    return train_loader, val_loader, test_loader


def train(
    net,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    model_name,
    dataset_name,
    g=None,
    num_epochs=1,
    print_step=1000,
):

    print("> case: {}".format(dataset_name.split(".")[0]))
    print("> model: {}".format(model_name))
    print("- num of params: {}".format(count_parameters(net)))
    print("- training")

    train_losses = []
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
            if g is None:
                pred = net(inputs)
            else:
                pred = net(g, inputs)
            # calc the loss
            # print(pred.shape)
            # print(labels.shape)
            loss = loss_func(pred, labels)
            # back propa
            loss.backward()
            # update the wieghts
            optimizer.step()
            # update running loss
            running_loss += loss.item()
            # print statistics
            if i % print_step == (print_step - 1):
                print(
                    "\t[epoch: {:d}, iter: {:5d}]\tloss: {:.5f}".format(
                        epoch + 1, i + 1, running_loss / print_step
                    )
                )
                train_losses.append(running_loss / print_step)
                running_loss = 0.0

        val_loss = 0.0
        net.eval()
        for data in val_loader:
            inputs, labels = data
            # transfer Data to GPU if available
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # forward propa
            if g is None:
                pred = net(inputs)
            else:
                pred = net(g, inputs)
            # calc the loss
            loss = loss_func(pred, labels)
            # update val loss
            val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        # save the best model's weights
        if min_val_loss > val_loss:
            print(
                "\t- validation loss decreased ({:.5f}->{:.5f}): the best model was updated.".format(
                    min_val_loss, val_loss
                )
            )
            min_val_loss = val_loss
            save_model(net, model_name, dataset_name.split(".")[0])

    # return train_losses


def test(net, test_loader, model_name, dataset_name, g=None):
    print("- testing")
    # load the best model
    load_model(net, model_name, dataset_name.split(".")[0])

    # Ref: https://discuss.pytorch.org/t/how-to-calculate-accuracy-for-multi-label-classification/94906/2
    accuracy = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            if g is None:
                preds = net(inputs)
            else:
                preds = net(g, inputs)

            preds = np.round(preds)

            correct = (preds == labels).sum().item()
            total = labels.size(1)

            acc = correct / total
            accuracy.append(acc)

    print("\taccuracy: {:.4f}%".format(mean(accuracy) * 100))


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
    weights_dir = path.join("./weights", dataset_name)
    weights_path = path.join(weights_dir, model_name)

    # load the weights
    model.load_state_dict(torch.load(weights_path + ".pt"))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
