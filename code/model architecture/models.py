import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import GraphConv, GATConv
import dgl.function as fn

##################################################
# Dense Net
##################################################


class DenseNet(nn.Module):
    def __init__(self, input_dim, output_dim, node_num):
        super(DenseNet, self).__init__()
        self.input_dim = input_dim
        self.node_num = node_num
        self.output_dim = output_dim
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(input_dim * node_num, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, self.node_num * self.output_dim)
        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.flatten(x)
        x = torch.reshape(x, (-1, 1, self.input_dim * self.node_num))
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        x = torch.reshape(x, (self.node_num, self.output_dim))
        # x = self.sigmoid(x)
        x = self.tanh(x)
        return x


##################################################
# Convolution Neural Net
##################################################


class ConvNet(nn.Module):
    def __init__(self, output_dim):
        super(ConvNet, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv1d_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv1d_3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.pooling = nn.MaxPool1d(kernel_size=4, stride=2)
        self.dense_1 = nn.Linear(576, 512)
        self.dense_2 = nn.Linear(512, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1st conv layer
        x = F.relu(self.conv1d_1(x.unsqueeze_(1)))
        x = self.pooling(x)
        # 2nd conv layer
        x = F.relu(self.conv1d_2(x))
        x = self.pooling(x)
        # 3rd conv layer
        x = F.relu(self.conv1d_3(x))
        # reshape
        x = x.view(x.size(0), -1)
        # dense layer
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)
        x = self.sigmoid(x)
        return x


##################################################
# Graph Conv Net
##################################################

# Ref:  https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#graphconv
# Ref: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn


class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        h = self.tanh(h)
        # h = self.sigmoid(h)
        return h


##################################################
# Graph Attention Net
##################################################

# Ref:
# https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#gatconv
# https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/9_gat.html?highlight=gat
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/gat.py


class GAT(nn.Module):
    def __init__(
        self,
        g,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            GATConv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        h = self.gat_layers[-1](self.g, h).mean(1)
        # h = self.sigmoid(h)
        h = self.tanh(h)
        return h
