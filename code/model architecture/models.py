import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.nn import GraphConv, GATConv

##################################################
# Dense Net
##################################################


class DenseNet(nn.Module):
    def __init__(self, input_dim, output_dim, node_num):
        super(DenseNet, self).__init__()
        self.node_num = node_num
        self.output_dim = output_dim
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(node_num, 512)
        self.dense_2 = nn.Linear(512, 512)
        self.dense_3 = nn.Linear(512, self.node_num * self.output_dim)
        self.tanh = nn.Tanh()  # tanh

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        x = torch.reshape(x, (-1, self.node_num, self.output_dim))
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


class GraphConvNet(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphConvNet, self).__init__()
        self.gcn_1 = GraphConv(in_feats, 100)
        self.gcn_2 = GraphConv(100, 200)
        self.gcn_3 = GraphConv(200, out_feats)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()  # tanh

    def forward(self, g, x):
        x = self.gcn_1(g, x)
        x = self.elu(x)
        print(x.size())
        x = self.gcn_2(g, x)
        x = self.elu(x)
        print(x.size())
        x = self.gcn_3(g, x)
        x = self.sigmoid(x)
        print(x.size())
        return x


##################################################
# Graph Attention Net
##################################################

# Ref:
# https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#gatconv
# https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/9_gat.html?highlight=gat


class GraphAttentionNet(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super(GraphAttentionNet, self).__init__()
        self.gat_1 = GATConv(in_feats, 100, num_heads)
        self.gat_2 = GATConv(100 * num_heads, 200, num_heads)
        self.gat_3 = GATConv(200 * num_heads, out_feats, 1)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()  # tanh

    def forward(self, g, x):
        x = self.gat_1(g, x)
        x = self.elu(x)
        print(x.size())
        x = self.gat_2(g, x)
        x = self.elu(x)
        print(x.size())
        x = self.gat_3(g, x)
        x = self.sigmoid(x)
        print(x.size())
        return x
