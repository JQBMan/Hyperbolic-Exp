''' All models: Baseline Model, HGCNConvModel and HGATConvModel'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

from hyper_layers import HNNLayer
from hyper_layers.FermiDiracDecoder import FermiDiracDecoder
from hyper_layers.HGATConv_geometric import HGATConv_gemoetric
from hyper_layers.HGCNConv_gemoetric import HGCNConv_geometric
from manifolds.poincare import *

# GCNModel
class GCNModel(nn.Module):
    def __init__(self, dim, hidden_1, hidden_2, u_nodes, i_nodes):
        super(GCNModel, self).__init__()
        self.dim = dim
        self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        self.conv1 = GCNConv(self.hidden_1, self.hidden_2)
        self.conv2 = GCNConv(self.hidden_2, self.dim)
        self.fc1 = nn.Linear(self.dim + self.dim, 1)
        # self.fc2 = nn.Linear(8, 1)

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        i_embedding = self.item_embedding(graph.x)
        i_embedding = self.conv1(i_embedding, graph.edge_index)
        i_embedding = self.conv2(i_embedding, graph.edge_index)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding))
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)
        # out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

# MLPModel
class MLPModel(nn.Module):
    def __init__(self, dim, hidden_1, hidden_2, u_nodes, i_nodes):
        super(MLPModel, self).__init__()
        self.dim = dim
        self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        #self.conv1 = GCNConv(self.hidden_1, self.hidden_2)
        #self.conv2 = GCNConv(self.hidden_2, self.dim)
        self.fc_item = nn.Linear(self.hidden_1, self.dim)
        self.fc1 = nn.Linear(self.dim+self.dim, 1)
        # self.fc2 = nn.Linear(8, 1)

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        i_embedding = self.item_embedding(graph.x)
        i_embedding = self.fc_item(i_embedding)
        #i_embedding = self.conv2(i_embedding, graph.edge_index)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding)) ### remove if error
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)
        # out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

# GATModel
class GATModel(nn.Module):
    def __init__(self, dim, hidden_1, hidden_2, u_nodes, i_nodes):
        super(GATModel, self).__init__()
        self.dim = dim
        self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        self.conv1 = GATConv(self.hidden_1, self.hidden_2)
        self.conv2 = GATConv(self.hidden_2, self.dim)
        self.fc1 = nn.Linear(self.dim+self.dim, 1)
        # self.fc2 = nn.Linear(8, 1)

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        i_embedding = self.item_embedding(graph.x)
        i_embedding = self.conv1(i_embedding, graph.edge_index)
        i_embedding = self.conv2(i_embedding, graph.edge_index)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding))
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)
        # out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

# EmbeddingModel
class EmbeddingModel(nn.Module):
    def __init__(self, dim, u_nodes, i_nodes):
        super(EmbeddingModel, self).__init__()
        self.dim = dim
        #self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.dim)
        #self.conv1 = GCNConv(self.hidden_1, self.hidden_2)
        #self.conv2 = GCNConv(self.hidden_2, self.dim)
        self.fc1 = nn.Linear(self.dim+self.dim, 1)
        # self.fc2 = nn.Linear(8, 1)

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        i_embedding = self.item_embedding(graph.x)
        #i_embedding = self.conv1(i_embedding, graph.edge_index)
        #i_embedding = self.conv2(i_embedding, graph.edge_index)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding)) ### remove if error
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)
        # out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

# HGCNModel
class HGCNModel(nn.Module):
    def __init__(self, dim, hidden_1, hidden_2, u_nodes, i_nodes, c_in=1.0, c_out=1.0, act=torch.relu, dropout=0):
        super(HGCNModel, self).__init__()
        self.dim = dim
        self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout
        # manifold
        self.manifold = PoincareBall()
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        self.act = act

        self.conv1 = HGCNConv_geometric(self.manifold, self.hidden_1, self.hidden_2, self.act, c_in=self.c_in, c_out=self.c_out, dropout=dropout)
        self.conv2 = HGCNConv_geometric(self.manifold, self.hidden_2, self.dim, self.act, c_in=self.c_in, c_out=self.c_out, dropout=dropout)
        self.fc1 = FermiDiracDecoder(self.manifold, self.c_in, self.c_out, self.dim)

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        i_embedding = self.item_embedding(graph.x)
        i_embedding = self.conv1(i_embedding, graph.edge_index)
        i_embedding = self.conv2(i_embedding, graph.edge_index)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding))
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)

        return out

class HGATModel(nn.Module):
    def __init__(self, dim, hidden_1, hidden_2, u_nodes, i_nodes, c_in=1.0, c_out=1.0, act=torch.relu, dropout=0):
        super(HGATModel, self).__init__()
        self.dim = dim
        self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout
        # manifold
        self.manifold = PoincareBall()
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        self.act = act
        self.conv1 = HGATConv_gemoetric(self.manifold, self.hidden_1, self.hidden_2, self.act, c_in=self.c_in, c_out=self.c_out, dropout=dropout)
        self.conv2 = HGATConv_gemoetric(self.manifold, self.hidden_2, self.dim, self.act, c_in=self.c_in, c_out=self.c_out, dropout=dropout)
        self.fc1 = FermiDiracDecoder(self.manifold, self.c_in, self.c_out, self.dim)

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        i_embedding = self.item_embedding(graph.x)
        i_embedding = self.conv1(i_embedding, graph.edge_index)
        i_embedding = self.conv2(i_embedding, graph.edge_index)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding))
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)
        return out

# # HNN Model
class HNNModel(nn.Module):
    def __init__(self, dim, hidden_1, hidden_2, u_nodes, i_nodes,  c=1.0, dropout=0., act=torch.relu):
        super(HNNModel, self).__init__()
        self.dim = dim
        self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        self.c = c
        self.dropout = dropout
        self.act = act
        self.manifold = PoincareBall()

        # self.c_out = c_out
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        self.conv1 = HNNLayer(self.manifold, self.hidden_1, self.hidden_2, c=self.c, dropout=self.dropout, act=self.act, use_bias=True)
        self.conv2 = HNNLayer(self.manifold, self.hidden_2, self.dim, c=self.c, dropout=self.dropout, act=self.act, use_bias=True)
        self.fc1 = nn.Linear(self.dim+self.dim, 1)
        # self.fc2 = nn.Linear(8, 1)

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        i_embedding = self.item_embedding(graph.x)
        i_embedding = self.conv1(i_embedding)
        i_embedding = self.conv2(i_embedding)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding))
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)
        # out = self.fc2(out)
        out = torch.sigmoid(out)
        return out














