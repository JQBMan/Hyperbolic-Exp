''' All models: Baseline Model, HGCNConvModel and HGATConvModel'''
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

from hyper_layers import HNN
from hyper_layers.FermiDiracDecoder import FermiDiracDecoder
from hyper_layers.HGAT import HGAT
from hyper_layers.HGCN import HGCN
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
    def __init__(self, dim, hidden_1, hidden_2, u_nodes, i_nodes, heads):
        super(GATModel, self).__init__()
        self.dim = dim
        self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        self.conv1 = GATConv(self.hidden_1, self.hidden_2, heads=heads, concat=False)
        self.conv2 = GATConv(self.hidden_2, self.dim, heads=heads, concat=False)
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
        # self.c_in = c_in
        # self.c_out = c_out
        self.c_in = nn.Parameter(torch.Tensor([c_in]))
        self.c_out = nn.Parameter(torch.Tensor([c_out]))
        self.dropout = dropout
        # manifold
        self.manifold = PoincareBall()

        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        self.act = act

        self.conv1 = HGCN(self.manifold, self.hidden_1, self.hidden_2, self.act, c_in=self.c_in, c_out=self.c_out, dropout=dropout)
        self.conv2 = HGCN(self.manifold, self.hidden_2, self.dim, self.act, c_in=self.c_in, c_out=self.c_out, dropout=dropout)
        self.fc1 = FermiDiracDecoder(self.manifold, self.c_in, self.c_out, self.dim)

    def encoder(self, x):
        x_tan = self.manifold.proj_tan0(x, self.c_in)
        x_hyp = self.manifold.expmap0(x_tan, c=self.c_in)
        x_hyp = self.manifold.proj(x_hyp, c=self.c_in)
        return x_hyp

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        # i_embedding = self.item_embedding(graph.x)
        i_embedding = self.encoder(self.item_embedding(graph.x))

        i_embedding = self.conv1(i_embedding, graph.edge_index)
        i_embedding = self.conv2(i_embedding, graph.edge_index)
        i_embedding = self.manifold.proj_tan0(self.manifold.logmap0(i_embedding, c=self.c_in), c=self.c_in)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding))
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)

        # out = torch.sigmoid(out)
        # out = torch.softmax(out)
        return out

class HGATModel(nn.Module):
    def __init__(self, dim, hidden_1, hidden_2, u_nodes, i_nodes, c_in=1.0, c_out=1.0, act=torch.relu, dropout=0, heads=1):
        super(HGATModel, self).__init__()
        self.dim = dim
        self.heads = heads
        self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        # self.c_in = c_in
        # self.c_out = c_out
        self.c_in = nn.Parameter(torch.Tensor([c_in]))
        self.c_out = nn.Parameter(torch.Tensor([c_out]))

        self.dropout = dropout
        # manifold
        self.manifold = PoincareBall()
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        self.act = act
        self.conv1 = HGAT(self.manifold, self.hidden_1, self.hidden_2, self.act, c_in=self.c_in, c_out=self.c_out, dropout=dropout, heads=self.heads)
        self.conv2 = HGAT(self.manifold, self.hidden_2, self.dim, self.act, c_in=self.c_in, c_out=self.c_out, dropout=dropout, heads=self.heads)
        self.fc1 = FermiDiracDecoder(self.manifold, self.c_in, self.c_out, self.dim)

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        i_embedding = self.item_embedding(graph.x)
        i_embedding = self.conv1(i_embedding, graph.edge_index)
        i_embedding = self.conv2(i_embedding, graph.edge_index)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding))
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)

        # out = torch.sigmoid(out)
        # out = torch.softmax(out)
        return out

# # HNN Model
class HNNModel(nn.Module):
    def __init__(self, dim, hidden_1, hidden_2, u_nodes, i_nodes,  c=1.0, dropout=0., act=torch.relu):
        super(HNNModel, self).__init__()
        self.dim = dim
        self.hidden_1, self.hidden_2 = hidden_1, hidden_2
        self.i_nodes, self.u_nodes = i_nodes, u_nodes
        # self.c = c
        self.c = nn.Parameter(torch.Tensor([c]))

        self.dropout = dropout
        self.act = act
        self.manifold = PoincareBall()

        # self.c_out = c_out
        self.user_embedding = nn.Embedding(self.u_nodes, self.dim)
        self.item_embedding = nn.Embedding(self.i_nodes, self.hidden_1)
        self.conv1 = HNN(self.manifold, self.hidden_1, self.hidden_2, c=self.c, dropout=self.dropout, act=self.act, use_bias=True)
        self.conv2 = HNN(self.manifold, self.hidden_2, self.dim, c=self.c, dropout=self.dropout, act=self.act, use_bias=True)
        # self.fc1 = nn.Linear(self.dim+self.dim, 1)
        self.fc1 = FermiDiracDecoder(self.manifold, self.c_in, self.c_out, self.dim * 2)
        # self.fc2 = nn.Linear(8, 1)

    def encoder(self, x):
        return self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)

    def forward(self, u, i, graph):
        u_embedding = self.user_embedding(u)
        # i_embedding = self.item_embedding(graph.x)
        i_embedding = self.encoder(self.item_embedding(graph.x))
        i_embedding = self.conv1(i_embedding)
        i_embedding = self.conv2(i_embedding)
        i_embedding = self.manifold.proj_tan0(self.manifold.logmap0(i_embedding, c=self.c), c=self.c)
        i_embedding = torch.squeeze(torch.matmul(i, i_embedding))
        u_i = torch.cat((u_embedding, i_embedding), 1)
        out = self.fc1(u_i)
        # out = self.fc2(out)
        # out = torch.sigmoid(out)
        return out














