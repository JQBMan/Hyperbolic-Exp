import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

from hyper_layers.FermiDiracDecoder import FermiDiracDecoder
from hyper_layers import HNNLayer as HNN
from hyper_layers.HGATConv_geometric import HGATConv_gemoetric as HGAT
from hyper_layers.HGCNConv_gemoetric import HGCNConv_geometric as HGCN
from manifolds.poincare import PoincareBall
from optimizer.RiemannianAdam import RiemannianAdam

class Model(nn.Module):
    # def __init__(self, u_hidden_size, i_hidden_size, number, i_hidden_list, hidden_list, args,
#                  heads=6, dataset='book', mode='GAT'):

#   u_hidden_size >> dimension(dim)   i_hidden_size >> hidden1    i_hidden_list >> [hidden1, hidden2, dim]
    def __init__(self, number, args):
        super(Model, self).__init__()
        self.u_hidden_size, self.i_hidden_size = args.dim, args.hidden1
        self.i_hidden_list = [args.hidden1, args.hidden2, args.dim]  #i_hidden_size, ..., u_hidden_size
        self.hidden_list = [1]
        self.u_nodes, self.i_nodes = number[args.dataset]['users'], number[args.dataset]['entities']
        self.u_embedding = nn.Embedding(self.u_nodes, self.i_hidden_list[-1]) #Embedding(u_nodes, dim)
        self.i_embedding = nn.Embedding(self.i_nodes, args.hidden1)    #Embedding(i_nodes, hidden1)  i_hidden_size
        self.convs = nn.ModuleList()
        self.args = args
        self.c_in = nn.Parameter(torch.Tensor([args.c_in]))
        self.c_out = nn.Parameter(torch.Tensor([args.c_out]))
        self.model = args.model.upper()

        if self.model == 'GCN':
            self.convs = nn.ModuleList([GCNConv(self.i_hidden_list[i - 1], self.i_hidden_list[i])
                                        for i in range(1, len(self.i_hidden_list))])
            self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr,
                                              weight_decay=args.l2_weight_decay)
        elif self.model == 'GAT':
            self.convs = nn.ModuleList([GATConv(self.i_hidden_list[i - 1], self.i_hidden_list[i], heads=args.heads, concat=False)
                                        for i in range(1, len(self.i_hidden_list))])
            self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr,
                                              weight_decay=args.l2_weight_decay)
        elif self.model == 'HGCN':
            self.manifold = PoincareBall()
            self.convs = nn.ModuleList([HGCN(self.manifold, self.i_hidden_list[i - 1], self.i_hidden_list[i],
                                             act=torch.relu, c_in=self.c_in, c_out=self.c_out)
                                        for i in range(1, len(self.i_hidden_list))])
            self.optimizer = RiemannianAdam(self.parameters(), lr=args.lr,
                                            weight_decay=args.l2_weight_decay)
        elif self.model == 'HGAT':
            self.manifold = PoincareBall()
            self.convs = nn.ModuleList([HGAT(self.manifold, self.i_hidden_list[i - 1], self.i_hidden_list[i],
                                             act=torch.relu, c_in=self.c_in, c_out=self.c_out)
                                        for i in range(1, len(self.i_hidden_list))])
            self.optimizer = RiemannianAdam(self.parameters(), lr=args.lr,
                                            weight_decay=args.l2_weight_decay)

        elif self.model == 'HNN':
            self.manifold = PoincareBall()
            self.convs = nn.ModuleList([HNN(self.manifold, self.i_hidden_list[i - 1], self.i_hidden_list[i], c=self.c_in)
                                        for i in range(1, len(self.i_hidden_list))])
            self.optimizer = RiemannianAdam(self.parameters(), lr=args.lr,
                                            weight_decay=args.l2_weight_decay)

        elif self.model == 'EMBEDDING':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr,
                                              weight_decay=args.l2_weight_decay)

        elif self.model == 'MLP':
            pass




        self.hidden_list = [self.i_hidden_list[-1] + self.u_hidden_size] + self.hidden_list
        
        if self.model not in ['HGAT','HGCN'] or args.cat == 0:  # 内积 or 级联
            self.liners = nn.ModuleList([nn.Linear(self.hidden_list[i - 1], self.hidden_list[i])
                                         for i in range(1, len(self.hidden_list))])
        else:
            self.liners = FermiDiracDecoder(self.manifold, self.c_in, self.c_out, self.i_hidden_list[-1] * 2)
        # self.optimizer = RiemannianSGD(self.parameters(),lr=args.learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)
        if self.hidden_list[-1] == 1:
            self.final = torch.sigmoid
            self.loss = nn.BCELoss()
        else:
            self.final = torch.softmax
            self.loss = nn.NLLLoss()

    def encoder(self, x):
        if self.model == 'HNN':
            return self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.args.c_in),
                                                            c=self.args.c_in), c=self.args.c_in)
        elif self.model == 'HGCN':
            x_tan = self.manifold.proj_tan0(x, self.args.c_in)
            x_hyp = self.manifold.expmap0(x_tan, c=self.args.c_in)
            x_hyp = self.manifold.proj(x_hyp, c=self.args.c_in)
            return x_hyp
        else:
            return x

    def forward(self, user, item, graph):
        u_emb = self.u_embedding(user)
        i_emb = self.encoder(self.i_embedding(graph.x))

        for layer in self.convs:
            if self.model == 'HNN':
                i_emb = layer(i_emb)
            else:
                i_emb = layer(i_emb, graph.edge_index)
        if self.model in ['HNN','HGCN']:
            i_emb = self.manifold.proj_tan0(self.manifold.logmap0(i_emb, c=self.args.c_in), c=self.args.c_in)

        i_emb = torch.squeeze(torch.matmul(item, i_emb))

        out = torch.cat((u_emb, i_emb), 1)

        # for layer in self.liners:
        #     out = layer(out)

        if self.model not in ['HGAT','HGCN']:
            for layer in self.liners:
                out = layer(out)
            return self.final(out)
        else:
            return self.liners(out)