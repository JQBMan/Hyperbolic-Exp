class Model(nn.Module):
    # def __init__(self, u_hidden_size, i_hidden_size, number, i_hidden_list, hidden_list, args,
#                  heads=6, dataset='book', mode='GAT'):

#   u_hidden_size >> dimension(dim)   i_hidden_size >> hidden1    i_hidden_list >> [hidden1, hidden2, dim]
    def __init__(self, u_hidden_size, i_hidden_size, number, i_hidden_list, hidden_list, args):
        super(Model, self).__init__()
        self.u_hidden_size, self.i_hidden_size = u_hidden_size, i_hidden_size
        self.u_nodes, self.i_nodes = number['users'], number['entities']
        self.u_embedding = nn.Embedding(self.u_nodes, i_hidden_list[-1]) #Embedding(u_nodes, dim)
        self.i_embedding = nn.Embedding(self.i_nodes, i_hidden_size)    #Embedding(i_nodes, hidden1)
        self.convs = nn.ModuleList()
        self.args = args
        self.c_in = nn.Parameter(torch.Tensor([args.c_in]))
        self.c_out = nn.Parameter(torch.Tensor([args.c_out]))
        self.mode = mode
        # [] = [hidden1] + i_hidden_list  >>> [hidden1, hidden2, dim]
        i_hidden_list = [i_hidden_size] + i_hidden_list
        if mode == 'GCN':
            self.convs = nn.ModuleList([GCNConv(i_hidden_list[i - 1], i_hidden_list[i])
                                        for i in range(1, len(i_hidden_list))])
            self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr,
                                              weight_decay=args.weight_decay)
        elif mode == 'GAT':
            self.convs = nn.ModuleList([GATConv(i_hidden_list[i - 1], i_hidden_list[i], heads=heads, concat=False)
                                        for i in range(1, len(i_hidden_list))])
            self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate,
                                              weight_decay=args.weight_decay)
        elif mode == 'HGCN':
            self.manifold = PoincareBall()
            self.convs = nn.ModuleList([HGCN(self.manifold, i_hidden_list[i - 1], i_hidden_list[i],
                                             act=torch.relu, c_in=self.c_in, c_out=self.c_out)
                                        for i in range(1, len(i_hidden_list))])
            self.optimizer = RiemannianAdam(self.parameters(), lr=args.learning_rate,
                                            weight_decay=args.weight_decay)
        elif mode == 'HGAT':
            self.manifold = PoincareBall()
            self.convs = nn.ModuleList([HGAT(self.manifold, i_hidden_list[i - 1], i_hidden_list[i],
                                             act=torch.relu, c_in=self.c_in, c_out=self.c_out)
                                        for i in range(1, len(i_hidden_list))])
            self.optimizer = RiemannianAdam(self.parameters(), lr=args.learning_rate,
                                            weight_decay=args.weight_decay)

        elif mode == 'HNN':
            self.manifold = PoincareBall()
            self.convs = nn.ModuleList([HNN(self.manifold, i_hidden_list[i - 1], i_hidden_list[i], c=self.c_in)
                                        for i in range(1,len(i_hidden_list))])
            self.optimizer = RiemannianAdam(self.parameters(), lr=args.learning_rate,
                                            weight_decay=args.weight_decay)
        hidden_list = [i_hidden_list[-1] + u_hidden_size] + hidden_list
        
        if mode not in ['HGAT','HGCN'] or args.cat == 0:
            self.liners = nn.ModuleList([nn.Linear(hidden_list[i - 1], hidden_list[i])
                                         for i in range(1, len(hidden_list))])
        else:
            self.liners = FermiDiracDecoder(self.manifold, self.c_in, self.c_out, i_hidden_list[-1]*2)
        # self.optimizer = RiemannianSGD(self.parameters(),lr=args.learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if hidden_list[-1] == 1:
            self.final = torch.sigmoid
            self.loss = nn.BCELoss()
        else:
            self.final = torch.softmax
            self.loss = nn.NLLLoss()

    def encoder(self, x):
        if self.mode == 'HNN':
            return self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.args.c_in),
                                                            c=self.args.c_in), c=self.args.c_in)
        elif self.mode == 'HGCN':
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
            if self.mode == 'HNN':
                i_emb = layer(i_emb)
            else:
                i_emb = layer(i_emb, graph.edge_index)
        if self.mode in ['HNN','HGCN']:
            i_emb = self.manifold.proj_tan0(self.manifold.logmap0(i_emb, c=self.args.c_in), c=self.args.c_in)

        i_emb = torch.squeeze(torch.matmul(item, i_emb))

        out = torch.cat((u_emb, i_emb), 1)

        # for layer in self.liners:
        #     out = layer(out)

        if self.mode not in ['HGAT','HGCN']:
            for layer in self.liners:
                out = layer(out)
            return self.final(out)
        else:
            return self.liners(out)