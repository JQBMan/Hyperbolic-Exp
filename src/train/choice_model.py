########################
# choose model
########################
import logging

import torch

from train.model import GCNModel, GATModel, MLPModel, EmbeddingModel
from train.model import HGCNModel, HGATModel, HNNModel
from optimizer.RiemannianAdam import RiemannianAdam



def choice_model(args, u_nodes, i_nodes, device):


    # choice the model
    # GNN
    if args.model == 'gcn':
        model = GCNModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)

    elif args.model == 'gat':
        model = GATModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes, args.heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)

    elif args.model == 'kgcn':
        return None

    elif args.model == 'kgcn-ls':
        return None

    # HGNN
    elif args.model == 'hgcn':
        model = HGCNModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes,
                          c_in=args.c_in, c_out=args.c_out, dropout=args.dropout, device=device).to(device)
        optimizer = RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)

    elif args.model == 'hgat':
        model = HGATModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes,
                          c_in=args.c_in, c_out=args.c_out, dropout=args.dropout, heads=args.heads, device=device).to(device)
        optimizer = RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)

    # NN
    elif args.model == 'hnn':
        model = HNNModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes, c=args.c_in, dropout=args.dropout, device=device).to(device)
        optimizer = RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)

    elif args.model == 'mlp':
        model = MLPModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)

    elif args.model == 'ripplenet':
        return None

    elif args.model == 'cke':
        return None

    elif args.model == 'shine':
        return None

    elif args.model == 'embedding':
        model = EmbeddingModel(args.dim, u_nodes, i_nodes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)

    else:
        logging.info('No such model.')
        return None, None

    return model, optimizer