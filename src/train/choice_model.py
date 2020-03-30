########################
# choose model
########################
import logging

import torch

from train.model import GCNModel, GATModel, MLPModel, EmbeddingModel
from train.model import HGCNModel, HGATModel, HNNModel


def choice_model(args, u_nodes, i_nodes, device):
    # choice the model
    # GNN
    if args.model == 'gcn':
        model = GCNModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes).to(device)
    elif args.model == 'gat':
        model = GATModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes).to(device)
    elif args.model == 'kgcn':
        return None
    elif args.model == 'kgcn-ls':
        return None
    # HGNN
    elif args.model == 'hgcn':
        torch.autograd.set_detect_anomaly(True)
        model = HGCNModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes,
                          c_in=args.c_in, c_out=args.c_out, dropout=args.dropout).to(device)
    elif args.model == 'hgat':
        torch.autograd.set_detect_anomaly(True)
        model = HGATModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes,
                          c_in=args.c_in, c_out=args.c_out, dropout=args.dropout).to(device)
    # NN
    elif args.model == 'hnn':
        torch.autograd.set_detect_anomaly(True)
        model = HNNModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes, c=args.c_in, dropout=args.dropout).to(device)
    elif args.model == 'mlp':
        model = MLPModel(args.dim, args.hidden1, args.hidden2, u_nodes, i_nodes).to(device)

    elif args.model == 'ripplenet':
        return None
    elif args.model == 'cke':
        return None
    elif args.model == 'shine':
        return None
    elif args.model == 'embedding':
        model = EmbeddingModel(args.dim, u_nodes, i_nodes).to(device)
    else:
        logging.info('No such model.')
        return None

    return model