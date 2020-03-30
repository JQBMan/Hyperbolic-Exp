'''Hyperbolic Aggregation layer'''
import torch
import torch.nn as nn

class HypAgg(nn.Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
