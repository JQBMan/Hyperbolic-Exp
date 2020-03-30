'''Hyperbolic Neural Network layer'''
import torch
import torch.nn as nn

from hyper_layers.HypAct import HypAct
from hyper_layers.HypLinear import HypLinear
class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias=True):
        super(HNNLayer, self).__init__()
        self.c = torch.nn.Parameter(torch.tensor(c))
        self.linear = HypLinear(manifold, in_features, out_features, self.c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, self.c, self.c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

