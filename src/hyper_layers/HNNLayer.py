'''Hyperbolic Neural Network layer'''
import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F

# from hyper_layers.HypAct import HypAct
# from hyper_layers.HypLinear import HypLinear
class HNNLayer(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, dropout=0, act=F.relu, use_bias=False):
        super(HNNLayer, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_channels = in_features
        self.out_channels = out_features
        self.dropout = dropout
        self.act = act
        self.use_bias = use_bias

        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_channels, self.in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化weight、bias
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        mv = self.manifold.mobius_matvec(self.weight, x, self.c)
        x = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias, self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            x = self.manifold.mobius_add(x, hyp_bias, c=self.c)
            x = self.manifold.proj(x, self.c)
        xt = self.act(self.manifold.logmap0(x, c=self.c))
        xt = self.manifold.proj_tan0(xt, c=self.c)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c), c=self.c)


# class HNNLayer(nn.Module):
#     """
#     Hyperbolic neural networks layer.
#     """
#
#     def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias=True):
#         super(HNNLayer, self).__init__()
#         self.c = c
#         self.linear = HypLinear(manifold, in_features, out_features, self.c, dropout, use_bias)
#         self.hyp_act = HypAct(manifold, self.c, self.c, act)
#
#     def forward(self, x):
#         h = self.linear.forward(x)
#         h = self.hyp_act.forward(h)
#         return h
#
