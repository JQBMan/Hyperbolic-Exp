# -*- coding:utf-8 -*-
# @Time: 2020/3/15 9:40
# @Author: jockwang, jockmail@126.com
import torch
import torch.nn as nn

from hyper_layers import HypLinear, HypAct


class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, manifold, c_in, c_out, input_dim, dropout=0.0, bias=True, act=torch.sigmoid, r=2., t=1.):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.bias = bias
        self.dropout = dropout
        self.linear = HypLinear(self.manifold, input_dim*2, 1, self.c_in, self.dropout, self.bias)
        self.act = HypAct(self.manifold, self.c_in, self.c_out, act)

    def forward(self, dist):
        dist = self.manifold.proj_tan0(dist, self.c_in)
        dist = self.manifold.expmap0(dist, c=self.c_in)
        dist = self.manifold.proj(dist, c=self.c_in)
        dist = self.linear(dist)
        dist = self.act(dist)
        return dist