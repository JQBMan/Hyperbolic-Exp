'''Hyperbolic Graph Convolution Neural Network, geometric '''
import torch
import torch.nn.init as init
from torch_scatter import scatter_add
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from hyper_layers import HypLinear, HypAct

class HGCNConv_geometric(MessagePassing):
    def __init__(self, manifold, in_channels, out_channels, act, improved=True, cached=False,
                 bias=True, normalize=True, c_in=1.0, c_out=1.0, dropout=0, **kwargs):
        super(HGCNConv_geometric, self).__init__(aggr='add', **kwargs)

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.act = act
        self.c_in = torch.nn.Parameter(torch.tensor(c_in))
        self.c_out = torch.nn.Parameter(torch.tensor(c_out))
        self.dropout = dropout

        #self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.linear = HypLinear(manifold, in_channels, out_channels, c_in, dropout, bias)
        #self.agg = HypGCNAgg(manifold, c_in, out_channels, dropout)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

        self.reset_parameters()

    def reset_parameters(self):
        #init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        # x = self.propagate(edge_index, size=size, x=x)
        # out = self.hyp_act.forward(x)
        # return out
        #x = torch.matmul(x, self.weight)  ##############

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                'Cached {} number of edges, but found {}. Please '
                'disable the caching behavior of this layer by removing '
                'the `cached=True` argument in its constructor.'.format(
                    self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                         edge_weight, self.improved,
                                         x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        x = self.linear.forward(x)

        x = self.manifold.logmap0(x, c=self.c_in)
        x = self.propagate(edge_index, x=x, norm=norm)
        x = self.manifold.expmap0(x, c=self.c_in)
        x = self.manifold.proj(x, c=self.c_in)

        out = self.hyp_act.forward(x)
        return out

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j if norm is not None else x_j


    def update(self, aggr_out):
        if self.bias is not None:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c_in)
            hyp_bias = self.manifold.expmap0(bias, self.c_in)
            hyp_bias = self.manifold.proj(hyp_bias, self.c_in)
            aggr_out = self.manifold.mobius_add(aggr_out, hyp_bias, c=self.c_in)
            aggr_out = self.manifold.proj(aggr_out, self.c_in)
        return aggr_out

def __repr__(self):
    return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                               self.out_channels)

