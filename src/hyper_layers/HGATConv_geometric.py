'''Hyperbolic Graph Attentional Network'''
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from hyper_layers import HypLinear, HypAct

class HGATConv_gemoetric(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    #def __init__(self, in_channels, out_channels, heads=1, concat=True,
    #             negative_slope=0.2, dropout=0, bias=True, **kwargs):
    def __init__(self, manifold, in_channels, out_channels, act, heads=1, concat=True,
                 negative_slope=0.2, c_in=1.0, c_out=1.0, dropout=0, bias=True, **kwargs):
        super(HGATConv_gemoetric, self).__init__(aggr='add', **kwargs)

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.act = act
        self.negative_slope = negative_slope
        self.c_in = torch.nn.Parameter(torch.tensor(c_in))
        self.c_out = torch.nn.Parameter(torch.tensor(c_out))
        self.dropout = dropout

        #self.weight = Parameter(
        #    torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.linear = HypLinear(manifold, in_channels, out_channels, c_in, dropout, bias)
        #self.agg = HypAgg(manifold, c_in, out_channels, dropout)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

        self.reset_parameters()

    def reset_parameters(self):
        #init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.att)
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # ##########################
        # # in orginal GAT
        # # only multiple X*W
        # if torch.is_tensor(x):
        #     x = torch.matmul(x, self.weight)
        # else:
        #     x = (None if x[0] is None else torch.matmul(x[0], self.weight),
        #          None if x[1] is None else torch.matmul(x[1], self.weight))
        # ##########################
        x = self.linear.forward(x)
        x = self.propagate(edge_index, size=size, x=x)
        out = self.hyp_act.forward(x)
        return out


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_j = self.manifold.logmap0(x_j, c=self.c_in) # map to tan space
        #x_tangent = self.manifold.logmap0(x, c=self.c)
        #support_t = torch.spmm(adj, x_tangent)
        #output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            x_i = self.manifold.logmap0(x_i, c=self.c_in)
            #alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = torch.cat([x_i, x_j], dim=-1)
            alpha = alpha * self.att
            alpha = alpha.sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = x_j * alpha.view(-1, self.heads, 1)
        out = self.manifold.expmap0(out, c=self.c_in)
        out = self.manifold.proj(out, c=self.c_in)
        return out

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c_in)
            hyp_bias = self.manifold.expmap0(bias, self.c_in)
            hyp_bias = self.manifold.proj(hyp_bias, self.c_in)
            aggr_out = self.manifold.mobius_add(aggr_out, hyp_bias, c=self.c_in)
            aggr_out = self.manifold.proj(aggr_out, self.c_in)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
