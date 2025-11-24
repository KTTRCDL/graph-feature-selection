import torch
from torch import nn
from dgl import ops
from dgl.nn.functional import edge_softmax
from dgl import function as fn
from dgl.nn.pytorch.conv import APPNPConv
import math
import torch.nn.functional as F

class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, x):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res)
        x = x + x_res
        return x

class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x

class FAGCNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.gate = nn.Linear(2 * dim * hidden_dim_multiplier, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
    
    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}
    
    def forward(self, graph, x):
        graph = graph.local_var()
        graph.ndata['h'] = x
        deg = graph.in_degrees().cuda().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        graph.ndata['d'] = norm
        graph.apply_edges(self.edge_applying)
        graph.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))
        return graph.ndata['z']

# refer https://github.com/dmlc/dgl/tree/master/examples/pytorch
class APPNPModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, graph=None, k=2, alpha=0.1, edge_drop=0.5, **kwargs):
        super().__init__()
        self.graph = graph
        self.layer = nn.Linear(dim * hidden_dim_multiplier, dim * hidden_dim_multiplier)
        self.propate = APPNPConv(k, alpha, edge_drop)

    def forward(self, graph, x):
        h = self.layer(x)
        h = self.propate(graph, h)
        return h

# refer https://github1s.com/dmlc/dgl/blob/master/python/dgl/nn/mxnet/conv/sgconv.py
class SGCModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, k=2, cached=True, bias=False, **kwargs) -> None:
        super().__init__()
        # self.sgcconv = SGConv(dim * hidden_dim_multiplier, dim * hidden_dim_multiplier, k, cached, bias)
    
    def forward(self, graph, x):
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        h = ops.u_mul_e_sum(graph, x, norm_coefs)
        # graph = graph.local_var()
        # h = self.sgcconv(graph, x)
        return h

# refer https://github.com/RecklessRonan/GloGNN/blob/master/large-scale/models.py
class ACMGCNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, **kwargs):
        super().__init__()
        self.dim = dim * hidden_dim_multiplier
        self.weight_low, self.weight_high, self.weight_mlp = nn.Parameter(torch.FloatTensor(self.dim, self.dim)), nn.Parameter(
                                torch.FloatTensor(self.dim, self.dim)), nn.Parameter(torch.FloatTensor(self.dim, self.dim))
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = nn.Parameter(torch.FloatTensor(self.dim, 1)), nn.Parameter(
                                torch.FloatTensor(self.dim, 1)), nn.Parameter(torch.FloatTensor(self.dim, 1))
        self.att_vec = nn.Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1. / math.sqrt(self.att_vec.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_low, output_high, output_mlp):
        T = 3
        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat([torch.mm((output_low), self.att_vec_low), torch.mm(
            (output_high), self.att_vec_high), torch.mm((output_mlp), self.att_vec_mlp)], 1)), self.att_vec)/T, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, graph, x):
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        output_low = ops.u_mul_e_sum(graph, x, norm_coefs) # output_low
        output_high = x - output_low # output_high

        output_low = F.relu(torch.mm(output_low, self.weight_low))
        output_high = F.relu(torch.mm(output_high, self.weight_high))
        output_mlp = F.relu(torch.mm(x, self.weight_mlp))

        self.att_low, self.att_high, self.att_mlp = self.attention(output_low, output_high, output_mlp)
        return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)

class GCNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5
        x = ops.u_mul_e_sum(graph, x, norm_coefs)
        x = self.feed_forward_module(graph, x)
        return x


class SAGEModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        message = ops.copy_u_mean(graph, x)
        x = torch.cat([x, message], axis=1)
        x = self.feed_forward_module(graph, x)
        return x


def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')


class GATModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.input_linear = nn.Linear(in_features=dim, out_features=dim)
        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.input_linear(x)
        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)
        x = x.reshape(-1, self.head_dim, self.num_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)
        x = self.feed_forward_module(graph, x)
        return x


class GATSepModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.input_linear = nn.Linear(in_features=dim, out_features=dim)
        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.input_linear(x)
        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)
        x = x.reshape(-1, self.head_dim, self.num_heads)
        message = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)
        x = self.feed_forward_module(graph, x)
        return x


class TransformerAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()
        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)
        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)
        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)
        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)
        x = ops.u_mul_e_sum(graph, values, attn_probs)
        x = x.reshape(-1, self.dim)
        x = self.output_linear(x)
        x = self.dropout(x)
        return x


class TransformerAttentionSepModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()
        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)
        self.output_linear = nn.Linear(in_features=dim * 2, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)
        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)
        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)
        message = ops.u_mul_e_sum(graph, values, attn_probs)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)
        x = self.output_linear(x)
        x = self.dropout(x)
        return x
