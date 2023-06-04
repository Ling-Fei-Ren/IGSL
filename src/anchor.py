import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from .generic_utils import to_cuda, create_mask
from .constants import VERY_SMALL_NUMBER, INF


def sample_anchors(node_vec, s):
    idx = torch.randperm(node_vec.size(0))[:s]
    # idx=s
    return node_vec[idx], idx

class MLP_(nn.Module):
    def __init__(self, input_dim, output_dim, device=True):
        super(MLP_, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.mlp_layer = nn.Linear(self.input_dim, self.output_dim)
        self.weight = torch.Tensor(self.input_dim, self.output_dim).to(self.device)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
    def forward(self,features):
        self.mlp_layer.to(self.device)
        result = self.mlp_layer(features)
        result = F.relu(result)
        return result

class AnchorGraphLearner(nn.Module):
    def __init__(self, input_size, topk=None, epsilon=None, num_pers=16, device=None):
        super(AnchorGraphLearner, self).__init__()
        self.device = device
        self.topk = topk
        self.epsilon = epsilon
        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        self.MLP = MLP_(input_size, 2, self.device)

    def forward(self, context, anchors, ctx_mask=None, anchor_mask=None):

        # expand_weight_tensor = self.weight_tensor.unsqueeze(1).cuda()
        # context_fc = context.unsqueeze(0).to(self.device)* expand_weight_tensor.to(self.device)
        # context_norm = F.normalize(context_fc, p=2, dim=-1)
        # anchors_fc = anchors.unsqueeze(0).to(self.device)* expand_weight_tensor.to(self.device)
        # anchors_norm = F.normalize(anchors_fc, p=2, dim=-1)
        context_norm=self.MLP(context)
        anchors_norm = self.MLP(anchors)

        attention = torch.matmul(context_norm, anchors_norm.transpose(-1, -2))
        # adj_sqrt1 = torch.sqrt(torch.sum(context_norm ** 2, dim=1, keepdim=True))
        # adj_sqrt2 = torch.sqrt(torch.sum(anchors_norm ** 2, dim=1, keepdim=True))
        # adj_sqrt = torch.matmul(adj_sqrt1, adj_sqrt2.transpose(-1, -2))
        # attention = attention / torch.clamp(adj_sqrt,min=VERY_SMALL_NUMBER)



        # attention = torch.matmul(context_norm, anchors_norm.transpose(-1, -2)).mean(0)
        markoff_value = 0
        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if anchor_mask is not None:
            attention = attention.masked_fill_(1 - anchor_mask.byte().unsqueeze(-2), markoff_value)

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)

        return attention,context_norm

    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = to_cuda((markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists




def batch_sample_anchors(node_vec, ratio, node_mask=None, device=None):
    idx = []
    num_anchors = []
    max_num_anchors = 0
    for i in range(node_vec.size(0)):
        tmp_num_nodes = int(node_mask[i].sum().item())
        tmp_num_anchors = int(ratio * tmp_num_nodes)
        g_idx = torch.randperm(tmp_num_nodes)[:tmp_num_anchors]
        idx.append(g_idx)
        num_anchors.append(len(g_idx))

        if max_num_anchors < len(g_idx):
            max_num_anchors = len(g_idx)

    anchor_vec = batch_select_from_tensor(node_vec, idx, max_num_anchors, device)
    anchor_mask = create_mask(num_anchors, max_num_anchors, device)

    return anchor_vec, anchor_mask, idx, max_num_anchors

def batch_select_from_tensor(node_vec, idx, max_num_anchors, device=None):
    anchor_vec = []
    for i in range(node_vec.size(0)):
        tmp_anchor_vec = node_vec[i][idx[i]]
        if len(tmp_anchor_vec) < max_num_anchors:
            dummy_anchor_vec = to_cuda(torch.zeros((max_num_anchors - len(tmp_anchor_vec), node_vec.size(-1))), device)
            tmp_anchor_vec = torch.cat([tmp_anchor_vec, dummy_anchor_vec], dim=-2)
        anchor_vec.append(tmp_anchor_vec)

    anchor_vec = torch.stack(anchor_vec, 0)

    return anchor_vec

def compute_anchor_adj(node_anchor_adj, anchor_mask=None):
    '''Can be more memory-efficient'''
    anchor_node_adj = node_anchor_adj.transpose(-1, -2)
    anchor_norm = torch.clamp(anchor_node_adj.sum(dim=-2), min=VERY_SMALL_NUMBER) ** -1
    # anchor_adj = torch.matmul(anchor_node_adj, torch.matmul(torch.diag(anchor_norm), node_anchor_adj))
    anchor_adj = torch.matmul(anchor_node_adj, anchor_norm.unsqueeze(-1) * node_anchor_adj)

    markoff_value = 0
    if anchor_mask is not None:
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-1), markoff_value)
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-2), markoff_value)

    return anchor_adj



class AnchorGCNLayer(nn.Module):
    """
    Simple AnchorGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(AnchorGCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, anchor_mp=True, batch_norm=True):
        support = torch.matmul(input, self.weight)
        if anchor_mp:
            node_anchor_adj = adj
            node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=VERY_SMALL_NUMBER)
            anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))
        else:
            #标准的GCN算法
            node_adj = adj
            output = torch.matmul(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class AnchorGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(AnchorGCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(AnchorGCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(AnchorGCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(AnchorGCNLayer(nhid, nclass, batch_norm=False))


    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)

        return x

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
