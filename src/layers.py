import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# from anchor import sample_anchors, compute_anchor_adj
import pickle as pkl
from operator import itemgetter
import math
from .anchor import MLP_
from .anchor import AnchorGraphLearner, sample_anchors, GraphAttentionLayer,SpGraphAttentionLayer
from .generic_utils import to_cuda
from . import constants as Constants
from .constants import VERY_SMALL_NUMBER, INF
from .model import compute_focal_loss
"""
    PC-GNN Layers
    Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
    Modified from https://github.com/YingtongDou/CARE-GNN
"""
from .model import Model   #GCN网络嵌入代码



#InterAgg
class InterAgg(nn.Module):

    def __init__(self, config,features, feature_dim, embed_dim,
                 adj_lists, intraggs,  inter='GNN', device=True):
        """
        Initialize the inter-relation aggregator
        :param features: the input node features or embeddings for all nodes
        :param feature_dim: the input dimension
        :param embed_dim: the embed dimension
        :param train_pos: positive samples in training set
        :param adj_lists: a list of adjacency lists for each single-relation graph
        :param intraggs: the intra-relation aggregators used by each single-relation graph
        :param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
        :param cuda: whether to use GPU
        """
        super(InterAgg, self).__init__()
        self.features = features
        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.intra_agg1 = intraggs[0]
        self.intra_agg2 = intraggs[1]
        self.intra_agg3 = intraggs[2]
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.config=config
        self.device=config['device']
        self.intra_agg1= self.intra_agg1.to(self.device)
        self.intra_agg2= self.intra_agg2.to(self.device)
        self.intra_agg3= self.intra_agg3.to(self.device)
        self.mlp_num_feat=self.config['mlp_num_feat']
        self.MLP=MLP_(self.feat_dim, self.mlp_num_feat, self.device)
        # initial filtering thresholds
        self.hidden_size=config['hidden_size']
        # parameter used to transform node embeddings before inter-relation aggregation
        self.weight = nn.Parameter(torch.FloatTensor(config['hidden_size']*len(intraggs)+self.config['mlp_num_feat'], self.embed_dim))
        init.xavier_uniform_(self.weight)

        self.weight2 = nn.Parameter(torch.FloatTensor(config['hidden_size'], 2))
        init.xavier_uniform_(self.weight2)

        self.weight3 = nn.Parameter(torch.FloatTensor(config['hidden_size'], 2))
        init.xavier_uniform_(self.weight3)


    def forward(self, nodes, labels,  train_flag=True):
        """
        :param nodes: a list of batch node ids
        :param labels: a list of batch node labels
        :param train_flag: indicates whether in training or testing mode
        :return combined: the embeddings of a batch of input node features
        :return center_scores: the label-aware scores of batch nodes
        """
        self.features_1=self.MLP(self.features)
        # 三个子图GNN
        r1_feats,loss1 = self.intra_agg1.forward(self.features_1,self.adj_lists[0], labels, nodes, train_flag)
        r2_feats,loss2 = self.intra_agg2.forward(self.features_1,self.adj_lists[1], labels, nodes, train_flag)
        r3_feats,loss3 = self.intra_agg3.forward(self.features_1,self.adj_lists[2], labels, nodes, train_flag)

        # get features or embeddings for batch nodes
        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).to(self.device)
        else:
            index = torch.LongTensor(nodes)
        self_feats = self.features_1[index]
        r1_feats = r1_feats[index]
        r2_feats = r2_feats[index]
        r3_feats = r3_feats[index]
        cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)
        combined = torch.relu(cat_feats.mm(self.weight).t())
        feats = self.features_1[index].t()
        loss=loss1+loss2+loss3
        return combined,feats,loss

class GCNLayer(nn.Module):
#     """
#     Simple AnchorGCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)
        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, anchor_mp=True,  batch_norm=True):
        support = torch.matmul(input, self.weight)
        if anchor_mp:
            node_anchor_adj = adj
            # adj_construct = torch.matmul(node_anchor_adj, node_anchor_adj.transpose(-1, -2))
            # adj_sqrt = torch.sqrt(torch.sum(node_anchor_adj ** 2, dim=1, keepdim=True))
            # adj_sqrt = torch.matmul(adj_sqrt, adj_sqrt.transpose(-1, -2))
            # adj_normalize = adj_construct / torch.clamp(adj_sqrt,min=VERY_SMALL_NUMBER)
            # adj_sum=torch.sum(adj_normalize, dim=1, keepdim=True)
            # adj_normalize=adj_normalize/adj_sum
            # output = torch.matmul(adj_normalize, support)

            node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True),min=VERY_SMALL_NUMBER)
            anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True),min=VERY_SMALL_NUMBER)
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))
        else:
            node_adj = adj
            output = torch.matmul(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output



class Anchor_GCNLayer(nn.Module):
#     """
#     Simple AnchorGCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
    def __init__(self, config, device, in_features, out_features, bias=False, batch_norm=False):
        super(Anchor_GCNLayer, self).__init__()
        self.config= config
        self.device = device
        self.xent = nn.CrossEntropyLoss()
        # nfeat, nhid, nclass, dropout, alpha, nheads
        self.mlp_layer = nn.Linear(config['hidden_size'], 2)
        self.mlp_layer1 = nn.Linear(config['hidden_size'], 2)
        # self.graph_encoders_x = SpGATLayer(in_features, out_features, out_features, 0.8, 0.2, 2).to(self.device)
        self.graph_encoders = GCNLayer(in_features, out_features)
        self.graph_encoders_1 = GCNLayer(out_features, out_features)
        self.graph_learner = AnchorGraphLearner(in_features,
                                                topk=config['graph_learn_topk'],
                                                epsilon=config['graph_learn_epsilon'],
                                                num_pers=config['graph_learn_num_pers'],
                                                device=self.device)

        self.graph_learner_1 = AnchorGraphLearner(out_features,
                                                topk=config['graph_learn_topk'],
                                                epsilon=config['graph_learn_epsilon'],
                                                num_pers=config['graph_learn_num_pers'],
                                                device=self.device)

    def forward(self, features, intra_agg,  labels, nodes, train_flag, batch_norm=True):
        init_adj = intra_agg
        init_node_vec = features
        loss1=0
        # init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, nodes)

        init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, self.config.get('num_anchors'))
        cur_node_anchor_adj, context_norm = self.graph_learner(init_node_vec, init_anchor_vec)
        first_init_agg_vec = self.graph_encoders(init_node_vec, cur_node_anchor_adj,anchor_mp=True)
        init_agg_vec=self.graph_encoders(init_node_vec,init_adj,anchor_mp=False)
        node_vec=self.config["graph_skip_conn"]*init_agg_vec+ (1-self.config["graph_skip_conn"])*first_init_agg_vec
        node_vec = torch.relu(node_vec)  # relu函数
        anchor_vec = node_vec[sampled_node_idx]
        max_iter_ = self.config.get('max_iter', 10)
        eps_adj = float(self.config.get('eps_adj', 0))  # cora: 5.5e-8, cora w/o input graph: 1e-8, citeseer w/o input graph: 1e-8, wine: 2e-5, cancer: 2e-5, digtis: 2e-5
        pre_node_anchor_adj = cur_node_anchor_adj

        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).to(self.device)
        else:
            index = torch.LongTensor(nodes)
        if train_flag== True:
            # loss1 =self.xent(context_norm[index], labels.squeeze())
            loss1 = compute_focal_loss(context_norm[index], labels.squeeze())
        loss = 0
        iter_ = 0
        while self.config['graph_learn'] and (iter_ == 0 or diff(cur_node_anchor_adj, pre_node_anchor_adj,
                                                                 cur_node_anchor_adj).item() > eps_adj) and iter_ < max_iter_:
            iter_ += 1
            pre_node_anchor_adj = cur_node_anchor_adj
            cur_node_anchor_adj,context_norm = self.graph_learner_1(node_vec, anchor_vec)
            cur_agg_vec = self.graph_encoders_1(node_vec, cur_node_anchor_adj,anchor_mp=True)
            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_agg_vec = update_adj_ratio * cur_agg_vec + (1 - update_adj_ratio) * first_init_agg_vec
            node_vec = (1 - self.config["graph_skip_conn"]) * cur_agg_vec + self.config["graph_skip_conn"] * init_agg_vec

            if self.cuda and isinstance(nodes, list):
                index = torch.LongTensor(nodes).to(self.device)
            else:
                index = torch.LongTensor(nodes)
            if train_flag == True:
                # loss += self.xent(context_norm[index], labels.squeeze())
                loss += compute_focal_loss(context_norm[index], labels.squeeze())

            node_vec = torch.relu(node_vec)
            anchor_vec = node_vec[sampled_node_idx]

        if iter_ > 0:
            loss = (loss +loss1)/ (iter_ +1)
        else:
            loss = loss1

        return init_agg_vec,loss

#子图嵌入
class IntraAgg(nn.Module):

    def __init__(self, config,features, feat_dim , embed_dim):
        """
        Initialize the intra-relation aggregator
        :param features: the input node features or embeddings for all nodes
        :param feat_dim: the input dimension
        :param embed_dim: the embed dimension
        :param train_pos: positive samples in training set
        :param rho: the ratio of the oversample neighbors for the minority class
        :param cuda: whether to use GPU
        """
        super(IntraAgg, self).__init__()
        self.config=config
        self.device=self.config['device']
        self.features = features
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.xent = nn.CrossEntropyLoss()
        self.model = Anchor_GCNLayer(self.config, self.device, self.config['mlp_num_feat'],config['hidden_size'])
        self.model = self.model.to(self.device)#标准GNN的入口函数

        self.model1 = Anchor_GCNLayer(self.config, self.device, self.config['hidden_size']+self.config['hidden_size'], config['hidden_size'])
        self.model1 = self.model1.to(self.device)  # 标准GNN的入口函数

        self.weight = torch.Tensor(self.config['mlp_num_feat'], config['hidden_size'])
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))

    def forward(self, features, intra_agg, labels, nodes, train_flag):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param to_neighs_list: neighbor node id list for each batch node in one relation
        :param batch_scores: the label-aware scores of batch nodes
        :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
        :param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes
        :param train_flag: indicates whether in training or testing mode
        :param sample_list: the number of neighbors kept for each batch node in one relation
        :return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
        :return samp_scores: the average neighbor distances for each relation after filtering
        """
        init_adj=intra_agg  #归一化的邻接矩阵
        init_node_vec = features  #节点初始特征矩阵
        init_agg_vec1= self.model(init_node_vec, init_adj, labels, nodes, train_flag,batch_norm=False)  # 基于邻接矩阵和初始特征的一层GCN嵌入, anchor_mp=False 表示标准的GCN，非anchor

        return init_agg_vec1

def diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_





