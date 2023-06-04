import torch
import torch.nn as nn
from torch.nn import init
from .graph_clf import GraphClf  #GCN网络嵌入代码
import torch.nn.functional as F
"""
	PC-GNN Model
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class PCALayer(nn.Module):
    """
	One Pick-Choose-Aggregate layer
	"""

    def __init__(self, num_classes, inter1, lambda_1):
        """
		Initialize the PC-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
        super(PCALayer, self).__init__()
        self.inter1 = inter1
        self.xent = nn.CrossEntropyLoss()
        # the parameter to transform the final embedding
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
        init.xavier_uniform_(self.weight)

        self.weight1 = nn.Parameter(torch.FloatTensor(num_classes, inter1.mlp_num_feat))
        init.xavier_uniform_(self.weight1)

        self.lambda_1 = lambda_1
        self.epsilon = 0.1

    def forward(self, nodes, labels, train_flag=True):#前向传播
        embeds1,embeds2, loss= self.inter1(nodes, labels, train_flag)
        scores = self.weight.mm(embeds1)
        scores1 = self.weight1.mm(embeds2)
        return scores.t(),scores1.t(),loss

    def to_prob(self, nodes, labels, train_flag=False):#验证与测试入口
        gnn_logits,embeds2, loss= self.forward(nodes, labels, train_flag)
        gnn_scores = torch.sigmoid(gnn_logits)
        return gnn_scores

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores, gnn_scores1, loss= self.forward(nodes, labels, train_flag)
        # gnn_loss = self.xent(gnn_scores, labels.squeeze())  # 最终损失
        gnn_loss = compute_focal_loss(gnn_scores, labels.squeeze())  # 最终损失
        #
        # gnn_loss1 = self.xent(gnn_scores1, labels.squeeze())  # 最终损失
        gnn_loss1 = compute_focal_loss(gnn_scores1, labels.squeeze())  # 最终损失
        final_loss = gnn_loss+0.1*gnn_loss1+0.0*loss

        return final_loss


class Model(object):  #GCN网络嵌入代码
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        self.config = config
        self.net_module = GraphClf #GCN网络嵌入代码
        self._init_new_network()

    def _init_new_network(self):
        self.network = self.net_module(self.config)

def compute_focal_loss(logits, labels, class_num=2):
    '''
    :param logits:
    :param labels:
    :return:
    '''
    labels = torch.reshape(labels, [-1])
    labels = F.one_hot(labels, class_num)
    # logits=torch.relu(logits)
    pred = F.softmax(logits)
    focal_loss= -1 * torch.sum(0.9*torch.pow(1- pred, 2) *torch.log(pred)*labels+ 0.1*torch.pow(pred, 2) *torch.log(1-pred)*(1-labels))

    return focal_loss
