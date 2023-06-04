import torch
import torch.nn as nn
import torch.nn.functional as F


from .scalable_graphlearn import AnchorGraphLearner
from .anchor import AnchorGCN   # GCN网络嵌入代码AnchorGCN
from .generic_utils import to_cuda, normalize_adj
from .constants import VERY_SMALL_NUMBER



class GraphClf(nn.Module):
    def __init__(self, config):
        super(GraphClf, self).__init__()
        self.config = config
        self.name = 'GraphClf'
        self.graph_learn = config['graph_learn']
        self.graph_metric_type = config['graph_metric_type']
        self.device = config['device']
        nfeat = config['num_feat']
        nclass = config['num_class']
        hidden_size = config['hidden_size']
        self.dropout = config['dropout']
        self.graph_skip_conn = config['graph_skip_conn']
        self.graph_include_self = config.get('graph_include_self', True)
        self.scalable_run = config.get('scalable_run', False)

        # GCN网络嵌入代码encoder
        self.encoder = AnchorGCN(nfeat=nfeat,
                                nhid=hidden_size,
                                nclass=nclass,
                                graph_hops=config.get('graph_hops', 2),
                                dropout=self.dropout,
                                batch_norm=config.get('batch_norm', False))

        self.encoder=self.encoder.to(self.device)
        if self.graph_learn:
            self.graph_learner = AnchorGraphLearner(nfeat, config['graph_learn_hidden_size'],
                                            topk=config['graph_learn_topk'],
                                            epsilon=config['graph_learn_epsilon'],
                                            num_pers=config['graph_learn_num_pers'],
                                            metric_type=config['graph_metric_type'],
                                            device=self.device)

            self.graph_learner = self.graph_learner.to(self.device)
            self.graph_learner2 = AnchorGraphLearner(hidden_size,
                                            config.get('graph_learn_hidden_size2',
                                            config['graph_learn_hidden_size']),
                                            topk=config.get('graph_learn_topk2', config['graph_learn_topk']),
                                            epsilon=config.get('graph_learn_epsilon2', config['graph_learn_epsilon']),
                                            num_pers=config['graph_learn_num_pers'],
                                            metric_type=config['graph_metric_type'],
                                            device=self.device)
            self.graph_learner2 = self.graph_learner2.to(self.device)
            print('[ Graph Learner ]')
            if config['graph_learn_regularization']:
              print('[ Graph Regularization]')
        else:
            self.graph_learner = None

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None, anchor_features=None):
        if self.graph_learn:
            if self.scalable_run:
                node_anchor_adj = graph_learner(node_features, anchor_features)
                return node_anchor_adj

            else:
                raw_adj = graph_learner(node_features)

                if self.graph_metric_type in ('kernel', 'weighted_cosine'):
                    assert raw_adj.min().item() >= 0
                    adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

                elif self.graph_metric_type == 'cosine':
                    adj = (raw_adj > 0).float()
                    adj = normalize_adj(adj)

                else:
                    adj = torch.softmax(raw_adj, dim=-1)

                if graph_skip_conn in (0, None):
                    if graph_include_self:
                        adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
                else:
                    adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

                return raw_adj, adj

        else:
            raw_adj = None
            adj = init_adj

            return raw_adj, adj


    def forward(self, node_features, init_adj=None):
        node_features = F.dropout(node_features, self.config.get('feat_adj_dropout', 0), training=self.training)
        raw_adj, adj = self.learn_graph(self.graph_learner, node_features, self.graph_skip_conn, init_adj=init_adj)
        adj = F.dropout(adj, self.config.get('feat_adj_dropout', 0), training=self.training)
        node_vec = self.encoder(node_features, adj)
        output = F.log_softmax(node_vec, dim=-1)
        return output, adj

