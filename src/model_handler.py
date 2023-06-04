import time, datetime
import os
import random
import pickle
import argparse
import numpy as np
import torch.cuda
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import networkx as nx
from src.utils import test_pcgnn, test_sage, load_data, pos_neg_split, normalize, pick_step,train_pcgnn
from src.model import PCALayer
from src.layers import InterAgg, IntraAgg
from src.layers_1 import InterAgg_1, IntraAgg_1
from src.graphsage import *


"""
	Training PC-GNN
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""


class ModelHandler(object):
	def __init__(self, config):
		args = argparse.Namespace(**config)
		# 导入图、特征和标签
		relations, feat_data, labels = load_data(args.data_name, prefix=args.data_dir)
		# train_test split
		np.random.seed(args.seed)
		random.seed(args.seed)
		# 测试集、验证集和测试集划分
		if args.data_name == 'yelp':# yelp
			index = list(range(len(labels)))
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio,
																	random_state=2, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
																	random_state=2, shuffle=True)
			config['num_feat'] = 32


		elif args.data_name == 'amazon':  # amazon
			# 0-3304 are unlabeled nodes
			index = list(range(3305, len(labels)))
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[3305:], stratify=labels[3305:],
																	train_size=args.train_ratio, random_state=2, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
																	test_size=args.test_ratio, random_state=2, shuffle=True)
			config['num_feat'] = 25

		if args.data_name == 'mimic': # mimic
			index = list(range(len(labels)))
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio,
																	random_state=2, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
																	random_state=2, shuffle=True)
			config['num_feat'] = 50


		print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'+
			f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')
		print(f"Classification threshold: {args.thres}")
		print(f"Feature dimension: {feat_data.shape[1]}")
		config['num_class'] = 2

		if not config['no_cuda'] and torch.cuda.is_available():
			print('[ Using CUDA ]')
			self.device = torch.device('cuda' if config['cuda_id'] < '0' else 'cuda:{}'.format(config['cuda_id']))
		else:
			self.device = torch.device('cpu')

		config['device'] = self.device
		self.config = config

		args.cuda =not config['no_cuda'] and torch.cuda.is_available()
		feat_data = normalize(feat_data)

		train_pos, train_neg = pos_neg_split(idx_train, y_train)

		# set input graph
		if args.model == 'SAGE' or args.model == 'GCN':
			adj_lists = relations[0]
		else:
			if(args.data_name !='mimic'):
				adj_lists = [torch.Tensor(relations[1]).to(self.device),torch.Tensor(relations[2]).to(self.device), torch.Tensor(relations[3]).to(self.device)]
			else:
				adj_lists = [torch.Tensor(relations[1]).to(self.device), torch.Tensor(relations[2]).to(self.device),
							 torch.Tensor(relations[3]).to(self.device),torch.Tensor(relations[4]).to(self.device)]

		print(f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')
		
		self.args = args
		self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': relations[0],
						'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
						'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,'train_pos': train_pos}


	def train(self):#  train
		args = self.args
		feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']# 节点特征+归一化邻接矩阵
		idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']# 测试集

		idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset['idx_test'], self.dataset['y_test']#验证集和测试集

		features= torch.Tensor(feat_data).to(self.device)
		# build one-layer models
		if args.model == 'PCGNN':  #默认PCNN
			if(args.data_name !='mimic'):#yelpchi和amazon采用3子图
				intra1 = IntraAgg(self.config,features, feat_data.shape[1], args.emb_size)
				intra2 = IntraAgg(self.config,features, feat_data.shape[1], args.emb_size)
				intra3 = IntraAgg(self.config,features, feat_data.shape[1], args.emb_size)
				inter1 = InterAgg(self.config,features, feat_data.shape[1], args.emb_size,
							  adj_lists, [intra1, intra2, intra3],  inter=args.multi_relation, device=self.device)
			else:  #mimic数据集采用的是4个子图
				intra1 = IntraAgg_1(self.config, features, feat_data.shape[1],  args.emb_size)
				intra2 = IntraAgg_1(self.config, features, feat_data.shape[1],  args.emb_size)
				intra3 = IntraAgg_1(self.config, features, feat_data.shape[1],  args.emb_size)
				intra4 = IntraAgg_1(self.config, features, feat_data.shape[1],  args.emb_size)
				inter1 = InterAgg_1(self.config, features, feat_data.shape[1], args.emb_size,
								  adj_lists, [intra1, intra2, intra3, intra4],  inter=args.multi_relation, device=self.device)

		elif args.model == 'SAGE':
			agg_sage = MeanAggregator(features, cuda=args.cuda)
			enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False, cuda=args.cuda)
		elif args.model == 'GCN':
			agg_gcn = GCNAggregator(features, cuda=args.cuda)
			enc_gcn = GCNEncoderGCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True, cuda=args.cuda)

		if args.model == 'PCGNN':
			gnn_model = PCALayer(2, inter1, args.alpha)
		elif args.model == 'SAGE':
			enc_sage.num_samples = 5
			gnn_model = GraphSage(2, enc_sage)
		elif args.model == 'GCN':
			gnn_model = GCN(2, enc_gcn)

		gnn_model=gnn_model.to(self.device)
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
		timestamp = time.time()
		timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
		dir_saver = args.save_dir + timestamp
		path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
		f1_mac_best, auc_best, ep_best = 0, 0, -1
		test_flag = 0
		error_list=[]
		# train the model
		for epoch in range(args.num_epochs):
			sampled_idx_train = pick_step(idx_train, y_train,self.dataset['labels'], self.dataset['homo'],
										  size=len(self.dataset['train_pos']) * 2)# 训练集的欠采样和过采样


			random.shuffle(sampled_idx_train)

			num_batches = int(len(sampled_idx_train) / args.batch_size) + 1# batch

			loss = 0.0
			epoch_time = 0
			# mini-batch training
			for batch in range(num_batches):
				start_time = time.time()
				i_start = batch * args.batch_size
				i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
				batch_nodes = sampled_idx_train[i_start:i_end]
				batch_label = self.dataset['labels'][np.array(batch_nodes)]
				num_1 = len(np.where(batch_label == 1)[0])
				num_2 = len(np.where(batch_label == 0)[0])
				p0 = (num_1 / (num_1 + num_2))
				p1 = 1 - p0
				prior = np.array([p1, p0])
				optimizer.zero_grad()
				if args.cuda:
					loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label).to(self.device)))
				else:
					loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
				loss.backward()
				optimizer.step()
				end_time = time.time()
				epoch_time += end_time - start_time
				loss += loss.item()

			print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

			# Valid the model for every $valid_epoch$ epoch
			if epoch % args.valid_epochs == 0:  #验证集测试
				if args.model == 'SAGE' or args.model == 'GCN':
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
					if auc_val >= auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)
				else:
					# error_list = train_pcgnn(idx_train, y_train, test_flag,gnn_model, prior, args.batch_size, epoch, args.thres)
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_pcgnn(idx_valid, y_valid,test_flag, gnn_model, args.batch_size, epoch, args.thres)
					if auc_val >= auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)

		print("Restore model from epoch {}".format(ep_best))
		print("Model path: {}".format(path_saver))
		gnn_model.load_state_dict(torch.load(path_saver))
		if args.model == 'SAGE' or args.model == 'GCN':#测试集验证
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_sage(idx_test, y_test, gnn_model, args.batch_size, args.thres)
		else:
			test_flag=1
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_pcgnn(idx_test, y_test, test_flag, gnn_model,args.batch_size, args.thres)
		return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test
