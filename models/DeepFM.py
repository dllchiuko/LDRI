import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('embedding_layer.py')

from .embedding_layer import EmbeddingLayer


class DeepFM(nn.Module):
    def __init__(self,
                 feature_size_map,
                 embedding_dim=16,
                 hidden_dims=[32, 32],
                 dropout=[0.5, 0.5],
                 use_dense=False):
        super(DeepFM, self).__init__()

        self.feature_size_map = feature_size_map
        self.sparse_feature_size = feature_size_map['id_feature'] + feature_size_map['cate_feature']
        self.sparse_field_dim = len(self.sparse_feature_size)
        self.dense_field_dim = feature_size_map['dense_feature'][0] if use_dense else 0
        self.all_field_dim = len(self.sparse_feature_size) + self.dense_field_dim
        self.use_dense = use_dense

        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.use_dense = use_dense

        self.fm_bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.fm_first_order_dense_linear = nn.Linear(self.dense_field_dim, 1) if self.use_dense else None
        self.fm_first_order_sparse_embedding = EmbeddingLayer(self.sparse_feature_size, 1, init=True, init_std=0.1)
        self.fm_second_order_sparse_embedding = EmbeddingLayer(self.sparse_feature_size, self.embedding_dim, init=True, init_std=0.1)

        self.dnn_out = nn.Sequential(nn.Linear(self.all_field_dim * self.embedding_dim, self.hidden_dims[0]),
                                     nn.BatchNorm1d(hidden_dims[0]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout[0]),
                                     nn.Linear(self.hidden_dims[0], hidden_dims[1]),
                                     nn.BatchNorm1d(hidden_dims[1]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout[1]))
        self.dnn_dense_linear = nn.Linear(self.dense_field_dim, self.all_field_dim * self.embedding_dim) if self.use_dense else None
        self.dnn_out_linear = nn.Linear(self.hidden_dims[-1], 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """fm part"""
        # first order
        fm_first_order_sparse = self.fm_first_order_sparse_embedding(x[:, :self.sparse_field_dim].long())  # [tensor(batch_size, 1), ...]
        fm_first_order_sparse = [i.reshape(-1, 1) for i in fm_first_order_sparse]  # [tensor(batch_size, 1), ...]
        fm_first_order_sparse = torch.cat(fm_first_order_sparse, dim=-1)  # [batch_size, field_size]
        fm_first_order_sparse = torch.sum(fm_first_order_sparse, dim=1, keepdim=True)  # [batch_size, 1]

        if self.use_dense:
            fm_first_order_dense = self.fm_first_order_dense_linear(x[:, -1])
            fm_first_order_out = fm_first_order_sparse + fm_first_order_dense
        else:
            fm_first_order_out = fm_first_order_sparse

        # second order
        fm_second_order_sparse = self.fm_second_order_sparse_embedding(x[:, :self.sparse_field_dim].long())  # vixi  [tensor(batch_size, embedding_dim), ...]
        fm_second_order_sparse = [i.unsqueeze(1).reshape(-1, 1, self.embedding_dim)
                                  for i in fm_second_order_sparse]
        fm_second_order_sparse = torch.cat(fm_second_order_sparse, dim=1)  # [batch_size, field_size, embedding_dim]
        fm_second_order_sparse_sum = torch.sum(fm_second_order_sparse, dim=1, keepdim=False)  # [batch_size, embedding_dim]  sum(vixi)
        fm_second_order_sparse_sum_square = torch.pow(fm_second_order_sparse_sum, 2)  # (sum(vixi))**2

        fm_second_order_sparse_square = torch.pow(fm_second_order_sparse, 2)  # vi**2xi**2  [batch_size, field_size, embedding_dim]
        fm_second_order_sparse_square_sum = torch.sum(fm_second_order_sparse_square, dim=1)  # sum(vi**2xi**2)  [batch_size, embedding_dim]

        fm_second_order = (fm_second_order_sparse_sum_square - fm_second_order_sparse_square_sum) * 0.5  # [batch_size, embedding_dim]
        fm_second_order_out = torch.sum(fm_second_order, dim=1, keepdim=True)  # [batch_size, 1]

        fm_out = self.fm_bias + fm_first_order_out + fm_second_order_out

        """dnn part"""
        dnn_sparse = torch.flatten(fm_second_order_sparse, 1)  # [batch_size, field_size * embedding_dim]

        if self.use_dense:
            dnn_dense = nn.ReLU()(self.dnn_dense_linear(x[:, -self.dense_field_dim:]))
            dnn_hidden = dnn_sparse + dnn_dense

        else:
            dnn_hidden = dnn_sparse

        dnn_out = self.dnn_out(dnn_hidden)  # [batch_size, field_size * embedding_dim] -> hidden_dim[0] -> hidden_dim[1]
        dnn_out = self.dnn_out_linear(dnn_out)  # [batch_size, 1]

        out = self.sigmoid(fm_out + dnn_out)  # [batch_size, 1]

        return out



