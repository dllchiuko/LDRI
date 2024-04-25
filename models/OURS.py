import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding_layer import EmbeddingLayer
from .base_model import BaseModel
from .DeepFM import DeepFM
from .MLP import MLP
from .NFM import NFM
from .AFM import AFM
from .WideDeep import WideDeep


class OURS(nn.Module):
    def __init__(self,
                 device,
                 model_name,
                 strategy,
                 feature_size_map,
                 feature_size_map_item,
                 n_day,
                 embed_dim_sparse=16,
                 embed_dim_sparse_item=16,
                 backbone='MLP', use_dense=True
                 ):
        super().__init__()

        self.device = device
        self.model_name = model_name
        self.strategy = strategy
        self.n_day = n_day
        self.use_dense = use_dense

        if backbone == 'MLP':
            self.backbone = MLP(feature_size_map=feature_size_map,
                                hidden_dims=[64, 32],
                                dropout=[0.5, 0.5],
                                use_dense=self.use_dense)
        elif backbone == 'DeepFM':
            self.backbone = DeepFM(feature_size_map=feature_size_map,
                                   hidden_dims=[32, 32],
                                   dropout=[0.5, 0.5],
                                   use_dense=False)

        elif backbone == 'NFM':
            self.backbone = NFM(feature_size_map=feature_size_map,
                                embed_dim_dnn=16,
                                hidden_dims=[16, 8],
                                dropout=[0.5, 0.5],
                                use_dense=self.use_dense)
        elif backbone == 'AFM':
            self.backbone = AFM(feature_size_map=feature_size_map,
                                embed_dim_sparse=16,
                                dropout=0.5,
                                use_dense=self.use_dense)
        elif backbone == 'WideDeep':
            self.backbone = WideDeep(feature_size_map=feature_size_map,
                                     embed_dim_sparse=embed_dim_sparse,
                                     hidden_dims=[64, 64, 64],
                                     dropout=[0.3, 0.4, 0.5],
                                     use_dense=self.use_dense)

        else:
            print('No such model please check your input carefully!')

        self.feature_size_map_item = feature_size_map_item
        self.sparse_feature_size_item = (feature_size_map_item['id_feature_item'] +
                                         feature_size_map_item['cate_feature_item'])
        self.sparse_field_dim_item = len(self.sparse_feature_size_item)
        self.dense_field_dim_item = feature_size_map_item['dense_feature_item'][0] if use_dense else 0
        self.embed_dim_sparse_item = embed_dim_sparse_item
        self.embed_dim_cat_item = self.embed_dim_sparse_item * self.sparse_field_dim_item + \
                                  self.dense_field_dim_item

        self.embedding = EmbeddingLayer(self.sparse_feature_size_item, self.embed_dim_sparse_item, init=True, init_std=0.01)
        self.batch_norm_dense = nn.BatchNorm1d(self.dense_field_dim_item)
        self.perceptron = TimeSensPerceptron(device=self.device,
                                             strategy=strategy,
                                             embed_dim=self.embed_dim_cat_item,
                                             n_day=self.n_day)

    def forward(self, x, diff):
        ym = self.backbone(x)  # input all tensors including user and item into backbone model to calculate ym
        if self.model_name == 'TaFR':
            return ym

        else:
            if self.use_dense:
                x_item = x[:, [1, 2, 9, 10, 11, 12, 13, 14, 15, 16]]  # filter feat that are only about item
                embed_sparse = torch.cat(self.embedding(x_item[:, :self.sparse_field_dim_item].long()),
                                         dim=1)
                embed_dense = x_item[:, -self.dense_field_dim_item:].type(torch.float32)
                # embed_dense = self.batch_norm_dense(x_item[:, -self.dense_field_dim_item:].type(torch.float32)).float()
                # embed_dense = torch.where(torch.isnan(embed_dense), torch.zeros_like(embed_dense), embed_dense)

                embed = torch.cat([embed_sparse,
                                   embed_dense],
                                  dim=1).float()
            else:
                x_item = x[:, [1, 2, 9, 10, 11, 12]]  # filter feat that are only about item
                embed_sparse = torch.cat(self.embedding(x_item[:, :self.sparse_field_dim_item].long()),
                                         dim=1)
                embed = embed_sparse

            pcp_x = embed
            yt, yt_mat = self.perceptron(pcp_x, diff)  # 将item的特征送入time sensitivity perceptron  (bs, 1)

            # # only train ym:
            # yt = torch.zeros_like(ym)
            # yt_mat = torch.zeros((yt.shape[0], 30))

            # # only train yt:
            # ym = torch.zeros_like(yt)
            return ym, yt, yt_mat  # (bs, 1), (bs, 30)


class TimeSensPerceptron(nn.Module):
    def __init__(self, device, strategy, embed_dim, n_day):
        super().__init__()

        self.device = device
        self.strategy = strategy
        self.embed_dim = embed_dim
        self.n_day = n_day

        self.hidden_dim = [64, 32]
        self.linear = nn.Linear(self.embed_dim, self.n_day)
        self.proj = nn.Sequential(nn.Linear(self.embed_dim, self.hidden_dim[0]),
                                  nn.BatchNorm1d(self.hidden_dim[0]),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(self.hidden_dim[0], self.hidden_dim[1]),
                                  nn.BatchNorm1d(self.hidden_dim[1]),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_dim[1], self.n_day))
        self.relu = nn.ReLU()

    def forward(self, x, diff):
        x_orig = x.clone()
        x_orig = self.linear(x_orig)
        hidden = self.proj(x)
        # print(hidden.shape, x_orig.shape)
        hidden = hidden + x_orig
        out = hidden[torch.arange(diff.shape[0]), diff].float()

        if self.strategy == 'raw':
            out = out
        elif self.strategy == 'smooth':
            out += hidden[torch.arange(diff.shape[0]),
                          torch.where((diff - 1 == -1),
                                      torch.zeros_like(diff).to(self.device),  # condition == True
                                      diff - 1)].float()  # condition = False
            out += hidden[torch.arange(diff.shape[0]),
                          torch.where((diff + 1 == self.n_day),
                                      torch.full(size=diff.shape, fill_value=self.n_day - 1).to(self.device),
                                      diff + 1)]

            out = out / 3

        out = nn.Sigmoid()(out)
        return out.view(-1, 1), hidden



#
# class TimeSensPerceptron(nn.Module):
#     def __init__(self, device, strategy, embed_dim, n_day):
#         super().__init__()
#
#         self.device = device
#         self.strategy = strategy
#         self.embed_dim = embed_dim
#         self.n_day = n_day
#
#         self.hidden_dim = [128, 64, 32]
#
#         self.proj = nn.Sequential(nn.Linear(self.embed_dim, 64),
#                                   nn.ReLU(),
#                                   nn.Dropout(0.3),
#                                   nn.Linear(64, self.n_day),
#                                   nn.ReLU())
#
#         self.proj_second = nn.Linear(self.n_day, self.n_day)
#         # self.proj_third = nn.Linear(self.n_day, self.n_day)
#         self.relu = nn.ReLU()
#
#     def forward(self, x, diff):
#         hidden = self.proj(x)
#         hidden_orig = hidden.clone()
#         hidden = self.proj_second(hidden)
#         hidden = hidden + hidden_orig
#         # hidden = self.proj_third(hidden) + hidden_orig
#
#         # hidden = nn.Sigmoid()(hidden)
#         out = hidden[torch.arange(diff.shape[0]), diff].float()
#         # out = out / nn.Sigmoid()(diff)
#
#         if self.strategy == 'raw':
#             out = out
#         elif self.strategy == 'smooth':
#             out += hidden[torch.arange(diff.shape[0]),
#                           torch.where((diff - 1 == -1),
#                                       torch.zeros_like(diff).to(self.device),  # condition == True
#                                       diff - 1)].float()  # condition = False
#             out += hidden[torch.arange(diff.shape[0]),
#                           torch.where((diff + 1 == self.n_day),
#                                       torch.full(size=diff.shape, fill_value=self.n_day - 1).to(self.device),
#                                       diff + 1)]
#
#             out = out / 3
#
#         out = nn.Sigmoid()(out)
#         return out.view(-1, 1), hidden
