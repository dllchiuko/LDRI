import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
# from old.models.embedding_layer import *
from .embedding_layer import EmbeddingLayer


class AFM(BaseModel):
    def __init__(self,
                 feature_size_map,
                 embed_dim_sparse=16,
                 dropout=0.5,
                 use_dense=True):
        super(AFM, self).__init__(feature_size_map=feature_size_map,
                                  embed_dim_sparse=embed_dim_sparse,
                                  use_dense=use_dense
                                  )

        self.fm_bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.fm_first_order_dense_linear = nn.Linear(self.dense_field_dim, 1) if self.use_dense else None
        self.fm_first_order_sparse_embedding = EmbeddingLayer(self.sparse_feature_size, 1, init=True, init_std=0.1)

        self.attn_embedding = EmbeddingLayer(self.sparse_feature_size, self.embed_dim_sparse, init=True, init_std=0.1)
        self.attention = nn.Linear(self.embed_dim_sparse, self.embed_dim_sparse, bias=True)
        self.projection = nn.Linear(self.embed_dim_sparse, 1, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_out = nn.Linear(self.embed_dim_sparse, 1, bias=False)

    def forward(self, x):
        """fm part"""
        # first order
        fm_first_order_sparse = self.fm_first_order_sparse_embedding(
            x[:, :self.sparse_field_dim].long())  # [tensor(batch_size, 1), ...]
        fm_first_order_sparse = [i.reshape(-1, 1) for i in fm_first_order_sparse]  # [tensor(batch_size, 1), ...]
        fm_first_order_sparse = torch.cat(fm_first_order_sparse, dim=-1)  # [batch_size, field_size]
        fm_first_order_sparse = torch.sum(fm_first_order_sparse, dim=1, keepdim=True)  # [batch_size, 1]

        if self.use_dense:
            fm_first_order_dense = self.fm_first_order_dense_linear(x[:, -self.dense_field_dim:].float())
            fm_first_order_dense = torch.where(torch.isnan(fm_first_order_dense),
                                               torch.zeros_like(fm_first_order_dense),
                                               fm_first_order_dense)
            fm_first_order_out = fm_first_order_sparse + fm_first_order_dense
        else:
            fm_first_order_out = fm_first_order_sparse  # [batch_size, 1]

        """attention part"""
        attn_embed_sparse = self.attn_embedding(x[:, :self.sparse_field_dim].long())
        attn_embed_sparse = [i.unsqueeze(1).reshape(-1, 1, self.embed_dim_sparse)
                             for i in attn_embed_sparse]
        attn_embed_sparse = torch.cat(attn_embed_sparse, dim=1)  # [batch_size, field_size, embedding_dim]

        row, col = list(), list()
        for i in range(self.sparse_field_dim - 1):
            for j in range(i + 1, self.sparse_field_dim):
                row.append(i), col.append(j)
        p, q = attn_embed_sparse[:, row], attn_embed_sparse[:, col]
        inner_product = p * q  # [None, field_size*(field_size-1)/2, embedding_dim]

        attn_scores = self.relu(self.attention(inner_product))
        attn_scores = torch.softmax(self.projection(attn_scores), dim=1)

        attn_hidden = torch.sum(attn_scores * inner_product, dim=1)  # (bs, embed_dim_sparse)
        attn_hidden = self.attn_dropout(attn_hidden)  # (bs, embed_dim_sparse)
        attn_out = self.attn_out(attn_hidden)  # (bs, 1)

        out = fm_first_order_out + attn_out

        return self.sigmoid(out)



