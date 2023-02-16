import torch
import torch.nn as nn
import torch.nn.functional as F
from gat_layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, args, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, args, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # delete last gat layer for classification -> In gtn, using linear layer for classification
        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge_weight) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        return x