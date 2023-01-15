import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj): # Eq 1,2 
        #print('in GAT-forward')
        #print('## 0 ', x.shape) torch.Size([2708, 1433])
        x = F.dropout(x, self.dropout, training=self.training)
        #print('## 1 ', x.shape) torch.Size([2708, 1433])
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # print('## 2 ', x.shape) torch.Size([2708, 64])
        x = F.dropout(x, self.dropout, training=self.training)
        #print('## 3 ', x.shape) torch.Size([2708, 64])
        #print('## 3-2 ', self.out_att(x, adj).shape) torch.Size([2708, 7])
        x = F.elu(self.out_att(x, adj))
        #print('## 4 ', x.shape) torch.Size([2708, 7])
        #print('## 5 ', F.log_softmax(x, dim=1).shape) torch.Size([2708, 7])
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

