import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

"""
GCN implementation

    Original GCN pytorch Code : https://github.com/tkipf/pygcn
    Origianl Paper : https://arxiv.org/pdf/1609.02907

    Z = A * X * W

    # N : number of samples : coar - 2708
      M : number of feature : coar - 1433
    X : node feature [N, M]
    A : normalized adjacency + identity matrix [N, N]
    W : 1st-layer Weight [M, 16], 2nd-layer Weight [16, F]
    bias : [16]

"""
class GraphConvolution(Module):

    # set dimension of input output filter bias
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    #reset parameters : weight , bias
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        # X*W
        support = torch.mm(input, self.weight)
        # A * (XW)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

""" In paper, they consider a two-layer GCN for semi-supervised node classification on graph.

    using two gcn layer
    
    Z = f(X,A) = softmax(A ReLU (AXW)W')

    x : node feature
        In the Chebyshev polynomizals, they set K = 1
        so using just x
    adj : normalized adj (+ self loop)
"""
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        #layer1 = ReLU(A_nXW)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        #layer2 = (A_n result1 W')
        x = self.gc2(x, adj)

        #return with softmax
        return F.log_softmax(x, dim=1)
    
