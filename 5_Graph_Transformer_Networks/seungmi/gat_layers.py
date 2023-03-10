import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, n, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.zero_vec = torch.zeros_like(torch.empty(n, n))

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    '''
    In Paper definition,
        for last layer, averaging multi heads attention and using activation function(nonlinearity) - ELU 
        for other layer, concatenation attention coefficients and using nonlinearity activation function - softmax/logistic sigmoid
    In experimental, 
        for first layer, using K = 8 attention heads -> F' = 8 features each nodes
                         using ELU activation function [nonlinearity]
        for second layer(prediction layer), using single attention head 
                                            using softmax activation activation function
    '''
    def forward(self, h, edge_index):
        
        #calculate attention coefficient
        #usgin shared weight W
        # self.W -> get weight from input
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        attention = self.zero_vec.to(h.device)
        #torch.where(condition , [x|if True, x] , [y|if Flase, y])
        #calculate source node which connected with target node

        # edge_weight = edge_weight.to_dense()
        # for i in range(edge_weight.size(0)):
        #     for j in range(edge_weight.size(1)):
        #         if edge_weight[i][j] > 0:
        #             attention[i][j] = e[i][j]
        #         else:
        #             attention[i][j] = zero_vec[i][j]
        for i in range(edge_index.size(-1)):
            attention[edge_index[0][i]][edge_index[1][i]] = e[edge_index[0][i]][edge_index[1][i]]
        # attention = torch.where(edge_weight > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        #In Transductive learning -> dropout input as p = 0.6
        attention = F.dropout(attention, self.dropout, training=self.training)

        #calculate feature representation
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

