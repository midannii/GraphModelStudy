import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing 
## source: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html

'''
### `self.propagate` from `MessagePassing`

def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
    #The initial call to start propagating messages.
    decomposed_layers = 1 if self.explain else self.decomposed_layers

    for hook in self._propagate_forward_pre_hooks.values():
        res = hook(self, (edge_index, size, kwargs))
        if res is not None:
            edge_index, size, kwargs = res

    size = self.__check_input__(edge_index, size)

    # Run "fused" message and aggregation (if applicable).
    if is_sparse(edge_index) and self.fuse and not self.explain:
        coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

        msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute('update', coll_dict)
        out = self.update(out, **update_kwargs)

    else:  # Otherwise, run both functions in separation.
        if decomposed_layers > 1:
            user_args = self.__user_args__
            decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
            decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
            decomp_out = []

        for i in range(decomposed_layers):
            if decomposed_layers > 1:
                for arg in decomp_args:
                    kwargs[arg] = decomp_kwargs[arg][i]

            coll_dict = self.__collect__(self.__user_args__, edge_index,
                                             size, kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            for hook in self._message_forward_pre_hooks.values():
                res = hook(self, (msg_kwargs, ))
                if res is not None:
                    msg_kwargs = res[0] if isinstance(res, tuple) else res
            out = self.message(**msg_kwargs)
            for hook in self._message_forward_hooks.values():
                res = hook(self, (msg_kwargs, ), out)
                if res is not None:
                    out = res

            if self.explain:
                explain_msg_kwargs = self.inspector.distribute(
                        'explain_message', coll_dict)
                out = self.explain_message(out, **explain_msg_kwargs)

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            for hook in self._aggregate_forward_pre_hooks.values():
                res = hook(self, (aggr_kwargs, ))
                if res is not None:
                    aggr_kwargs = res[0] if isinstance(res, tuple) else res

            out = self.aggregate(out, **aggr_kwargs)

            for hook in self._aggregate_forward_hooks.values():
                res = hook(self, (aggr_kwargs, ), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.distribute('update', coll_dict)
            out = self.update(out, **update_kwargs)

            if decomposed_layers > 1:
                decomp_out.append(out)

        if decomposed_layers > 1:
            out = torch.cat(decomp_out, dim=-1)

    for hook in self._propagate_forward_hooks.values():
        res = hook(self, (edge_index, size, kwargs), out)
        if res is not None:
            out = res

    return out
'''

from utils import uniform

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu')) # weight initialization 

        self.conv1 = RGCNConv( # in_channels=100, out_channels=100, num_relations*=2, num_bases
            100, 100, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv( # in_channels=100, out_channels=100, num_relations*=2, num_bases
            100, 100, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        #print('1 entity, edge_index, edge_type, edge_norm: ', entity.shape, edge_index.shape, edge_type.shape, edge_norm.shape)
        ## torch.Size([n]) torch.Size([2, 30000]) torch.Size([30000]) torch.Size([30000])
        x = self.entity_embedding(entity)
        #print('2 x: ', x.shape) torch.Size[n,100]
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        #print('3 x: ', x.shape) torch.Size[n,100]
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        #print('4 x: ', x.shape) torch.Size[n,100]
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        #print('5 x: ', x.shape) torch.Size[n,100]
        
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2) # batch matrix multiplication
            # out shape = torch.Size([30000, 100]) 
            # edge_norm.view(-1,1).shape = torch.Size([30000, 1])
            ## 둘을 곱하면  torch.Size([30000, 100])
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x): # Eq 2/
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
