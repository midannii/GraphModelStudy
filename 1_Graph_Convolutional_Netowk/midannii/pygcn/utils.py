import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str)) 
    # >>> idx_features_labels
    # array([['31336', '0', '0', ..., '0', '0', 'Neural_Networks'],
    #   ['1061127', '0', '0', ..., '0', '0', 'Rule_Learning'],
    #   ['1106406', '0', '0', ..., '0', '0', 'Reinforcement_Learning'],
    #   ...,
    #   ['1128978', '0', '0', ..., '0', '0', 'Genetic_Algorithms'],
    #   ['117328', '0', '0', ..., '0', '0', 'Case_Based'],
    #   ['24043', '0', '0', ..., '0', '0', 'Neural_Networks']],
    #  dtype='<U22')
    ### shape: (2708, 1435) 즉 (1435,)의 data가 2708개 존재함 
    ### 2708 scientific publications classified into one of seven classes. 
    ### The citation network consists of 5429 links
    
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) # shape: (2708, 1433)
    labels = encode_onehot(idx_features_labels[:, -1]) # shape: (2708, 7)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32) # shape: (2708, 7)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    # link in cora dataset 
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32) # shape: (5429, 2)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), 
                     dtype=np.int32).reshape(edges_unordered.shape) # shape: (5429, 2)
    # adjacency matrix 
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32) # shape: (2708, 2708)

    # build symmetric adjacency matrix
    ## Eq. 9에서의 assumption 
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    ## A.multiply(B): Point-wise multiplication A*B 
    
    # normalize for spectral graph convolutions (Eq. 3)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) # make A_tilda in Eq. 2
    ## sp.eye: Returns a sparse (m x n) matrix where the kth diagonal is all ones and everything else is zeros.

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
