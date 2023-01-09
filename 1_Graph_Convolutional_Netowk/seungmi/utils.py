import numpy as np
import scipy.sparse as sp
import torch
import json

from sklearn.preprocessing import StandardScaler

#node class encoding by one-hot
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# Load Dataset (Coar)
"""
    csr_matrix / coo matrix : sparse matrix
    idx_features_labels ; [0 : idx] [1:-1 : features] [-1 : labels(str)]

    dataset only about undirected
    -> symmetric adjacency matrix + identity matrix(self loop for considering self feature)

    dataset : coar
    citation network dataset 
    edge : citation link <- undirected 
    node : document
    documnet has class label

    variable
        features : node feature vector
        labels : node class label
        idx : node index
        adj : adjacency matrix
"""
# Load Dataset (PPI)
'''
PPI dataset
    G.json : graph dataset -> train / test / valid
            target - source
    class_map.json : dictionary mapping the graph node ids to classes
    feats.npy : node features / ordering given by id_map.sjon

    variable
        features : node feature vector
        labels : node class label
        idx : node index
        adj : adjacency matrix
'''
def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if dataset == "cora":
        # load node data
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # load edge information and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # nomalize featrue and adjacency matrix(+ identity matrix : self loop)
        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        # split dataset by node idx
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    elif dataset == "ppi" :
        G_data = json.load(open(path + dataset + "-G.json"))
        G_nodes = G_data['nodes']
        G_links = G_data['links']

        idx_train, idx_test, idx_val = [], [], []
        tx = []
        row, col = [], []

        length = len(G_nodes) 

        for i in G_nodes:
            #split test/val/train dataset by idx node
            if i['test'] == False and i['val'] == False:
                idx_train.append(i['id'])
            elif i['test'] == True and i['val'] == False:
                idx_test.append(i['id'])
            elif i['test'] == False and i['val'] == True:
                idx_val.append(i['id'])
            else:
                tx.append(i['id'])

        if len(tx) != 0:
            idx_train += tx

        for i in G_links:
            row.append(i['source'])
            col.append(i['target'])

        adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(length, length))

        feats = np.load(path + dataset + "-feats.npy")
        features = sp.csr_matrix(feats, dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # nomalize featrue and adjacency matrix(+ identity matrix : self loop)
        # features = normalize(features)
        # adj = normalize(adj + sp.eye(adj.shape[0]))

        scaler = StandardScaler(with_mean=False)
        scaler.fit(features)
        features = scaler.transform(features)
        scaler = StandardScaler(with_mean=False)
        scaler.fit(adj)
        adj = scaler.transform(adj + sp.eye(adj.shape[0]))

        labels = []
        label = json.load(open(path + dataset+ "-class_map.json"))
        for i in range(length):
            labels.append(label[str(i)])

        unique = []
        label = []
        for i in labels:
            check = True
            for j in unique:
                if i == j:
                    check = False
                    label.append(unique.index(i))
            if check:
                unique.append(i)
                label.append(unique.index(i))

        labels = encode_onehot(label)

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
