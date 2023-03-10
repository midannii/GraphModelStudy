import numpy as np
import scipy.sparse as sp
import torch
import json
from networkx.readwrite import json_graph
import networkx as nx

def encode_onehot(labels): # 맨 뒤에 있는 label 처리
    classes = set(labels) # {'Reinforcement_Learning', 'Neural_Networks', 'Case_Based', 'Probabilistic_Methods', 'Theory', 'Rule_Learning', 'Genetic_Algorithms'}
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)} # label명:one-hot vector 로 딕셔너리 생성
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot # (2708, 7)


def load_data(path="../data/cora/", dataset="cora"): # Node : 2708개, Feat 개수 : 1435
    """Load citation network dataset (cora only for now)"""
    print(f'Loading {dataset} dataset...')

    idx_features_labels = np.genfromtxt(f"{path}{dataset}.content", dtype=np.dtype(str)) # (2708, 1435)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) # (2708, 1433)
    labels = encode_onehot(idx_features_labels[:, -1]) # label의 one-hot encoding 생성 (2708, 7)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32) # (2708,) citation index 파악
    idx_map = {j: i for i, j in enumerate(idx)} #  논문id : index로 구성된 딕셔너리
    edges_unordered = np.genfromtxt(f"{path}{dataset}.cites", dtype=np.int32) #(5429, 2)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), #(5429, 2), 각 노드를 index로 변환하여 범위 제한
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), # 인접행렬 생성 <2708x2708 sparse matrix of type '<class 'numpy.float32'>'	with 5429 stored elements in COOrdinate format>
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) # symmertric adjacency matrix를 만들고 indentity matrix와 더해줌

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense())) # torch.Size([2708, 1433])
    labels = torch.LongTensor(np.where(labels)[1]) # torch.Size([2708])
    adj = sparse_mx_to_torch_sparse_tensor(adj) # torch sparse tensor로 바꿔줌

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

################################################################################################ added
def load_data_ppi(path="../data/ppi/", dataset="ppi"): # Node : 56944개, class 개수 : 121
    """Load ppi network dataset"""
    print(f'Loading {dataset} dataset...')

    G = json_graph.node_link_graph(json.load(open(f"{path}{dataset}-G.json")))

    labels = json.load(open(f"{path}{dataset}-class_map.json")) # length : 56944
    labels = sorted({int(i):l for i, l in labels.items()}.items())

    feats = np.load(f"{path}{dataset}-feats.npy") # (56944, 50)
    features = sp.csr_matrix(feats, dtype=np.float32) # sparse matrix (56944, 50)

    adj = nx.adjacency_matrix(G, dtype=np.float32) # <56944x56944 sparse matrix of type '<class 'numpy.float32'>'	with 1612348 stored elements in Compressed Sparse Row format>

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) # symmertric adjacency matrix를 만들고 indentity matrix와 더해줌

    idx_train = [n for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']] # 0~44905
    idx_val = [n for n in G.nodes() if G.nodes[n]['val']] # 44906 ~ 51419
    idx_test = [n for n in G.nodes() if G.nodes[n]['test']] #51420 ~ 56943

    features = torch.FloatTensor(np.array(features.todense())) # torch.Size([56944, 50])
    labels = torch.LongTensor(np.where(labels)[1]) # torch.Size([56944])
    adj = sparse_mx_to_torch_sparse_tensor(adj) # torch sparse tensor로 바꿔줌

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
################################################################################################

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
