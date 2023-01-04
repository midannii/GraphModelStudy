import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels): # 맨 뒤에 있는 label 처리
    classes = set(labels) # {'Reinforcement_Learning', 'Neural_Networks', 'Case_Based', 'Probabilistic_Methods', 'Theory', 'Rule_Learning', 'Genetic_Algorithms'}
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)} # label명:one-hot vector 로 딕셔너리 생성
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot # (2708, 7)


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print(f'Loading {dataset} dataset...')

    idx_features_labels = np.genfromtxt(f"{path}{dataset}.content", dtype=np.dtype(str)) # (2708, 1435)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) # (2708, 1433)
    labels = encode_onehot(idx_features_labels[:, -1]) # label의 one-hot encoding 생성

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32) # (2708,) citation index 파악
    idx_map = {j: i for i, j in enumerate(idx)} #  논문id : index로 구성된 딕셔너리
    edges_unordered = np.genfromtxt(f"{path}{dataset}.cites", dtype=np.int32) #(5429, 2)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), #(5429, 2), 각 노드를 index로 변환하여 범위 제한
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), # 인접행렬 생성 (2708, 2708)
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix -> directed adjacency matrix를 undirected로 바꿔줌
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) # symmertric adjacency matrix를 만들고 indentity matrix와 더해줌 (Ã = A + I_N)

    idx_train = range(2000) # label은 2708개인데 1500까지만 나눈 이유,,,?
    idx_val = range(2000, 2200)
    idx_test = range(2200, 2700)

    features = torch.FloatTensor(np.array(features.todense())) # torch.Size([2708, 1433]), todense는 CSR형태의 sparse matrix를 ndarray로 바꿔줌
    labels = torch.LongTensor(np.where(labels)[1]) # torch.Size([2708])
    adj = sparse_mx_to_torch_sparse_tensor(adj) # torch sparse tensor로 바꿔줌

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # (2708, 1)
    r_inv = np.power(rowsum, -1).flatten() # (2708, ) 원본 논문에선 (-1/2)를 썼는데 이렇게 해도 성능변화가 거의 없다고함
    r_inv[np.isinf(r_inv)] = 0. #diagonal 생성
    r_mat_inv = sp.diags(r_inv) # sparse matrix 생성 (2708 X 2708)
    mx = r_mat_inv.dot(mx) # DA
    return mx

''' from tensorflow-GCN. 여기서는 정확히 D^(-1/2)AD^(1/2)로 구현함
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
'''

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
