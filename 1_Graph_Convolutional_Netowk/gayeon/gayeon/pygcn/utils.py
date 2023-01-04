import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)} #identity 행렬을 만들어서 라벨별로 한줄씩 인코딩함
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str)) #genfromtxt로 읽어옴, content: word attributes + class_label
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) #행렬 부분(word attributes) 희소행렬 압축 저장
    labels = encode_onehot(idx_features_labels[:, -1]) # 텍스트로 되어있는 라벨 가져와서 onehot encoding

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32) #paper_id에 해당하는 부분
    idx_map = {j: i for i, j in enumerate(idx)} #순서대로 paper_id가 안되어 있어서 매핑해줌
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32) #35	1033
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), #flatten(): 1차원으로 평탄화해줌, [cited, cite]한 논문 순으로 idx mapping
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), #coo_matrix: 원소의 좌표 + data// interaction 개수(edges.shape[0])만큼 1을 만들고, cited, cite한 idx 줘서 adj 행렬만듦
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #adj.T와 adj.T>adj를 곱하는 거

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

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
    """Row-normalize sparse matrix""" #열의 합의 역수로 다 나눠줘서 정규화 진행
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten() # rowsum 배열 각각의 원소에 대해 역수
    r_inv[np.isinf(r_inv)] = 0. #0 역수 무한이니까 0이면 0 넣음
    r_mat_inv = sp.diags(r_inv) #대각행렬 만들기
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx): #sparse matrix를 만들고 torch sparse tensor로 바꿔줘야 사용할 수 있음
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32) #coo matrix
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) #row, col 순서대로 저장
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
