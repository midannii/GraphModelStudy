from utils import *

'''
A toy Protein-Protein Interaction network dataset. 
The dataset contains 24 graphs. The average number of nodes per graph is 2372. 
Each node has 50 features and 121 labels. 
20 graphs for training, 2 for validation and 2 for testing.
'''

def load_ppi_data(dataset='PPI'):
    print('Loading {} dataset...'.format(dataset))
    train_data = PPIDataset(mode='train') # 20개의 그래프에 총 44906개의 node 
    valid_data = PPIDataset(mode='valid') # 2개의 그래프에 총 6514개의 node 
    test_data = PPIDataset(mode='test') # 2개의 그래프에 총 5524개의 node 
    ## total dataset 개수 = 44906+6514+5524 = 56944
    
    # make feature & label from DGL graph 
    features, labels = [],[]
    for d in train_data: 
        features.append(d.ndata['feat'])
        #labels.append(d.ndata['label'])
        for node in d.nodes(): labels.append(node)
    l_train = 44906
    for d in valid_data: 
        features.append(d.ndata['feat'])
        #labels.append(d.ndata['label'])
        for node in d.nodes(): labels.append(node)
    l_valid = 6514
    for d in test_data: 
        features.append(d.ndata['feat'])
        #labels.append(d.ndata['label'])
        for node in d.nodes(): labels.append(node)
    l_test = 5524
    features = torch.cat(features, 0) # shape: (56944, 50)
    #labels = torch.cat(labels, 0) # shape: (56944, 121)
    labels = torch.tensor(labels)
    print('## labels: ', labels.shape)
    
    # link in dataset : train 1,271,274 + valid 205,434 + test 167,500 = 1,644,208
    # adjacency matrix 
    edges = []
    for d in train_data: 
        edges1, edges2 = d.edges()
        for j in range(len(edges1)): edges.append([int(edges1[j]), int(edges2[j])]) # 1,271,274개
    for d in valid_data: 
        edges1, edges2 = d.edges()
        for j in range(len(edges1)): edges.append([int(edges1[j]), int(edges2[j])]) # 205,434개 
    for d in test_data: 
        edges1, edges2 = d.edges()
        for j in range(len(edges1)): edges.append([int(edges1[j]), int(edges2[j])]) # 167,500개
    edges = np.array(edges) # shape: (1644208, 2)
    ### node의 종류는 (0,3480)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32) # shape: (56944, 56944)

    # build symmetric adjacency matrix
    ## Eq. 9에서의 assumption 
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    ## A.multiply(B): Point-wise multiplication A*B 
    
    # normalize for spectral graph convolutions (Eq. 3)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) # make A_tilda in Eq. 2
    ## sp.eye: Returns a sparse (m x n) matrix where the kth diagonal is all ones and everything else is zeros.

    idx_train = range(l_train)
    idx_val = range(l_train, l_valid)
    idx_test = range(l_valid, l_test)

    features = torch.FloatTensor(np.array(features))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train) 
    idx_val = torch.LongTensor(idx_val) 
    idx_test = torch.LongTensor(idx_test) 

    return adj, features, labels, idx_train, idx_val, idx_test
