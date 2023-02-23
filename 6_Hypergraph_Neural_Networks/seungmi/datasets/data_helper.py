import scipy.io as scio
import numpy as np
import os
import random
import torch

def load_ft(data_dir, feature_name='GVCNN'):

    if feature_name == 'FB':
        #nodes : 4039 edges : 88234 feature: max(575)
        #edge load
        adj = {}
        for i in range(4039):
            adj[i] = []

        with open(os.path.join(data_dir, 'facebook_combined.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split()
                adj[int(line[0])].append(int(line[1]))

        classes = [0, 107, 348, 414, 686, 1684, 1912, 3437, 3980] #-> [0-9]
        
        #feature load
        label = np.zeros(4039)
        feature = np.zeros([4039, 576])          

        for i, l in enumerate(classes):
            with open(os.path.join(data_dir, str(l)+'.feat')) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip('\n').split()
                    label[int(line[0])] = i
                    for k,v in enumerate(line[1:-1]):
                        feature[int(line[0])][k] = v

        # nodes = list(idx.keys())
        idx_train = random.sample(range(4039), int(4039*0.8))
        idx_test = [x for x in range(4039) if x not in idx_train]

        return feature, adj, label, idx_train, idx_test 
        
    else: 
        data = scio.loadmat(data_dir)
        lbls = data['Y'].astype(np.long)
        if lbls.min() == 1:
            lbls = lbls - 1
        idx = data['indices'].item()

        if feature_name == 'MVCNN':
            fts = data['X'][0].item().astype(np.float32)
        elif feature_name == 'GVCNN':
            fts = data['X'][1].item().astype(np.float32)
        else:
            print(f'wrong feature name{feature_name}!')
            raise IOError

        idx_train = np.where(idx == 1)[0]
        idx_test = np.where(idx == 0)[0]
        return fts, lbls, idx_train, idx_test


