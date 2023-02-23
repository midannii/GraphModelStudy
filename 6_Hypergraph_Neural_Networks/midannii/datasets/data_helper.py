import scipy.io as scio
import numpy as np


def load_ft(data_dir, feature_name='GVCNN'):
    data = scio.loadmat(data_dir)
    lbls = np.array(data['Y'], dtype=int)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx = data['indices'].item()

    if feature_name == 'MVCNN':
        fts = np.array(data['X'][0].item(), dtype=float)
    elif feature_name == 'GVCNN':
        fts = np.array(data['X'][1].item(), dtype=float)
    else:
        print(f'wrong feature name{feature_name}!')
        raise IOError

    idx_train = np.where(idx == 1)[0]
    idx_test = np.where(idx == 0)[0]
    return fts, lbls, idx_train, idx_test

