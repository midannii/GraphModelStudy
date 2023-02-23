import scipy.io as scio
import numpy as np
import os
import random
def load_ft(data_dir, feature_name='GVCNN'):
    print(feature_name,flush=True)
    if feature_name =='facebook':
        
        targetdir = data_dir + '/facebook'
        files = os.listdir(targetdir)
        condition = '*.feat'

        lbls = np.zeros(4039)
        fts = np.zeros([4039,576])
        
        file_list = [file for file in files if file.endswith(".feat")]
        k=[]
        for i in file_list:
            with open(os.path.join(targetdir,str(i))) as f:
                feats = f.readlines()
                for feat in feats:
                    feat = feat.strip('\n').split()
                    lbls[int(feat[0])] = int(feat[0])
                    for j in range(1,len(feat)-1):
                        fts[int(feat[0])][int(j)] = feat[int(j)]
                    

            
        #     l = i.split('.')
        #     k.append(int(l[0]))
        # k = k.sort()
        
   
        
        
 
        return fts,lbls, idx_train, idx_test
       
       
        
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


