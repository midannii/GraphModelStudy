import numpy as np
import pandas as pd
import torch
import utils as u
import os

"""
#edges
#nodes
#nodes_feats : 128
#num_classes : 10
#num_nodes : 236894
#feature_per_node : 128
#max_time : 1.000000
#min_time : 0.000000
#time_steps : 27
#num_non_existing

load_edges
prepare_node_feats --> deepwalk

"""
class DBLP_Dataset():
    def __init__(self, args):
        assert args.task in ['link_pred', 'node_cls'], 'dblp only implements link_pred'
        args.dblp_args = u.Namespace(args.dblp_args)
        # self.nodes_labels_times = self.load_node_labels(args.dblp_args)
        self.edges = self.load_edges(args.dblp_args)
        self.nodes_feats = self.load_node_feats(args.dblp_args)
        
    def load_node_feats(self, dblp_args):
        data = np.load(os.path.join(dblp_args.folder, dblp_args.feats_file))
        nodes = data[0]

        nodes_feats = nodes

        self.num_nodes = len(nodes)
        self.feats_per_node = nodes.shape[-1]

        return nodes_feats
    
    def prepare_node_feats(self,node_feats):
        node_feats = node_feats[0]
        return node_feats

    def time_reindexing(self, time):
        index = {}
        time = list(sorted(set(time)))

        for i, t in enumerate(time):
            index[t] = i

        return index

    def load_edges(self, dblp_args):
        with open(os.path.join(dblp_args.folder, dblp_args.edges_file), 'r') as f:
            lines = f.readlines()
            data = []

            for line in lines:
                edge = line.strip().split()
                for n, i in enumerate(edge):
                    if n < 2:
                        i = int(i)
                        edge[n] = torch.tensor(i, dtype=torch.long)
                    else:
                        i = float(i)
                        edge[n] = torch.tensor(i)

                data.append(edge)

            time = [i[2] for i in data]
            index = self.time_reindexing(time)
            for i, edge in enumerate(data):
                data[i][2] = index[edge[2]]

            tcols = u.Namespace({'source': 0,
                                'target': 1,
                                'time': 2})
            
            data = torch.tensor(data, dtype = torch.long)

            data = data[:,[tcols.target, tcols.source, tcols.time]]
            
            self.max_time = data[:,tcols.time].max()
            self.min_time = data[:,tcols.time].min()

            # idx : source, target, time     vals : weigth
            return {'idx': data, 'vals': torch.ones(data.size(0))}
