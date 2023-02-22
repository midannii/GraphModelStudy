import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

from dhg import Graph, Hypergraph
from dhg.models import HGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):
    net.eval()
    outs = net(X, G)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res

#################################################################### preprocessing 코드
"""
This is cora dataset:
  ->  num_classes
  ->  num_vertices
  ->  num_edges
  ->  dim_features
  ->  features
  ->  edge_list
  ->  labels
  ->  train_mask
  ->  val_mask
  ->  test_mask
"""
# 위와 같은 구조로 바꿔줘야 함

def preprocess_facebook(path):
    ego_nodes = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    data = {}
    data['num_classes'] = 10
    data['num_vertices'] = 4039
    data['num_edges'] = 88234
    data['labels'] = torch.tensor([0 for i in range(4039)])
    data['train_mask'] = 0
    data['val_mask'] = 0
    data['test_mask'] = 0

    ################################################################ data['labels'] 생성
    for i, nodeID in enumerate(ego_nodes):
        with open(os.path.join(path, str(nodeID) +".edges"), "r") as f:
            lines = f.readlines()
            for line in lines:
                temp = line.split()
                edge = []
                data['labels'][int(temp[0])] = i
                data['labels'][int(temp[1])] = i

    ################################################################ data['edge_list'] 생성
    with open(os.path.join(path, "facebook_combined.txt"), "r") as f:
        lines = f.readlines()
        edgelist = []
        for line in lines:
            temp = line.split()
            edge = []
            edge.append(int(temp[0]))
            edge.append(int(temp[1]))
            edge= tuple(edge)
            edgelist.append(edge)
        data['edge_list'] = edgelist

    ################################################################ data['dim_features'] 생성
    with open(os.path.join(path, "0.egofeat"), "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split(' ')
            data['dim_features'] = len(temp)
    
    ################################################################ data['features'] 생성
    data['features'] = [[] for _ in range(data['num_vertices'])]
    for nodeID in ego_nodes:
        with open(os.path.join(path, str(nodeID) +".feat"), "r") as f:
            lines = f.readlines()
            for line in lines:
                temp = line.split()
                cur_ID = int(temp[0])
                for index, cent_ID in enumerate(temp):
                    if index != 0:
                        data['features'][cur_ID-1].append(int(cent_ID))

    data['features'] = torch.tensor(np.array(data['features']))

    

if __name__ == "__main__":
    set_seed(2022)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    path = "/home/tomma112/HGNN/facebook"
    data = preprocess_facebook(path)
    X, lbl = data["features"], data["labels"]
    G = Graph(data["num_vertices"], data["edge_list"])
    HG = Hypergraph.from_graph_kHop(G, k=1)
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    net = HGNN(data["dim_features"], 16, data["num_classes"])
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    HG = HG.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, HG, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, HG, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, HG, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)