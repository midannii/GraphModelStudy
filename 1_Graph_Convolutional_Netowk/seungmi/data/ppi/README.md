# Protein-Protein Interactions (PPI) dataset

Links to dataset used in this git 
- Protein-Protein Interactions [Source](http://thebiogrid.org/download.php) [Preprocessed](http://snap.stanford.edu/graphsage/ppi.zip)

Detail of dataset : [GraphSAGE github](https://github.com/williamleif/GraphSAGE)

Protein-Protein Interaction dataset for inductive node classification

A toy Protein-Protein Interaction network dataset. The dataset contains 24 graphs. The average number of nodes per graph is 2372. Each node has 50 features and 121 labels. 20 graphs for training, 2 for validation and 2 for testing.

- G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
- id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
- class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
- feats.npy [optional] -- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
- walks.txt [optional] -- A text file specifying random walk co-occurrences (one pair per line) (*only for unsupervised version of graphsage)