Original Github URL : https://github.com/tkipf/pygcn

Graph Convolutional Networks in PyTorch
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

![Graph Convolutional Networks](img/figure.png)

### Semi-Supervised Node Classification

This implementation makes use of the Cora dataset from [2].

- GCN
  Spectral Graph Convolutions & Layer-Wise Linear Model
  Using node feature as signal, transformed signal to Fourier domain
  But for expensive computation cost, approximated by a truncated expansion in term of Chebyshev polynomials $T_k(x)$  

  $$ g_\theta \star x \approx \sum_{k=0}^K \theta'_kT_k(\tilde L)x $$  

  Set $k = 1$ and $\lambda_{max} \approx 2$for layer-wise convolution operation and using single parameter.  

  $$ g_\theta \star x \approx \theta (I_N + D^{-1/2}AD^{-1/2})x $$  

  For avoiding exploding/vanishing gradients, use renormalization trick  

  $$ Z = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X\Theta$$  


- model
In paper, they use only two-layer GCN for semi-supervised node classification on graph

$$ Z = f(X, A) = softmax(\hat A ReLU(\hat AXW^{(0)})W^{(1)}) $$

- Annotation
  1) $X$ : input data $[N, C]$
  2) $\hat{A}$ : symmetric adjacency matrix $[N, N]$
  3) $W^{(l)}$ : l-th layer weight  
  4) Z : $[N, F]$

  In Cora dataset, 
  $N$ = 2708. $C$ = 1433
  $F$ = 7

  In PPI dataset
  $N$ = 56944 $C$ = 50
  $F$ = 3769

  > Test set results: loss= 0.7660 accuracy= 0.8040  

- Weight
1) first-layer : input-to-hidden weight matrix
  $W^{(0)}$  $[C, H]$
2) second-layer : hidden-to-output weight matrix
  $W^{(1)}$ $[H, F]$
- Loss : cross-entropy error  

$$\mathcal{L} = - \sum_{l\in{\mathcal{Y}_L}} {\sum_{f=1}^F {Y_{lf}lnZ_{lf}}}$$  

## Requirements

  * PyTorch
  * Python

## Usage

```python train.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
