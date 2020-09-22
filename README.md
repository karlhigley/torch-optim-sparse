# torch-sparse-optim

This library implements "sparser" versions of PyTorch optimizers, which only apply momentum and weight decay updates to parameters where the gradients are non-zero.

It contains four optimizers:
- SparserSGD
- SparserAdam
- SparserSGDW
- SparserAdamW

The latter two follow the approaches outlined in ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) by Loshchilov & Hunter from ICLR 2019.

Except for SGDW, they're all straightforward ports of the existing optimizers from PyTorch, modified only to convert momentum and weight decay to sparse updates. The SGDW optimizer additionally applies a small change to where/how weight decay is applied, as outlined in the paper above.
