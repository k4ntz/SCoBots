# Pruning Results

## Pruning of Coinrun with REINFORCE

### Integrated gradient pruning

Pruning weights for feature 0 : IG-Value 0.0 < 0.01ard: 3.69    Steps: 14       
Pruning weights for feature 2 : IG-Value 0.0 < 0.01
Pruning weights for feature 3 : IG-Value 0.00030837831325640985 < 0.01
Pruning weights for feature 4 : IG-Value 0.0 < 0.01
Pruning weights for feature 5 : IG-Value 0.0 < 0.01
Pruning weights for feature 6 : IG-Value 0.0 < 0.01
Pruning weights for feature 7 : IG-Value 0.0 < 0.01
Pruning weights for feature 8 : IG-Value 0.0 < 0.01
Pruning weights for feature 9 : IG-Value 0.0 < 0.01
Pruning weights for feature 11 : IG-Value 0.0 < 0.01
Pruning weights for feature 12 : IG-Value 0.0 < 0.01
Pruning weights for feature 13 : IG-Value 0.0 < 0.01
Pruning weights for feature 14 : IG-Value 0.0 < 0.01
Pruning weights for feature 15 : IG-Value 0.0 < 0.01
Pruning weights for feature 16 : IG-Value 0.0 < 0.01
Pruning weights for feature 17 : IG-Value 0.0 < 0.01
Pruning weights for feature 18 : IG-Value 0.0 < 0.01
Pruning weights for feature 19 : IG-Value 0.0 < 0.01
Pruning weights for feature 20 : IG-Value 0.0 < 0.01
Sparsity in h.weight: 90.48%

### Weight treshhold pruning

Pruning weights for feature 0 : tensor(0., grad_fn=<MeanBackward0>) < 0.005     
Pruning weights for feature 4 : tensor(2.5792e-05, grad_fn=<MeanBackward0>) < 0.005
Pruning weights for feature 5 : tensor(0., grad_fn=<MeanBackward0>) < 0.005
Pruning weights for feature 18 : tensor(0., grad_fn=<MeanBackward0>) < 0.005
Pruning weights for feature 20 : tensor(0., grad_fn=<MeanBackward0>) < 0.005
Sparsity in h.weight: 23.81%