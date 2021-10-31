import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune


# own pruning class for disabling nodes from prev layer 
# based on threshold
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, pruning_feature, feature_size):
        self.threshold = pruning_feature
        self.feature_size = feature_size

    def compute_mask(self, tensor, default_mask):
        # calc avg weight of connection for each feature input
        for i in range(self.feature_size):
            avg_weight = torch.mean(tensor[i::self.feature_size])
            if abs(avg_weight) < self.threshold:
                print("Pruning weights for feature", i, ":", avg_weight, "<", self.threshold)
                tensor[i::self.feature_size] = 0
        return tensor != 0


# own pruning class for disabling nodes from prev layer 
# based on ig values and threshold
class IGPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, pruning_feature, feature_size):
        self.ig_values = pruning_feature
        self.threshold = 0.01
        self.feature_size = feature_size

    def compute_mask(self, tensor, default_mask):
        # calc avg weight of connection for each feature input
        for i in range(self.feature_size):
            ig_value = self.ig_values[i]
            if abs(ig_value) < self.threshold:
                print("Pruning weights for feature", i, ": IG-Value", ig_value, "<", self.threshold)
                tensor[i::self.feature_size] = 0
        return tensor != 0


def prune_nn(nn, pruning_method = "threshold-pr", pruning_feature = 0.01):
    # set params to prune
    parameters_to_prune = (
        (nn.h, 'weight'),
    )
    # select pruning method
    if pruning_method == "threshold-pr": 
        pruning_method = ThresholdPruning
    elif pruning_method == "ig-pr":
        pruning_method = IGPruning
    # prune
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method = pruning_method,
        pruning_feature = pruning_feature,
        feature_size = 21,
    )
    prune.remove(nn.h, 'weight')
    # print
    print("Sparsity in h.weight: {:.2f}%".format(
            100. * float(torch.sum(nn.h.weight == 0.0))
            / float(nn.h.weight.nelement())
        )
    )
    # create list which inputs should be set to zero
    pruned_input = []
    for i, param in enumerate(nn.parameters()):
        if i == 1:
            for i1 in range(param.shape[1]):
                if param[0][i1] == 0:
                    pruned_input.append(i1)
    # return pruned neural network
    return nn, pruned_input