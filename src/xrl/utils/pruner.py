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


# own pruning class for disabling nodes from prev layer 
# based on avg weight value
class AvgPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, pruning_feature, feature_size):
        self.threshold = pruning_feature
        self.feature_size = feature_size

    def compute_mask(self, tensor, default_mask):
        # calc avg weight of connection for each feature input
        avg_weight = torch.mean(tensor)
        nt = self.threshold * avg_weight
        for i in range(tensor.size()[0]):
            if abs(tensor[i]) < nt:
                print("Pruning weight nr.", i, ": abs(", tensor[i], ")<", nt)
                tensor[i] = 0
        return tensor != 0


# main functin to call
def prune_nn(nn, pruning_method_s = "threshold-pr", pruning_feature = 0.01, feature_size = 21):
    # set params to prune
    parameters_to_prune = (
        (nn.h, 'weight'),
    )
    # select pruning method
    pruning_method = None
    if pruning_method_s == "threshold-pr": 
        pruning_method = ThresholdPruning
    elif pruning_method_s == "ig-pr":
        pruning_method = IGPruning
    elif pruning_method_s == "avg-pr":
        pruning_method = AvgPruning
    # prune
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method = pruning_method,
        pruning_feature = pruning_feature,
        feature_size = feature_size,
    )
    prune.remove(nn.h, 'weight')
    # print
    print("Sparsity in h.weight: {:.2f}%".format(
            100. * float(torch.sum(nn.h.weight == 0.0))
            / float(nn.h.weight.nelement())
        )
    )
    # prune final layer
    if pruning_method_s == "avg-pr":
        parameters_to_prune = (
            (nn.out, 'weight'),
        )
        prune.global_unstructured(
           parameters_to_prune,
           pruning_method = pruning_method,
           pruning_feature = pruning_feature,
           feature_size = feature_size,
        )
        prune.remove(nn.out, 'weight')
        print("Sparsity in out.weight: {:.2f}%".format(
            100. * float(torch.sum(nn.out.weight == 0.0))
            / float(nn.out.weight.nelement())
        )
    )
    print(pruning_method)
    # create list which inputs should be set to zero
    pruned_input = []
    for i, param in enumerate(nn.parameters()):
        if i == 1:
            for i1 in range(param.shape[1]):
                if param[0][i1] == 0:
                    pruned_input.append(i1)
    # return pruned neural network
    return nn, pruned_input