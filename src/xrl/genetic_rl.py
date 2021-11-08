# DONT DELETE OR MOVE THIS FILE!!
# CANT MOVE THIS FILE BECAUSE OF SERIALIZATION PROBLEMS WITH PRETRAINED MODELS:
# https://github.com/pytorch/pytorch/issues/18325
# TODO: Move to algorithm-Folder and retrain on dgx2
# TODO: Change saving all models to saving state dict to avoid further problems

import torch
import torch.nn as nn
import torch.nn.functional as F

# define policy network for genetic algo
class policy_net(nn.Module):
    def __init__(self, input, hidden, actions, make_hidden = True): 
        super(policy_net, self).__init__()
        # should make one hidden layer
        self.make_hidden = make_hidden

        if self.make_hidden:
            self.h = nn.Linear(input, hidden)
            self.out = nn.Linear(hidden, actions)
        else:
            self.out = nn.Linear(input, actions)


    def forward(self, x):
        if self.make_hidden:
            x = F.relu(self.h(x))
        return F.softmax(self.out(x), dim=1)