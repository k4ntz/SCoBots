import torch
import torch.nn as nn
import torch.nn.functional as F

# define policy network for genetic algo
class policy_net(nn.Module):
    def __init__(self, input, hidden, actions, make_hidden = True): 

        super(policy_net, self).__init__()
        # should make one hidden layer
        self.h = nn.Linear(input, hidden)
        self.groupNorm = nn.GroupNorm(4, hidden)
        self.out = nn.Linear(hidden, actions)

        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []


    def forward(self, x):
        x = self.h(x)
        x = self.groupNorm(x)
        x = F.relu(x)   
        return F.softmax(self.out(x), dim=1)