import torch.nn as nn
import torch.nn.functional as F


# fully connected net with group norm
class FC_Normed_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 

        super().__init__()
        self.h = nn.Linear(input_size, hidden_size)
        self.groupNorm = nn.GroupNorm(4, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.h(x)
        x = self.groupNorm(x)
        x = F.relu(x)   
        return F.softmax(self.out(x), dim=1)
