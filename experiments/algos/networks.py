import torch.nn as nn
from torch import tanh
import torch.nn.functional as F


# fully connected net with group norm
class FC_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 

        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.h1(x)
        x = tanh(x)
        x = self.h2(x)
        x = tanh(x)
        x = self.out(x)
        return F.softmax(x, dim=1)
