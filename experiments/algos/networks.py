import torch.nn as nn
from torch import tanh
import torch.nn.functional as F


# policy net
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 

        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # xavier normal and bias 0 init
        gain = 5.0 / 3.0 #recommended gain for tanh activations
        nn.init.xavier_normal_(self.h1.weight, gain)
        nn.init.zeros_(self.h1.bias)


        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.h1(x)
        x = tanh(x)
        x = self.out(x)
        return F.softmax(x, dim=1)


# value function net
class ValueNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 

        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # xavier normal and bias 0 init
        gain = 5.0 / 3.0 #recommended gain for tanh activations
        nn.init.xavier_normal_(self.h1.weight, gain)
        nn.init.zeros_(self.h1.bias)

        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.h1(x)
        x = tanh(x)
        return self.out(x)
