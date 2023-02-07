"""MLP Definitions"""
from torch import nn
from torch.nn  import Tanh
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Policy MLP"""
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size (int): input_layer size
            hidden_size (int): hidden_layer size
            output_size (int): output size
        """
        super().__init__()
        self.hlayer1 = nn.Linear(input_size, hidden_size)
        self.tanh = Tanh()
        self.out = nn.Linear(hidden_size, output_size)

        # xavier normal and bias 0 init
        gain = 5.0 / 3.0 #recommended gain for tanh activations
        nn.init.xavier_normal_(self.h1.weight, gain)
        nn.init.zeros_(self.h1.bias)

        nn.init.xavier_normal_(self.out.weight)
        self.out.weight.data /= 100
        nn.init.zeros_(self.out.bias)


    def forward(self, t_in):
        """forward pass"""
        hidden = self.hlayer1(t_in)
        tanh = self.tanh(hidden)
        lin_out = self.out(tanh)
        return F.softmax(lin_out, dim=1)


class ValueNet(nn.Module):
    """Value MLP"""
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size (int): input_layer size
            hidden_size (int): hidden_layer size
            output_size (int): output size
        """
        super().__init__()
        self.hlayer1 = nn.Linear(input_size, hidden_size)
        self.tanh = Tanh()
        self.out = nn.Linear(hidden_size, output_size)

        # xavier normal and bias 0 init
        gain = 5.0 / 3.0 #recommended gain for tanh activations
        nn.init.xavier_normal_(self.h1.weight, gain)
        nn.init.zeros_(self.h1.bias)

        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, t_in):
        """forward pass"""
        hidden = self.hlayer1(t_in)
        tanh = self.tanh(hidden)
        return self.out(tanh)
