"""MLP Definitions"""
from torch import nn
from torch.nn  import Tanh, ReLU
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Policy MLP"""
    def __init__(self, input_size, hidden_size, output_size, act_f="relu"):
        """
        Args:
            input_size (int): input_layer size
            hidden_size (int): hidden_layer size
            output_size (int): output size
        """
        super().__init__()
        self.hlayer1 = nn.Linear(input_size, hidden_size)
        self.hlayer2 = nn.Linear(hidden_size, hidden_size)
        if act_f == "tanh":
            self.act1 = Tanh()
            self.act2 = Tanh()
            gain = 5.0 / 3.0 #recommended gain for tanh activations
        else:
            self.act1 = ReLU()
            self.act2 = ReLU()
            gain = 1.41421356237 #sqrt(2) gain for tanh activations
        self.out = nn.Linear(hidden_size, output_size)

        # xavier normal and bias 0 init
        nn.init.xavier_normal_(self.hlayer1.weight, gain)
        nn.init.zeros_(self.hlayer1.bias)
        nn.init.xavier_normal_(self.hlayer2.weight, gain)
        nn.init.zeros_(self.hlayer2.bias)

        nn.init.xavier_normal_(self.out.weight)
        self.out.weight.data /= 100
        nn.init.zeros_(self.out.bias)


    def forward(self, t_in):
        """forward pass"""
        hidden1 = self.hlayer1(t_in)
        act1 = self.act1(hidden1)
        hidden2 = self.hlayer2(act1)
        act2 = self.act2(hidden2)
        lin_out = self.out(act2)
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
        nn.init.xavier_normal_(self.hlayer1.weight, gain)
        nn.init.zeros_(self.hlayer1.bias)

        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, t_in):
        """forward pass"""
        hidden = self.hlayer1(t_in)
        tanh = self.tanh(hidden)
        return self.out(tanh)
