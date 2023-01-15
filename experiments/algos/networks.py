import torch.nn as nn
import torch.nn.functional as F


# fully connected net
class FC_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 

        super().__init__()
        self.h = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.6)
        self.out = nn.Linear(hidden_size, output_size)

        # xavier normal and bias 0 init
        gain = 5.0 / 3.0 #recommended gain for tanh activations
        nn.init.xavier_normal_(self.h1.weight, gain)
        nn.init.zeros_(self.h1.bias)

        nn.init.xavier_normal_(self.h2.weight, gain)
        nn.init.zeros_(self.h2.bias)

        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.h(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=1)
