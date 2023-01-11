import torch.nn as nn
import torch.nn.functional as F


# fully connected net with group norm
class FC_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 

        super().__init__()
        self.h = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.6)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.h(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=1)
