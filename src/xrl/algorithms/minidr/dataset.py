import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    def __init__(self, history, history_size):
        self.h = history #history is passed as list and updated outside
        self.history_size = history_size

    def __len__(self):
        return self.history_size

    def __getitem__(self, idx):
        idx = idx % len(self.h) #do not exceed history length
        episode = self.h[idx]

        idx_sample = np.random.randint(0, (len(episode)-1)//4) #sample random part of episode

        # one entry is last state, action, state and reward as seperate entries
        last_states = episode[idx_sample * 4]
        actions= episode[idx_sample * 4 + 1]
        states = episode[idx_sample * 4 + 2]
        rewards= episode[idx_sample * 4 + 3]

        # flatten raw features list
        last_states = [[0,0] if x==None else x for x in last_states]
        last_states = np.array(np.array(last_states).tolist()).flatten()
        # convert to tensor
        last_states = torch.tensor(last_states).float()
        actions = torch.tensor(actions)
        states = torch.tensor(np.array(states.tolist()).flatten()).float()
        rewards = torch.tensor(rewards)

        return last_states, actions, states, rewards