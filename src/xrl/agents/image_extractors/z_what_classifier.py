import torch
import torch.nn as nn


class ZWhatClassifier(nn.Module):
    def __init__(self, z_what_shape, nb_class, **kwargs):
        super().__init__()
        self._h5 = nn.Linear(z_what_shape, nb_class)
        self.__name__ = f"ZWhatClassifier: {z_what_shape} -> {nb_class}"

    def forward(self, objects, action=None):
        pos, z_whats = objects
        classes = self._h5(torch.tensor(z_whats).float())
        return pos, classes
