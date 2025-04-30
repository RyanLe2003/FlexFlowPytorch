import torch

class PCGNode:
    def __init__(self, name, parents, data=None):
        self.name = name
        self.parents = parents
        self.data = data
    
    def forward(self):
        raise NotImplementedError

    