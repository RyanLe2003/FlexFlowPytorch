import torch

class PCGNode:
    def __init__(self, name, parents):
        self.name = name
        self.parents = parents
    
    def forward(self, input_values_all):
        raise NotImplementedError

    