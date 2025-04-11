from pcg_node import PCGNode
import torch
import torch.nn as nn

class ReLUNode(PCGNode):
    def __init__(self, name, parents):
        super().__init__(name, parents)

    def forward(self, input_values_all):
        m = nn.ReLU()
        return tuple([m(input_values_all[self.parents[0]])])