from pcg_node import PCGNode
import torch

class MatmulNode(PCGNode):
    def __init__(self, name, parents):
        super().__init__(name, parents)

    def forward(self, input_values_all):
        return tuple([torch.matmul(input_values_all[self.parents[0]], input_values_all[self.parents[1]])])