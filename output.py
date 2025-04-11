from pcg_node import PCGNode
import torch

class OutputNode(PCGNode):
    def __init__(self, name, parents):
        super().__init__(name, parents)

    def forward(self, input_values_all):
        return tuple([input_values_all[self.parents[0]]])