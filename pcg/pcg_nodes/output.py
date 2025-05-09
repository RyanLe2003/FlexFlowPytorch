from pcg.pcg_nodes.pcg_node import PCGNode
import torch.distributed as dist

class OutputNode(PCGNode):
    def __init__(self, name, parents, machine_view):
        super().__init__(name, parents)
        self.machine_view = machine_view

    def forward(self, name_to_node):        
        self.data = name_to_node[self.parents[0]].data

        # print(f"{global_rank}-{self.name}: Output Done (Forward)")