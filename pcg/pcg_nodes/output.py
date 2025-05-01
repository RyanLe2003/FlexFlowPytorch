from pcg.pcg_nodes.pcg_node import PCGNode
import torch.distributed as dist

class OutputNode(PCGNode):
    def __init__(self, name, parents, machine_view):
        super().__init__(name, parents)
        self.machine_view = machine_view

    def forward(self, name_to_node):
        global_rank = dist.get_rank()

        # hack: we are actually concerned with the previous node's parents
        parent = name_to_node[self.parents[0]].parents[0]
        pparent = name_to_node[parent]

        if global_rank not in pparent.machine_view:
            return
        
        self.data = name_to_node[self.parents[0]].data