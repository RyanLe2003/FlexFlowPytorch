from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

# this will eventually need to be added when I get pcg
# would likely need to convert json to list to tensor
class InputNode(PCGNode):
    def __init__(self, name, parents, machine_view, data):
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        tensor = None
        if global_rank in machine_view:
            tensor = data
        
        super().__init__(name, parents, tensor)
        self.machine_view = machine_view

    def forward(self, name_to_node):
        pass


        



