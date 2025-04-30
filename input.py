from pcg_node import PCGNode
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
            tensor = torch.tensor(data, dtype=torch.float32, device=f'cuda:{local_rank}', requires_grad=True)
        
        super().__init__(name, parents, tensor)
        self.machine_view = machine_view

    def forward(self, name_to_node):
        pass


        



