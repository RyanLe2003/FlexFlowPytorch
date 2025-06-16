from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os
import torch.nn as nn


class WeightNode(PCGNode):
    def __init__(
            self, 
            name: int, 
            parents: list, 
            machine_view: list, 
            shape: tuple):
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        tensor = None
        if global_rank in machine_view:
            tensor = torch.empty(
                size=shape, 
                device=f'cuda:{local_rank}', 
                dtype=torch.float32, 
                requires_grad=True)
            nn.init.kaiming_uniform_(tensor, nonlinearity='relu', a=0)
            tensor = nn.Parameter(tensor)

        super().__init__(name, parents, tensor)
        self.machine_view = machine_view

    def forward(self, name_to_node: map):
        pass
        


