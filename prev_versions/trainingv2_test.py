import unittest
import torch
import os
from pcg_node import PCGNode
from node_types import node_types
from parallel_ops.parallel_ops import parallel_ops
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate, distribute_tensor
from algebraic_ops import algebraic_ops
from torch.distributed._tensor import DTensor
import torch.distributed as dist

from trainingv2 import execute_pcg

class TestTrainingv2Basic(unittest.TestCase):
    def setUp(self) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

        self.local_weight = torch.randn(4, 2, device="cuda", requires_grad=True)
        self.local_input = torch.tensor([[1, 1, 1, 1]], device="cuda", dtype=torch.float32)
        self.local_target = torch.tensor([[2, 2]], dtype=torch.float32)

    def test_execute_pcg(self):

        input_node = PCGNode(
            id="input",
            input=[self.local_input],
            type=node_types.INPUT,
            machine_mapping=None,
            placements=None,
            parents=[]
        )

        weight_node = PCGNode(
            id="weight",
            input=[self.local_weight],
            type=node_types.WEIGHT,
            machine_mapping=None,
            placements=None,
            parents=[]
        )

        repInput_node = PCGNode(
            id="repInput",
            input=[],
            type=parallel_ops.REPLICATE,
            machine_mapping=init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp")),
            placements= [Replicate(), Shard(1)],
            parents=["input"]
        )

        repWeight_node = PCGNode(
            id="repWeight",
            input=[],
            type=parallel_ops.REPLICATE,
            machine_mapping=init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp")),
            placements= [Shard(0), Replicate()],
            parents=["weight"]
        )

        matmul_node = PCGNode(
            id="matmul",
            input=[],
            type=algebraic_ops.MATMUL,
            machine_mapping=None,
            placements=None,
            parents=["repInput", "repWeight"]
        )

        allR_reduce_node = PCGNode(
            id="allR_Reduce",
            input=[],
            type=parallel_ops.REDUCE,
            machine_mapping=None,
            placements=None,
            parents=["matmul"]
        )

        allR_replicate_node = PCGNode(
            id="allR_Rep",
            input=[],
            type=parallel_ops.REPLICATE,
            machine_mapping=init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp")),
            placements=[Replicate(), Replicate()],
            parents=["allR_Reduce"],
        )

        output_node = PCGNode(
            id="output",
            input=[],
            type=node_types.OUTPUT,
            machine_mapping=None,
            placements=None,
            parents=["allR_Rep"]
        )

        pcg = {
            "input": input_node,
            "weight": weight_node,
            "repInput": repInput_node,
            "repWeight": repWeight_node,
            "matmul": matmul_node,
            "allR_Reduce": allR_reduce_node,
            "allR_Rep": allR_replicate_node,
            "output": output_node,
        }

        num_epochs = 10
        execute_pcg(pcg, self.local_target, num_epochs)
            
            
if __name__ == "__main__":
    unittest.main()
