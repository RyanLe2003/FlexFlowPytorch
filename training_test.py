import unittest
from unittest.mock import patch
import torch

from training import execute_pcg
from PCGNode import PCGNode
from parallel_ops.parallel_ops import parallel_ops
from node_status import node_status
from node_types import node_types
from algebraic_ops import algebraic_ops

class TestTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.tensor1 = torch.tensor([[1, 2], [3, 4]])  
        self.tensor2 = torch.tensor([[5, 6], [7, 8]])  
        self.tensor3 = torch.tensor([[2, 0], [0, 2]])

    def test_execute_pcg(self):

        node_a = PCGNode(
            id="a",
            type=node_types.INPUT,
            parents=[],
            data=[self.tensor1],
        )

        node_b = PCGNode(
            id="b",
            type=node_types.INPUT,
            parents=[],
            data=[self.tensor2],
        )

        node_c = PCGNode(
            id="c",
            type=node_types.OUTPUT,
            parents=[],
            machine_mapping=[],
            data=[self.tensor3],
        )

        node_d = PCGNode(
            id="d",
            type=node_types.OPERATION,
            parents=["a"],
            machine_mapping=[0, 1],
            operation=parallel_ops.PARTITION,
            dim=1,
        )

        node_e = PCGNode(
            id="e",
            type=node_types.OPERATION,
            parents=["b"],
            machine_mapping=[0, 1],
            operation=parallel_ops.PARTITION,
            dim=0,
        )

        node_f = PCGNode(
            id="f",
            type=node_types.OPERATION,
            parents=["d", "e"],
            operation=algebraic_ops.MATMUL,
        )

        node_g = PCGNode(
            id="g",
            type=node_types.OPERATION,
            parents=["c"],
            machine_mapping=[0, 1],
            operation=parallel_ops.REPLICATE,
        )

        node_h = PCGNode(
            id="h",
            type=node_types.OPERATION,
            parents=["f", "g"],
            operation=algebraic_ops.MATMUL,
        )

        node_i = PCGNode(
            id="i",
            type=node_types.OPERATION,
            parents=["h"],
            machine_mapping=[0],
            operation=parallel_ops.REDUCE,
        )

        node_j = PCGNode(
            id="j",
            type=node_types.OUTPUT,
            parents=["i"],
        )

        pcg = {
            "a": node_a,
            "b": node_b,
            "c": node_c,
            "d": node_d,
            "e": node_e,
            "f": node_f,
            "g": node_g,
            "h": node_h,
            "i": node_i,
            "j": node_j,
        }

        execute_pcg(pcg)

if __name__ == "__main__":
    unittest.main()


        



