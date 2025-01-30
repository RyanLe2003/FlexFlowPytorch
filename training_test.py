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
        self.tensor1 = torch.tensor([1, 2, 3])
        self.tensor2 = torch.tensor([4, 5, 6])
        self.tensor3 = torch.tensor([2, 2, 2])
    
    # @patch("training.partition_tensor")
    # @patch("training.combine_tensors")
    # @patch("training.replicate_tensor")
    # @patch("training.reduce_tensors")
    def test_execute_pcg(self):
        # mock_partition.return_value = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
        # mock_combine.return_value = torch.tensor([1, 2, 3])
        # mock_replicate.return_value = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])]
        # mock_reduce.return_value = torch.tensor([2, 4, 6])

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
            parents=[node_a],
            machine_mapping=[0, 1, 2],
            operation=parallel_ops.PARTITION,
            dim=0,
        )

        node_e = PCGNode(
            id="e",
            type=node_types.OPERATION,
            parents=[node_b],
            machine_mapping=[0, 1, 2],
            operation=parallel_ops.PARTITION,
            dim=0,
        )

        node_f = PCGNode(
            id="f",
            type=node_types.OPERATION,
            parents=[node_d, node_e],
            operation=algebraic_ops.MATMUL,
        )

        node_g = PCGNode(
            id="g",
            type=node_types.OPERATION,
            parents=[node_c],
            machine_mapping=[0, 1, 2],
            operation=parallel_ops.REPLICATE,
        )

        node_h = PCGNode(
            id="h",
            type=node_types.OPERATION,
            parents=[node_f, node_g],
            operation=algebraic_ops.MATMUL,
        )

        node_i = PCGNode(
            id="i",
            type=node_types.OPERATION,
            parents=[node_h],
            machine_mapping=[0],
            operation=parallel_ops.REDUCE,
        )

        node_j = PCGNode(
            id="j",
            type=node_types.OUTPUT,
            parents=[node_i],
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

        # self.assertEqual(node_a.status, node_status.COMPLETED)
        # self.assertEqual(node_b.status, node_status.COMPLETED)
        # self.assertEqual(node_c.status, node_status.COMPLETED)
        # self.assertEqual(node_d.status, node_status.COMPLETED)
        # self.assertEqual(node_e.status, node_status.COMPLETED)

        # args, kwargs = mock_partition.call_args
        # self.assertTrue(torch.equal(args[0], self.tensor1))
        # self.assertEqual(args[1], 0)
        # self.assertEqual(args[2], 3)

        # args, kwargs = mock_combine.call_args
        # self.assertTrue(torch.equal(args[0][0], torch.tensor([1])))
        # self.assertTrue(torch.equal(args[0][1], torch.tensor([2])))
        # self.assertTrue(torch.equal(args[0][2], torch.tensor([3])))
        # self.assertEqual(args[1], 0)

        # args, kwargs = mock_replicate.call_args
        # self.assertTrue(torch.equal(args[0], torch.tensor([1, 2, 3])))
        # self.assertEqual(args[1], 2)

        # args, kwargs = mock_reduce.call_args
        # self.assertTrue(torch.equal(args[0][0], torch.tensor([1, 2, 3])))
        # self.assertTrue(torch.equal(args[0][1], torch.tensor([1, 2, 3])))

        # self.assertTrue(torch.equal(node_e.data[0], torch.tensor([2, 4, 6])))

if __name__ == "__main__":
    unittest.main()


        



