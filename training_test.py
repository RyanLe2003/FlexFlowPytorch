import unittest
from unittest.mock import patch
import torch

from training import execute_pcg
from PCGNode import PCGNode
from parallel_ops.operation_names import operation_names
from node_status import node_status

class TestTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.tensor1 = torch.tensor([1, 2, 3])
    
    @patch("training.partition_tensor")
    @patch("training.combine_tensors")
    @patch("training.replicate_tensor")
    @patch("training.reduce_tensors")
    def test_execute_pcg(self, mock_reduce, mock_replicate, mock_combine, mock_partition):
        mock_partition.return_value = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
        mock_combine.return_value = torch.tensor([1, 2, 3])
        mock_replicate.return_value = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])]
        mock_reduce.return_value = torch.tensor([2, 4, 6])

        node_a = PCGNode(
            name="a",
            dependencies=[],
            machine_mapping=[],
        )
        node_a.data = [self.tensor1]

        node_b = PCGNode(
            name="b",
            dependencies=["a"],
            machine_mapping=[0],
            operation=operation_names.PARTITION,
            dim=0,
            num_partitions=3,
        )

        node_c = PCGNode(
            name="c",
            dependencies=["b"],
            machine_mapping=[0],
            operation=operation_names.COMBINE,
            dim=0,
        )

        node_d = PCGNode(
            name="d",
            dependencies=["c"],
            machine_mapping=[0],
            operation=operation_names.REPLICATE,
            num_replicas=2,
        )

        node_e = PCGNode(
            name="e",
            dependencies=["d"],
            machine_mapping=[0],
            operation=operation_names.REDUCE,
        )

        pcg = {
            "a": node_a,
            "b": node_b,
            "c": node_c,
            "d": node_d,
            "e": node_e,
        }

        execute_pcg(pcg)

        self.assertEqual(node_a.status, node_status.COMPLETED)
        self.assertEqual(node_b.status, node_status.COMPLETED)
        self.assertEqual(node_c.status, node_status.COMPLETED)
        self.assertEqual(node_d.status, node_status.COMPLETED)
        self.assertEqual(node_e.status, node_status.COMPLETED)

        args, kwargs = mock_partition.call_args
        self.assertTrue(torch.equal(args[0], self.tensor1))
        self.assertEqual(args[1], 0)
        self.assertEqual(args[2], 3)

        args, kwargs = mock_combine.call_args
        self.assertTrue(torch.equal(args[0][0], torch.tensor([1])))
        self.assertTrue(torch.equal(args[0][1], torch.tensor([2])))
        self.assertTrue(torch.equal(args[0][2], torch.tensor([3])))
        self.assertEqual(args[1], 0)

        args, kwargs = mock_replicate.call_args
        self.assertTrue(torch.equal(args[0], torch.tensor([1, 2, 3])))
        self.assertEqual(args[1], 2)

        args, kwargs = mock_reduce.call_args
        self.assertTrue(torch.equal(args[0][0], torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(args[0][1], torch.tensor([1, 2, 3])))

        self.assertTrue(torch.equal(node_e.data[0], torch.tensor([2, 4, 6])))

if __name__ == "__main__":
    unittest.main()


        



