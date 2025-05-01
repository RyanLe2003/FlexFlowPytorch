import unittest
from unittest.mock import patch
import torch

from training import execute_pcg
from pcg.pcg_node import PCGNode
from parallel_ops.parallel_ops import parallel_ops
from node_types import node_types
from algebraic_ops import algebraic_ops

class TestTraining1(unittest.TestCase):
    def test_execute_pcg(self):
        input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        weight_tensor = torch.nn.Parameter(torch.tensor([[2],[0]], dtype=torch.float32, requires_grad=True).to('cuda'))
        
        # Create nodes for the computation graph
        input_node = PCGNode(id="input", type=node_types.INPUT, parents=[], data=[input_tensor])
        weight_node = PCGNode(id="weight", type=node_types.WEIGHT, parents=[], data=[weight_tensor])

        # Partition input tensor across 2 machines
        partition_node = PCGNode(
            id="partition_input",
            type=node_types.OPERATION,
            parents=["input"],
            operation=parallel_ops.PARTITION,
            machine_mapping=[0, 1],
            dim=0
        )

        # Replicate weight tensor across 2 machines
        replicate_node = PCGNode(
            id="replicate_weight",
            type=node_types.OPERATION,
            parents=["weight"],
            operation=parallel_ops.REPLICATE,
            machine_mapping=[0, 1]
        )

        # Perform distributed matrix multiplication
        matmul_node = PCGNode(
            id="matmul",
            type=node_types.OPERATION,
            parents=["partition_input", "replicate_weight"],
            operation=algebraic_ops.MATMUL
        )

        # Reduce the results from both machines
        reduce_node = PCGNode(
            id="reduce_output",
            type=node_types.OPERATION,
            parents=["matmul"],
            operation=parallel_ops.REDUCE,
            machine_mapping=[0, 1]
        )

        output_node = PCGNode(
            id="output",
            type=node_types.OUTPUT,
            parents=["reduce_output"],
        )

        pcg = {
            "input": input_node,
            "weight": weight_node,
            "partition_input": partition_node,
            "replicate_weight": replicate_node,
            "matmul": matmul_node,
            "reduce_output": reduce_node,
            "output": output_node
        }

        # execute_pcg(pcg)

class TestTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.tensor1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)  
        self.tensor2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, requires_grad=True)
        self.tensor3 = torch.tensor([[1], [2]], dtype=torch.float32, requires_grad=True)

        # self.tensor1 = torch.randn(2, 2, requires_grad=True)
        # self.tensor2 = torch.randn(2, 2, requires_grad=True)
        # self.tensor3 = torch.randn(2, 2, requires_grad=True)

    def test_execute_pcg(self):

        node_a = PCGNode(
            id="a",
            type=node_types.INPUT,
            parents=[],
            data=[self.tensor1],
        )

        node_b = PCGNode(
            id="b",
            type=node_types.WEIGHT,
            parents=[],
            data=[self.tensor2],
        )

        node_c = PCGNode(
            id="c",
            type=node_types.WEIGHT,
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

class TestMLPPCG(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 4  # Number of samples in a batch
        self.input_channels = 4 #8  # Input size
        self.hidden_dim = 4  # Hidden layer dimension
        self.output_channels = 4 #2  # Output size

        # Initialize input tensor and weights
        self.input_tensor = torch.randn(self.batch_size, self.input_channels)
        self.weight1 = torch.randn(self.input_channels, self.hidden_dim)  # Weight for first layer
        self.weight2 = torch.randn(self.hidden_dim, self.output_channels)  # Weight for second layer

    def test_mlp_pcg(self):
        # Input node
        node_input = PCGNode(
            id="input",
            type=node_types.INPUT,
            parents=[],
            data=[self.input_tensor],
        )

        # Weight nodes (parameters to be updated during training)
        node_weight1 = PCGNode(
            id="weight1",
            type=node_types.WEIGHT,
            parents=[],
            data=[self.weight1],
        )
        node_weight2 = PCGNode(
            id="weight2",
            type=node_types.WEIGHT,
            parents=[],
            data=[self.weight2],
        )

        partition_a = PCGNode(
            id="parta",
            type=node_types.OPERATION,
            parents=["input"],
            operation=parallel_ops.PARTITION,
            machine_mapping=[0,1],
            dim=0
        )

        replicate_a = PCGNode(
            id="repa",
            type=node_types.OPERATION,
            parents=["weight1"],
            operation=parallel_ops.REPLICATE,
            machine_mapping=[0, 1]
        )

        replicate_b = PCGNode(
            id="repb",
            type=node_types.OPERATION,
            parents=["weight2"],
            operation=parallel_ops.REPLICATE,
            machine_mapping=[0, 1]
        )


        # First MatMul operation: Input × Weight1
        node_matmul1 = PCGNode(
            id="matmul1",
            type=node_types.OPERATION,
            parents=["parta", "repa"],
            operation=algebraic_ops.MATMUL,
        )

        # First ReLU activation
        node_relu1 = PCGNode(
            id="relu1",
            type=node_types.OPERATION,
            parents=["matmul1"],
            operation=algebraic_ops.RELU,
        )

        # Second MatMul operation: Hidden Representation × Weight2
        node_matmul2 = PCGNode(
            id="matmul2",
            type=node_types.OPERATION,
            parents=["relu1", "repb"],
            operation=algebraic_ops.MATMUL,
        )

        # Second ReLU activation (optional, depending on architecture)
        node_relu2 = PCGNode(
            id="relu2",
            type=node_types.OPERATION,
            parents=["matmul2"],
            operation=algebraic_ops.RELU,
        )

        # Output node
        node_output = PCGNode(
            id="output",
            type=node_types.OUTPUT,
            parents=["relu2"],
        )

        pcg = {
            "input": node_input,
            "weight1": node_weight1,
            "weight2": node_weight2,
            "matmul1": node_matmul1,
            "relu1": node_relu1,
            "matmul2": node_matmul2,
            "relu2": node_relu2,
            "output": node_output,
            "parta": partition_a,
            "repa": replicate_a,
            "repb": replicate_b
        }

        # execute_pcg(pcg)

if __name__ == "__main__":
    unittest.main()


        



