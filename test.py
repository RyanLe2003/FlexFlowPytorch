from operations import partition_tensor
from operations import combine_tensors
from operations import replicate_tensor
from operations import reduce_tensors
import torch
import unittest

class TestOperations(unittest.TestCase):

    def test_partition_tensor(self):
        tensor = torch.arange(16).reshape(8, 2)
        partitions = partition_tensor(tensor, 0, 4)

        expected_values = [
            tensor[:2],  
            tensor[2:4],  
            tensor[4:6],
            tensor[6:], 
        ]

        for partition, expected in zip(partitions, expected_values):
            self.assertTrue(torch.equal(partition, expected))
    
    def test_combine_tensors(self):
        tensor1 = torch.tensor([[1, 2]])
        tensor2 = torch.tensor([[3, 4]])
        result = combine_tensors([tensor1, tensor2], 0)
        expected = torch.tensor([[1, 2], [3, 4]])

        self.assertTrue(torch.equal(result, expected))
    
    def test_replicate_tensor(self):
        tensor = torch.arange(16).reshape(8, 2)

        desired_copies = 4
        result = replicate_tensor(tensor, desired_copies)

        self.assertTrue(len(result) == desired_copies)
        
        for copy in result:
            self.assertTrue(torch.equal(tensor, copy))
    
    def test_reduce_tensor(self):
        tensor1 = torch.tensor([[1, 2]])
        tensor2 = torch.tensor([[3, 4]])

        expected = torch.tensor([[4, 6]])
        result = reduce_tensors([tensor1, tensor2])
        
        self.assertTrue(torch.equal(expected, result))

if __name__ == '__main__':
    unittest.main()