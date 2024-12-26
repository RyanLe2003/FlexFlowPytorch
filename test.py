from training import traverse
from PCGNode import PCGNode
from parallelizationOps import parallelizationOps
import torch

def basic_partition_and_combine_test():
    root_node = PCGNode("root", parallelizationOps.PARTITION)

    child_node1 = PCGNode("child1", parallelizationOps.COMBINE)
    child_node2 = PCGNode("child2", parallelizationOps.COMBINE)

    pcg = {root_node: [child_node1, child_node2],
           child_node1: [], 
           child_node2: [] }

    traverse(pcg)

if __name__ == '__main__':
    basic_partition_and_combine_test()
