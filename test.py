from training import executePCG
from PCGNode import PCGNode
from parallelizationOps import parallelizationOps
import torch

def basic_partition_and_combine_test():
    root_node = PCGNode("root", parallelizationOps.PARTITION)

    child_node = PCGNode("child1", parallelizationOps.COMBINE)
    root_node.add_child(child_node)

    child_node = PCGNode("child2", parallelizationOps.COMBINE)
    root_node.add_child(child_node)

    executePCG(root_node)

if __name__ == '__main__':
    basic_partition_and_combine_test()
