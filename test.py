from training import traverse
from PCGNode import PCGNode
from PCGNodeType import PCGNodeType
from parallelizationOps import parallelizationOps

def basic_partition_and_combine_test():
    root_node = PCGNode(PCGNodeType.PARALLEL, parallelizationOps.PARTITION)
    child_node1 = PCGNode(PCGNodeType.PARALLEL, parallelizationOps.COMBINE)
    child_node2 = PCGNode(PCGNodeType.PARALLEL, parallelizationOps.COMBINE)

    pcg = {root_node: [child_node1, child_node2],
           child_node1: [], 
           child_node2: [] }

    traverse(pcg)

if __name__ == '__main__':
    basic_partition_and_combine_test()
