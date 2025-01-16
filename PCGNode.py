from node_status import node_status

class PCGNode:
    def __init__(self, name, dependencies, machine_mapping, operation=None,
                 dim=None, num_partitions=1, num_replicas=1) -> None:
        self.name = name
        self.operation = operation
        self.status = node_status.READY if not dependencies else node_status.WAITING
        self.dependencies = dependencies  # name of parent nodes
        self.machine_mapping = machine_mapping
        self.data = []
        self.dim = dim
        self.num_partitions = num_partitions
        self.num_replicas = num_replicas