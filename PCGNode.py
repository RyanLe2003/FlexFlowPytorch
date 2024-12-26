class PCGNode:
    def __init__(self, operation, parallel_strategy=None) -> None:
        self.operation = operation
        self.parallel_strategy = parallel_strategy
        self.inputs = []
        self.outputs = []