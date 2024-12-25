class PCGNode:
    def __init__(self, operation, parallel_strategy=None) -> None:
        self.operation = operation
        self.parallel_strategy = parallel_strategy
        self.children = []
        self.parents = []
    
    def add_child(self, child):
        self.children.append(child)
        child.parents.append(self)
