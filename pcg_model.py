import torch.nn as nn

from concurrent.futures import ThreadPoolExecutor
import logging

import time
import threading

logging.basicConfig(level=logging.DEBUG)
class PCGModel(nn.Module):
    def __init__(self, pcg, weights, dependency_graph):
        super().__init__()
        self.pcg = pcg
        self.weights = weights
        self.dependency_graph = dependency_graph
        self.remaining_parents = {}
        self.ready_nodes = []
    
    
    def exec_forward(self, node, values):
        thread_id = threading.get_ident()
        start_time = time.time()
        logging.info(f"START: {node} (Thread {thread_id}) at {start_time:.4f}")

        res = self.pcg[node].forward(values)

        if len(res) > 1:
            for i in range(len(res)):
                name = f"{node}_{i}"
                values[name] = res[i]
        else:
            values[node] = res[0]

        for n, parents in self.dependency_graph.items():
            if node in parents:
                self.remaining_parents[n] -= 1
                if self.remaining_parents[n] == 0:
                    self.ready_nodes.append(n)
        
        self.ready_nodes.remove(node)

        end_time = time.time()
        logging.info(f"END: {node} (Thread {thread_id}) at {end_time:.4f}, Duration: {end_time - start_time:.4f}s")
    
    def forward(self, input_tensor):
        for name, parents in self.dependency_graph.items():  # this creates issues for input/weight dependencies
            self.remaining_parents[name] = len(parents)

        values = {'input': input_tensor}
        for key, param in self.weights.items():
            values[key] = param

        for node_name, num_parents in self.remaining_parents.items():
            if num_parents == 0:
                self.ready_nodes.append(node_name)

        with ThreadPoolExecutor() as executor:
            while self.ready_nodes:
                futures = []
                for node in self.ready_nodes:
                    futures.append(executor.submit(self.exec_forward, node, values))
                
                for future in futures:
                    future.result() # block here until finishes
            
        return values['output']




                
            





