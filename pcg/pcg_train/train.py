import torch
import torch.nn as nn

def train(order, name_to_node, target, params, output_node):
    # forward pass
    for node in order:
        name_to_node[node].forward(name_to_node)
    
    prediction = output_node.data
    if prediction is None:
        return
    
    # backward pass
    loss_fn = nn.MSELoss()
    loss = loss_fn(prediction, target)
    loss.backward()

    if params:
        optimizer = torch.optim.SGD(params, lr=0.01)
        optimizer.step()
        optimizer.zero_grad()