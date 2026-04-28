import torch
import torch.nn as nn
from torch.optim import AdamW

def test_optimizer_momentum():
    model = nn.Linear(10, 10)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Simulate a step to populate momentum
    x = torch.randn(1, 10)
    y = model(x)
    y.sum().backward()
    optimizer.step()
    
    # Save momentum buffers
    exp_avg_before = {p: optimizer.state[p]['exp_avg'].clone() for p in model.parameters()}
    
    # Untie: add a new parameter
    new_param = nn.Parameter(torch.randn(10, 10))
    optimizer.add_param_group({'params': [new_param], 'lr': 1e-3})
    
    # Check if old parameters still have their momentum
    for p in model.parameters():
        assert torch.equal(optimizer.state[p]['exp_avg'], exp_avg_before[p]), "Momentum wiped!"
    
    print("Optimizer Momentum Preservation: PASS")

if __name__ == "__main__":
    test_optimizer_momentum()
