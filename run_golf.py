import torch
import torch.optim as optim
from core.lattice import Logos44_ParameterGolf
from core.tokenizer import ArchetypalTokenizer
import time

def load_nucleation_data(tokenizer):
    with open("nucleation/supersaturated.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    # Create simple (input, target) pairs for autoregressive training
    x = torch.tensor(tokens[:-1]).unsqueeze(0)
    y = torch.tensor(tokens[1:]).unsqueeze(0)
    return x, y

def trigger_nucleation():
    print("[Z=0] Initiating Coherence Engineering Nucleation Sequence...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ArchetypalTokenizer()
    model = Logos44_ParameterGolf(vocab_size=4096, dim=512, iterations=12).to(device)
    
    x, y = load_nucleation_data(tokenizer)
    x, y = x.to(device), y.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 10 Minutes strict limit for Parameter Golf
    start_time = time.time()
    time_limit = 10 * 60 
    
    print("[Z=0] Supersaturated solution injected. Firing 8xH100 loop (simulated).")
    iteration = 0
    
    while time.time() - start_time < time_limit:
        optimizer.zero_grad()
        output = model(x)
        
        # Reshape for CrossEntropy: (batch*seq_len, vocab_size)
        loss = criterion(output.view(-1, 4096), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            elapsed = time.time() - start_time
            print(f"T+{elapsed:.1f}s | Iteration {iteration} | Impedance (Loss): {loss.item():.4f}")
            
            # If impedance drops near zero, phase lock is achieved early
            if loss.item() < 0.01:
                print("[Z=0] Absolute Phase Lock Achieved. Halting training.")
                break
                
        iteration += 1

    print("[Z=0] Nucleation complete. Model crystallized.")
    torch.save(model.state_dict(), "logos44_z0_weights.pth")

if __name__ == "__main__":
    trigger_nucleation()
