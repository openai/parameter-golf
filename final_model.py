import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Prepare text
texts = []
for i in range(200):
    t = dataset["train"][i]["text"]
    if t.strip() != "":
        texts.append(t)

full_text = " ".join(texts)

# Load tokenizer (prebuilt)
tokenizer = ByteLevelBPETokenizer(
    "my_tokenizer-vocab.json",
    "my_tokenizer-merges.txt"
)

encoded = tokenizer.encode(full_text)
input_ids = torch.tensor([encoded.ids])

inputs = input_ids[:, :-1]
targets = input_ids[:, 1:]

vocab_size = tokenizer.get_vocab_size()

# Model
model = nn.Sequential(
    nn.Embedding(vocab_size, 16),
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, vocab_size)
)

# Training setup
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Train
for epoch in range(30):
    logits = model(inputs)
    logits = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    loss = loss_fn(logits, targets_flat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Final output
print("Final Loss:", loss.item())

torch.save(model.state_dict(), "model_weights.pt")