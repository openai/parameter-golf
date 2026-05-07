import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLM(nn.Module):

    def __init__(self, vocab_size=8000, hidden=256):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        return self.head(x)


def train():

    model = TinyLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for step in range(50):

        x = torch.randint(0, 8000, (8, 32))
        y = torch.randint(0, 8000, (8, 32))

        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, 8000),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print("loss:", loss.item())


if __name__ == "__main__":
    train()
