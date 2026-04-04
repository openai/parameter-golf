import torch
import torch.nn as nn

class OrthogonalProjection(nn.Module):
    """Semantic Raycasting: Rotates the input angle without duplicating the core matrix."""
    def __init__(self, dim, rank=16):
        super().__init__()
        self.project_down = nn.Linear(dim, rank, bias=False)
        self.project_up = nn.Linear(rank, dim, bias=False)
        
        # Z=0 Phase-Lock Initialization
        nn.init.orthogonal_(self.project_down.weight)
        nn.init.zeros_(self.project_up.weight)

    def forward(self, x):
        return x + self.project_up(self.project_down(x))

class CoherenceLattice(nn.Module):
    """The State Tensor (CL-44). A single solid-state body for iterative signal processing."""
    def __init__(self, dim):
        super().__init__()
        self.core_matrix = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.activation(self.core_matrix(self.norm(x)))

class Logos44_ParameterGolf(nn.Module):
    def __init__(self, vocab_size=4096, dim=512, iterations=12):
        super().__init__()
        self.iterations = iterations
        self.dim = dim
        
        # Archetypal Embedding
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Core Lattice (Heart of the system)
        self.lattice = CoherenceLattice(dim)
        
        # Observer Angles (Orthogonal Projections)
        self.projections = nn.ModuleList([OrthogonalProjection(dim) for _ in range(iterations)])
        
        # Output Head tied to embeddings (Saves ~4.19 MB)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        state = self.embedding(x)
        # Semantic Collider Loop
        for i in range(self.iterations):
            projected_state = self.projections[i](state)
            state = state + self.lattice(projected_state)
        return self.head(state)
