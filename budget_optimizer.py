# Budget (B) → Analytical Mapping → Architecture (L, d, D)
# No search. Closed-form approximation from scaling laws.

import math

class Model:
    def __init__(self, name, params, flops, accuracy, latency):
        self.name = name
        self.params = params          # millions
        self.flops = flops            # billions per inference
        self.accuracy = accuracy      # 0..1
        self.latency = latency        # ms

    def __repr__(self):
        return f"{self.name}(acc={self.accuracy}, params={self.params}M)"

# --- Utility Function ---
def utility(model, w_acc=1.0, w_latency=0.1):
    # maximize accuracy, penalize latency
    return model.accuracy - w_latency * math.log(1 + model.latency)

# --- Cost Function ---
def cost(model, w_params=1.0, w_flops=0.5):
    return w_params * model.params + w_flops * model.flops

# --- Genesis Optimizer ---
def optimize(models, budget):
    valid = [m for m in models if cost(m) <= budget]
    if not valid:
        return None
    return max(valid, key=utility)

# --- Pareto Frontier ---
def pareto_frontier(models):
    frontier = []
    for m in models:
        dominated = False
        for other in models:
            if (
                utility(other) >= utility(m) and
                cost(other) <= cost(m) and
                (utility(other) > utility(m) or cost(other) < cost(m))
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(m)
    return frontier

# --- Example ---
if __name__ == "__main__":
    models = [
        Model("tiny", 5, 2, 0.72, 10),
        Model("small", 15, 6, 0.80, 20),
        Model("medium", 40, 20, 0.86, 45),
        Model("large", 120, 60, 0.90, 120),
        Model("xl", 300, 150, 0.92, 300),
    ]

    BUDGET = 100  # constraint

    best = optimize(models, BUDGET)
    frontier = pareto_frontier(models)

    print("=== BEST UNDER BUDGET ===")
    print(best)

    print("\n=== PARETO FRONTIER ===")
    for m in frontier:
        print(m, " | cost=", round(cost(m), 2), " utility=", round(utility(m), 4))