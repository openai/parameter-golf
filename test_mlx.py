import mlx.core as mx
logits = mx.array([[1.0, 5.0, 3.0, 2.0], [0.1, 0.5, 0.9, 0.2]])
k = 2
indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
print(indices)
