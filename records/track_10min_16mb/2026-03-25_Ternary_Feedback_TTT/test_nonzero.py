import mlx.core as mx
mask = mx.array([True, False, True])
# Can we get indices?
inds = mx.nonzero(mask)
x = mx.array([10.0, 20.0, 30.0])
print(x[inds])
