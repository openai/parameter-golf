import mlx.core as mx

def fn(x, mask):
    # dynamic slice
    sub_x = x[mask]
    return mx.sum(sub_x)

c_fn = mx.compile(fn)
x = mx.random.normal((10, 10))
mask = x > 0
print(c_fn(x, mask))
