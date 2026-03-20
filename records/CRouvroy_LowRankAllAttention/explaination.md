(Persistent Memory)
We replace the feed-forward NN from the trasnformer blocks.
This is based on the paper "Augmenting Self-attention with Persistent Memory".

(Quantization)
We forced a quantization into INT8 for tensors with size $> 16,384$ (others are kept into FP16).

(LoRA Routing)

We factorize matrices in low-rank ($W = W_d W_u$), because for $r <<d$, $d \times d >> d \times r + r \times d$.