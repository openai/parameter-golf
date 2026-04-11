# Record: 11L HWNODE (Continuous Matrix ODE) + Cubric N-gram (val_bpb=0.5527)

This submission improves upon the prior `0.5644` SOTA baseline from PR #800 by replacing MLPs with a novel architecture called Hammerstein-Wiener Neural ODE (HWNODE). 

The theory behind this architechure is that while you are so limited for size, layers matter more than width. Thus if you can generate multiple less rich layers, you can outperform a single layer of the same size. Secondarily, if models are able to behave well quantized, they must have at least enough representatonal capacity to match the performance originarily. I also used spectral normalization to both enforce stability and to subtley encourage orthagonality which results directly in more unique generated layers. This submission proves that this works.

### Architecture Highlights
- **Hammerstein-Wiener Neural ODE (HWNODE):** 

This is a novel SSM that uses a Linear neural ODE wrapped in two non-linearities a la Hammerstein-Wiener models from control theory to dynamically generate new layers from one set of weights. This is done by taking the taylor series approximation for the function f(Lode(G(x))) which is differentiable via the chain rule, and generating theoretically N layers (though of diminishing scale since the factorial denominator grows quickly). Practically, we set it to 2 new layers, and we wrap each term in relu so that it is non-linear. This outperms the MLPs marginally, though there is much room for improvement and scope expansion.


### Evaluation and Metrics
- **Artifact Size:** 15.74 MB (`16507081 bytes`)
- **Step Count:** Completed almost 6,000 steps inside the 600-second wallclock.
- **Final Target:** `val_loss = 0.933` / `val_bpb = 0.5527`

### Run Command

```bash
export HWNODE_STATE_DIM=864
export HWNODE_ORDER=2
export TRAIN_SEQ_LEN=1024
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Reproducibility

Tested deterministically on 8xH100 SXM clusters. Reaches ~6200 steps within the 600s wallclock limit. HWNODE dynamically limits parameter counts to 31M, scaling perfectly into the INT5/INT6 quantization matrix to cleanly compress beneath the strict 16MB threshold. Base seed 1337 officially scores `0.5527` BPB natively after Drift-free TTT and Cubric N-gram processing.