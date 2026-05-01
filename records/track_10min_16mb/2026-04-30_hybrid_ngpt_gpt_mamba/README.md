# Hybrid nGPT / GPT / Mamba Submission

3-seed mean val_bpb = 1.173

This submission is a hybrid model mixing nGPT transformer layers, standard GPT layers, and Mamba2 layers.

I have the following layers:

```python
[nT, M, M, T, M, M, T, M, T, nT]
```

where:

- `nT`  stands for modified nGPT layer
- `T`  stands for GPT transformer layer
- `M`  stands for Mamba2 layer

## Main architecture details

### Modified nGPT layers

The nGPT layers are not exactly the same as in the original nGPT paper.

The main differences are:

- I normalize only the Q and K matrices after every optimizer step. For this short training setup, this worked better than normalizing more all matrices.
- The attention block and MLP block both receive the input directly, i.e. MLP gets the input not the result of the attention
- The final nGPT layer output is:

```python
h_att + h_mlp - h
```

where `h` is the layer input, and `h_att`, `h_mlp` are the normalized outputs of the attention and MLP parts.

Putting nGPT layers at the beginning and end of the model helpes a lot. The model became much more stable numerically, and I could move some tensors to bf16 without hurting the loss.

### Mamba2 layers

The Mamba2 layers are standard Mamba2 layers.

Parameters:

```text
d_state  = 128
d_conv   = 4
expand   = 2
head_dim = 64
```

I originally wanted to use Mamba to make 8k or 16k context and train the model using fp8. However, I could not get significant speed-up with fp8 with this model.

I could not get `torch.compile` to work on the Mamba layers without graph breaks. There is a non-contiguous tensor inside the Mamba path. Making it contiguous  makes the model slower. Because of this, I compiled layers separately or only compiled contiguous transformer blocks.
I could not find simple solution to this problem. I did not have enought time to study this problem deeper.

### GPT-style transformer layers

The `T` layers are standard transformer-style layers, with one small structural choice:

- The MLP gets the input of the layer directly, not after attention.
- The T layer output is:

```python
h_att(h) + h_mlp(h)
```

where `h_att` and `h_mlp` are the outputs of the attention and MLP blocks.

The MLP width is:

```python
(3.5) * emb_dim
```

The activation is:

```python
LeakyReLU(x) ** 2
```

I tried to fuse the MLP manually and also looked at existing kernels. But for this embedding size and these tensor shapes, `torch.compile` in PyTorch 2.11 already did a very good job, at least in my setup.

## Context length

The model uses:

```text
max_context = 8192
```

The original plan was to mix transformer and Mamba layers, use 8k or 16k context, and then speed things up with fp8. In practice, the fp8 speedup was not enough for this setup, so I dropped the idea.

## Optimizer

I use  of AdamW and Muon.

Weight decay:

```text
weight_decay = 0.75
```

## Evaluation

I use sliding-window evaluation.

## Rule compliance

This submission follows the competition rules:

- No TTT.
- No tokenizer changes.
- No data processing changes.
- No validation data used during training.
- Training is under the 600 second limit.
- Evaluation is under the 600 second limit.
- Final artifact is under 16,000,000 bytes.

## Notes

This is probably not the best possible version of this idea.

I think the model could get lower loss by making it narrower and adding another layer. Recursive layers, like in some other submissions, would  help. I did not have enough time to test all these combinations.

I think this hybrid construction is promising. The nGPT layers at the start and end are especially useful for stability.

One thing: the second run may be faster than the first one because `torch.compile` cache is already saved. Delete cache for fair timing.
