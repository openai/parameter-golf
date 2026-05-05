# Non-Record: Codebooks! - val_bpb 1.2067 (3-seed mean)

n.b. This is not a competitive record submission, but it was done under record conditions and hopefully will make its way into a leaderboard submission at some point!

**val bpb: 1.20667** (3-seed mean, std=0.00365)

| Seed | Steps | Pre-quant BPB | Post-quant BPB | **Sliding BPB** | Artifact |
|-|-|-|-|-|-|
| 42   | 4822  | 1.10450 | 1.2280 | **1.211** | 15863950 |
| 1024 | 4826  | 1.10397 | 1.21940 | **1.20207** | 15881168 |
| 1337 | 4866  | 1.10417 | 1.22427| **1.20694** | 15859963 |
| **Mean** | 4838 | 1.10421 | 1.22389 | **1.20667** | 15868360 |

I've been back for a day or two and have been messing about with VQ/codebook approaches; it seems like the competition is dying down a bit so I thought I'd do a little write-up since my approach is only half-working at the moment, for the benefit of anyone else interested in this line of work. Putting together a record submission at this point requires a bunch of systems/TTT stuff anyway that I don't want in this PR, even if I could get the quant gap down to something competitive.

In general, the motivation for trying codebooks is that vector quantization may be able to get us under the int6 limit for MLP/attn weights we've all been running into (although practically competitive submissions after compression are already around ~3.5 bpw) and get down to 1-3 bits while still training and optimizing in healthier datatypes. Codebooks are certainly the most powerful mode of compression, if you know what codes to use, and that's downstream of knowing more about our model's structure than Brotli/LZMA does. Unfortunately I'm not there yet - while I can get to around ~1.20 bpb in competition conditions with this setup, and I can squeeze another 2 layers in, I can't close the quant gap. I do want to work a little harder on this over the next few weeks, but I'm going to do some systems work elsewhere first because I wanna learn CuTEDSL. 

Below is a rundown of my discoveries and what I'm sticking with:

## EP8 Lattice Fixed Codebook

I took this from the [QuIP# paper](https://github.com/Cornell-RelaxML/quip-sharp/tree/main), which was one of several, together with [AQLM](https://arxiv.org/abs/2401.06118) and [VPTQ](https://arxiv.org/abs/2409.17066), that I've taken some inspiration from. In our environment there's huge upside to having a fixed codebook, since then we don't need to store the codebook and can save 1-2MB. In particular, this codebook is the most dense 8D spherical lattice we have, and so it should be great for Gaussian 8D vectors. We can block up our weights into 8D blocks and then store 16-bit indices, for a total of 2.0bpw, and add another 8-bit scale vector for 3.0 bpw in the codebook. Pushing the scale vector lower than 8-bits tends to damage things significantly.

Confidence: 8/10 likely not going to be able to use learned codebooks due to size limits

## Hadamard Transform 

This was the other part of QuIP#, essentially applying a random sign-flip+rotation to the blocked weights, which is meant to make them more isotropic and iid Gaussian. Weirdly I didn't find this worked as well as they said it would, and I think that's because the model weights are already pretty isotropic. However, it may confer some small benefit on the order of 0.002 bpb, so it stays. Confidence: 5/10

## Hessian-aware Assignment + Scales

This was definitely the best thing I did, since it took my codebooks from not working at all to mostly working; I used the GPTQ machinery already in the baseline and repurposed `collect_hessians` to produce metrics by which to select the codebook index and scales. This was dramatically better than Euclidean distance at `val_bpb`, which is understandable - as many have noted, raw MSE does not necessarily imply you've captured downstream performance, and this allows us to pick the codebook compression that is least damaging to the weight's role in the loss, similar to GPTQ. Confidence: 7/10

## Lightweight Codebook Penalties

Unfortunately, while I would really like to do QAT with this setup, and force the model to get used to passing information through the codebook, it's painfully slow - the Hadamard part is relatively fast, but materializing the codebook and doing the assignment above is very time-consuming, and there's the usual problem with VQ where it, owing to being discrete, doesn't have an obvious backwards pass and must use STE or other hacks. Since we're in such a compute-constrained regime on the record board, I have to settle for proxies to QAT, and indeed QAT hasn't worked great so far in the other record entries. I might do a non-record submission with super-long step times where I can do codebook quantization in the forward pass soon. 

For now, I simply run an approximate version of the codebook quantization every 16 steps, and then have an auxiliary L2 loss that should force weights close to their codebook counterparts, which I turn on at the end of training. This is at least intended to give the model and optimizer a heads up about quantization and to begin to prioritize. I tried some cooler ideas but they worked about as well; again I think doing full QAT would be ideal. I tried KL-distillation with the quantized model as a student, but that ate time we don't have.

## Outlier Paths

One gimme is always to provide a route around quantization for particularly difficult tensors; I had about 700kb left, so I decided to simply allow that to fallback the tensors with the worst Hessian-derived reconstruction error to int8. This earned me back a tiny bit of bpb, but nothing major. Confidence: 8/10, lameness 10/10.

## Reject Bin

Some things I tried that didn't work:

Multiple codebooks sound like an absolutely awesome idea (I love the [AQLM paper](https://arxiv.org/abs/2401.06118)), but I found them hard to optimize, particularly codebooks intended to store residual corrections. AQLM itself has some really gnarly stuff, since you're solving this joint optimization of multiple discrete objects. They also take up a lot of space. I think doing some kind of hierarchical/residual/additive codebooks thing would be cool, but I need to figure out why this codebook isn't working great first before adding another. 

Shared codebooks: One idea that sounds great is storing one codebook for MLP and one for Attn, but obviously that requires storing 2 codebooks, which wastes space. The sharing worked well, but it worked well enough to justify using a shared codebook between all tensors. 

Learning codebooks in general; again, since these are discrete clusterings, we can't really use gradient descent and so commonly people use k-means; this takes a lot of time since it's not really accelerated by GPUs, and doesn't let you optimize codebooks for our downstream goal of compression, which is what we want. Ultimately we have a choice over a) what the entries in the codebook are, b) which index to pick, and we only need to optimize one at once.

Entropy-weighted assignment: Tried various gambits to encourage the model to reuse codes when it could; this worked and decreased compressed storage size as expected, but damaged performance more. 

Mega-bitcrushed scales: as expected, going down below 2 BPW with this setup produced completely incoherent models, which makes sense, these are not bitnets. 

Voronoi auxiliary loss: I had this idea where, if I had some kind of loss that punished being on the boundary between cells of the codebook, it would encourage regularization; it kind of worked, but not as well as the simpler L2 auxiliary loss described above.

Snapping: I'm a huge proponent of doing dumb stuff first, so I tried just snapping the model to the quantized vector locations every few steps while training. This actually worked surprisingly well, better than many of the gigabrain methods I tried, but not as well as the L2 loss described above.

## Conclusion

I'm a little miffed I wasn't able to reduce the quantization gap further; the raw model before quantization with 13 layers is certainly able to be top of the leaderboard, without any TTT, so the challenge is just fitting the codebook structure to the model. As I said, I suspect that better QAT strategies may be the secret to unlocking codebooks at a competitive level. 

I am proud of how my setup in this project gives me very fine-grained control over where in the model to spend bytes and therefore entropy; the codebook allows you to know BPW ahead of time, and allocate more or less to embedding, more or less to norms vs directions, etc. I think with a better understanding (or knowledge via more regularization) of the latent space it should be possible to design a codebook for the way this particular parameter-golf family of models learns. 

As I said, I think the competition is dying down, but there is still plenty of meat on the bones of the record leaderboard; but from now on most of the gain will come from systems work to get more data out of the limited compute than ML about compression. With that in mind, I'm excited to get to work on some kernels.