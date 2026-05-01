## Ternary + int4 quantization for in-DRAM inference

### Description

Recent work [^1] has demonstrated the viability of in-DRAM computation using completely unmodified, commercial DDR4 chips. This method of computation is built using two primitives that are enabled by unconventionally driving the internals of DRAM. These two primitives are RowCopy and the Majority (MAJ) operation - specifically MAJ3. In normal operation, DRAM employs sense amplifiers to read off small amounts of charge in arrays of capacitors. The amount of charge stored in each capacitor represents the bit value stored in that capacitor. By unconventially driving the DRAM device, charge sharing across multiple capacitors can be achieved, which results in either "copying" the charge from one row of capcitors to another (RowCopy), or reading out the total charge stored across multiple rows - which translate to a majority operation.

Based on the results published in [^1], I demonstrate the viability of training a GPT-like model optimized for in-DRAM inference. In particular, low-bit integer arithmetic can be implemented efficiently. For a matrix-vector product, the matrix is stored in DRAM with one matrix row stored in one column of the memory array. Then, operations that implement the dot-product of a vector with a row in the matrix are performed on all the columns of the memory array in parallel, effectively parallelizing across the row dimension of the matrix. For a typical DDR SODIMM, the effective parallelism is 65536, and with bank-level parallelism up to 3-4 times that can be achieved.

An important implementation note is that a dot product can only be computed full in-DRAM if the matrix row can be stored in one DRAM subarray column, where a subarray is the internal unit of the memory array over which RowCopies and MAJ operations can be done. Typical subarray sizes are 512-1024, so with a few bits per element we can only compute the dot product in-DRAM in batches of 128 or 256 or so. Finally, the actual integer arithmetic is implemented bit-by-bit, so the runtime for a b1-bit integer multiplied by a b2-bit integer is O(b1 b2). Thus, lower bit is better. This is demonstrated in Figures 12 and 14 of [^1], which compare the latency and energy consumption, respectively, of matrix vector operations in DRAM. Noteably, for 2-bit matrix elements and 4-bit vector elements, the latency is far lower with DRAM than with comparable (comparisons are made to systems with equivalent memory bandwidth) CPU and GPU implementations. Furthermore, the energy consumption with in-DRAM compute is significantly lower (by almost 75% compared to the GPU). For 4-bit matrix elements and 4-bit vector elements, DRAM still matches the latency of the GPU and still beats the CPU.

Another important implementation detail is that the different bits of the matrix elements are computed as partial sums in parallal across different columns of the DRAM memory arrays. Then, the partial sums are recombined outside of the DRAM. This allows for arbitrary scaling of each of the n levels represented by an n-bit matrix element. So for example, whereas a 2-bit value could have a fixed representation as either -1 or 1, with this strategy one could assign two arbitrary floats alpha and beta to the two values.

Given this, my quantization strategy is:
- Quantize matrix elements to ternary + offset (equivalent to two arbitrary float levels) in 128-element blocks. I apply this to the query, key, value, and MLP matrices using a custom BitLinear module
- Quantize Key and Value vectors to 4 bits integers, with per-block offsets and scalings. This effectively applies the SageAttention strategy to the Key vectors.
- Use STE for training, but make it quantization-aware by applying the above quantization in the forward pass

This allows me to
- Compute FFN layers and attention entirely with 4-bit arithmetic
- Double the hidden dimension to 1024 and increase the layer to 10 because we can pack many more weights (ternary valued) in the same space.


### Future work

- To fully support in-DRAM attention computation, I would have to quantize the score values to 4-bit integers. This is currently not done, as scores are handled internally by `scaled_dot_product_attention`
- Right now, the Key and Value vectors are quantized to the regular 4-bit levels of 1, 2, 4, 8. I could expand this to arbitrary block-level float levels.



[^1]: Kubo, Tatsuya, et al. "Mvdram: Enabling gemv execution in unmodified dram for low-bit llm acceleration." arXiv preprint arXiv:2503.23817 (2025).
