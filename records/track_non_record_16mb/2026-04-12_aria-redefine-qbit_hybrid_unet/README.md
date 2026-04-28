Hybrid Recurrent U-Net: aria-redefine-qbit Baseline
This submission implements a Depth-Recurrent U-Net Transformer designed to maximize representational capacity within the 16.7MB parameter constraint of the OpenAI Parameter Golf challenge.

Key Results
Val BPB: 1.4016 (at 1000 steps)

Serialized Size: 16.53 MB (int8 + zlib)

MLP Scaling: 1.9x

Optimizer: Muon with Newton-Schulz (k=5)


Architectural Innovations1. Hybrid Decoder RecurrenceTo decouple sequential reasoning depth from the parameter budget, this model utilizes a weight-shared looping mechanism in the decoder stage.The Logic: The Encoder layers remain unique to extract initial features. The Decoder layers then iteratively refine these features by re-processing the encoder's skip connections over multiple RECURRENT_LOOPS.The Benefit: This turns a 12-layer physical stack into a virtually deeper network without increasing the .ptz file size.2. U-Net Skip ConnectionsThe architecture uses a "U" shape where early encoder outputs are cached and fused back into corresponding decoder layers via weighted skip connections:$$x_{out} = x_{in} + w_{skip} \cdot x_{encoder}$$


This preserves high-frequency token information that is often lost in deep, narrow models.

3. Entropy-Aware Compression
The model uses a calculated MLP_MULT of 1.9x. This was empirically determined to be the "Goldilocks" zone—providing enough parameters for the Muon optimizer to find a stable minima while staying safely under the 16.7MB limit after int8 quantization and zlib compression.

Execution Strategy
Optimization
We utilize the Muon optimizer for the transformer blocks. To maximize throughput on H100 hardware, the Newton-Schulz orthogonalization kernels are wrapped in @torch.compile.

H100 Scaling Plan
The code is parameterized via environment variables:

RECURRENT_LOOPS: Can be scaled from 1 to 5 depending on the wall-clock throughput during the 600-second sprint.

MLP_MULT: Hardcoded at 1.9 to guarantee 16MB compliance.

How to Reproduce
Ensure torch, numpy, and zlib are installed.

Run the training script:

Bash
python train_gpt.py
The script defaults to RECURRENT_LOOPS=1 to reproduce the 1.4016 BPB baseline reported in submission.json.
