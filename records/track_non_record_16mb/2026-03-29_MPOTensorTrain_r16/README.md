# Matrix Product Operator Tensor Train Decomposition (r=16)

Replaced the standard dense linear feedforward matrices within the MLP block with a low-rank Matrix Product Operator (MPO). The dense tensor operations were factored into two entangled cores initialized at rank $r=16$. 

This structural substitution reduced the parameter footprint to 12.6M parameters while maintaining gradient descent convergence over the 10-billion token distribution. Executed natively on an A100 for 20,000 iterations to map the exact cross-entropy decay curve under high compression constraints.