LeakyReLU(0.5) Activation

Changed relu² to LeakyReLU(0.5)² in MLP forward pass.

Results
- val_bpb: 1.4175
- Baseline val_bpb: 1.4245
- Improvement: -0.007

Change
One line change in train_gpt.py:
x = torch.nn.functional.leaky_relu(self.fc(x), negative_slope=0.5)
