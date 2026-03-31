cat <<EOF > records/samuel-m3pro-12layer/README.md
# Model Craft: Parameter Golf Submission (M3 Pro 12-Layer)

### Submission Details
- **Author:** Samuel Wangai (@swangai7178)
- **Hardware:** Apple MacBook Pro M3 Pro (12-Core CPU, 18-Core GPU)
- **Final BPB:** 1.9105
- **Compressed Size:** 7.5MB
- **Training Time:** 587.8s (Under 600s limit)

### Architecture & Strategy
- **Layer Depth:** 12 layers with a model dimension of 384.
- **Optimizer:** Muon for rapid convergence on Apple Silicon.
- **Thermals:** Forced fans to 4200+ RPM to maintain performance and prevent thermal throttling.
- **Memory Management:** Used manual cache clearing to handle the 12-layer depth on 18GB Unified Memory.
EOF