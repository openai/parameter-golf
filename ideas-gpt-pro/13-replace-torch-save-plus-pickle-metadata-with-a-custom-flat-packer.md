# 13. Replace `torch.save` Plus Pickle Metadata With a Custom Flat Packer

## Category

Serialization and compression changes

## Why

`torch.save` wastes bytes on names, Python object structure, and general-purpose serialization. A flat binary format with short tensor IDs, compact shape metadata, and concatenated payload or scale blobs should save real space.

That recovered space is better spent on a better model than on serialization overhead. After that, the next step would be grouped 6-bit or 4-bit quantization on the biggest matrices, but only after the fake-quant tail.

## Tradeoffs

- Speed: load time should still be trivial at these sizes
- Size: better, potentially much better if the model is pushed upward
- Complexity/risk: moderate-high

## Repo Fit

This is more enabling infrastructure than a direct bpb gain, but it fits the export path.
