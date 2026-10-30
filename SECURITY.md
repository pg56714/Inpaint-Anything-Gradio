# Security

## Known issues

Last checked: 2026-10-30.

These dependencies still have open advisories because other packages cap their versions. Upgrade them once the cap is lifted.

| Package | Version | Fixed in | Held back by | Impact and mitigation |
|---|---|---|---|---|
| pillow | 12.3.0 | 13.0.0 | gradio (`pillow<13`) | Crafted image files (for example TIFF or PDF) can crash the process or read out-of-bounds memory when decoded. Run the Gradio app locally only (no `share=True`, no public port) and open only images you trust. |

## Model checkpoints

The LaMa checkpoint (`big-lama/models/best.ckpt`) stores PyTorch Lightning callback state that `torch.load(weights_only=True)` cannot read, so `lama/saicinpainting/training/trainers/__init__.py` loads it with `weights_only=False`. Loading a pickle can run arbitrary code: only use the official big-lama checkpoint linked in the README.

## GPU

PyTorch is installed from the CUDA 12.6 wheel index (torch 2.14.1+cu126), which needs an NVIDIA driver that supports CUDA 12.6 or newer. Inference was verified on a local GPU with the real model weights.
