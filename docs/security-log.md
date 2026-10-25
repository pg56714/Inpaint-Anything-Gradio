# Security Log

English | [繁體中文](security-log.zh-TW.md)

- Date: 2026-10-25
- Branch: `fix/security-upgrades`
- Scope: `requirements.txt`, `lama/`

## Summary

imgaug (no fix available) is no longer loaded or installed for inference, kornia was unpinned so it works with current torch, and gradio now requires a patched release. pillow still has open advisories because gradio caps its version.

## Changes

| Item | Before | After | Reason |
|---|---|---|---|
| gradio | unpinned | >=6.16 | File read, SSRF and path traversal advisories (fixed in 6.16) |
| albumentations | ==0.5.2 | removed | Pulls in imgaug (insecure deserialization, no fixed release) |
| kornia | ==0.5.0 | unpinned | 0.5.0 does not support current torch |
| `lama/saicinpainting/training/data/datasets.py` | - | lazy import | albumentations / imgaug are imported only inside `get_transforms()`, which only training uses |
| `lama/.../countless/requirements.txt` | Pillow>=6.2.0 | Pillow>=10.3.0 | Older Pillow releases have known advisories |

## What you need to know

- Training LaMa needs albumentations 0.5.2 (and therefore imgaug) installed manually; inference no longer loads them.
- `lama/requirements.txt` and `lama/docker/` are upstream training files and were left unchanged; they still pin old versions.

## Remaining issues

### Dependency versions held back upstream

| Package | Version | Advisories | Why it is not fixed | Impact and mitigation |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 (fixed in 13.0.0) | gradio >= 6.16 requires `pillow<13`; no gradio release allows Pillow 13 yet. | Crafted image files (for example TIFF or PDF) can crash the process or read out-of-bounds memory when decoded. Run the Gradio app locally only (no `share=True`, no public port) and open only images you trust. Upgrade once gradio allows Pillow 13. |

## How it was verified

- osv-scanner on the resolved `requirements.txt`; only pillow remains.
- torch 2.14.1 (CPU), kornia 0.9.0 and pytorch-lightning 2.6.6: `lama_inpaint` imports without albumentations, and the kornia refinement ops run.
- The Gradio UI builds and serves with the SAM, LaMa and Stable Diffusion models stubbed out.

## Not verified

- Inference on a GPU with the real model weights (the test environment has no GPU and no weights).
- LaMa training.
