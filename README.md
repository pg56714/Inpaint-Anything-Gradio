# Inpaint-Anything-Gradio

> **Security:** see [SECURITY.md](SECURITY.md) for known dependency issues.

## Getting Started

[HuggingFace Space](https://huggingface.co/spaces/pg56714/Inpaint-Anything)

![example1](/example/main.jpg)

## Installation

```bash
# Create and activate a python 3.12 environment.
conda create -n inpaint-anything python=3.12 -y

conda activate inpaint-anything

pip install -r requirements.txt
```

### Weights

Download the weights and save them in the `pretrained_models` directory.

- [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- [big-lama](https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip) (unzip it to `pretrained_models/big-lama`)

## Running the Project

```bash
python app.py
```

## Source

[Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything)

## Citation

```
@article{yu2023inpaint,
  title={Inpaint Anything: Segment Anything Meets Image Inpainting},
  author={Yu, Tao and Feng, Runseng and Feng, Ruoyu and Liu, Jinming and Jin, Xin and Zeng, Wenjun and Chen, Zhibo},
  journal={arXiv preprint arXiv:2304.06790},
  year={2023}
}
```
