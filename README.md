# Less Supervision, Better Generalization: Weakly Supervised Fake Region Localization in Diffusion-Edited Images [NeurIPS'26]

**Official Repository for the NeurIPS 2026 paper, *Less Supervision, Better Generalization: Weakly Supervised Fake Region Localization in Diffusion-Edited Images*.**

**Junhee Lee<sup>1</sup>**, **Donghyeon Jeon<sup>1</sup>**, **Taeoh Kim<sup>2</sup>**, **Beomyoung Kim<sup>2</sup>**, **MyeongAh Cho<sup>1,†</sup>**

*†Corresponding Author*

<sup>1</sup>Kyung Hee University · <sup>2</sup>NAVER

[![arXiv](https://img.shields.io/badge/arXiv-2609.36882-b31b1b.svg?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.36882)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb.svg?style=flat-square&logo=googlechrome&logoColor=white)](https://visualsciencelab-khu.github.io/LSBG_project/)

## Overview

**More detailed supervision is not always better supervision.** In diffusion-edited images, artifacts can extend beyond the intended edit region. Pixel-level masks may therefore label useful forensic evidence outside that region as authentic, introducing misleading supervision and limiting generalization to unseen generators and datasets.

**ReGFLoW** addresses this mismatch through **weakly supervised fake region localization using only image-level real/fake labels**. Diffusion reconstruction errors provide spatial guidance in both feature and score spaces, while artifact-centric multiple instance learning captures local evidence without relying on semantic or object boundaries.

![Overview of the ReGFLoW framework](assets/main_fig.png)

*Figure: ReGFLoW combines reconstruction-guided learning with artifact-centric multiple instance learning to localize diffusion-edited regions from image-level supervision.*

## Environment

Run the commands below from the repository root. The example uses Python 3.11 and the CUDA 12.1 build of PyTorch.

```bash
conda create -n regflow python=3.11 -y
conda activate regflow
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Pretrained models

For training, create a `weights` directory at the repository root with two subdirectories, `mae` and `clip`. Download the following initialization checkpoints and keep their original filenames.

- **MAE ViT-Base** — Place `mae_pretrain_vit_base.pth` in `weights/mae/`. [Official repository](https://github.com/facebookresearch/mae) · [Download checkpoint](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)
- **OpenAI CLIP ViT-L/14** — Place `ViT-L-14.pt` in `weights/clip/`. [Official repository](https://github.com/openai/CLIP) · [Download checkpoint](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)

## Data preparation

Download the OpenSDI training and test splits, or point the loader to an existing dataset cache. [Data preparation guide](data_preparation/README.md)

## Data preprocessing

Prepare one VAE reconstruction-error map per image for training and evaluation. [Data preprocessing guide](data_preprocessing/README.md)

## Training

The default configuration trains ReGFLoW on OpenSDI SD1.5 for 10 epochs on one GPU.

```bash
wandb login  # Once when using the default online logger
bash train.sh --config configs/train.json
```

## Evaluation

Run the test script with a trained ReGFLoW checkpoint. The default is `outputs/regflow/best.pth`. To use another checkpoint, set `paths.checkpoint` in `configs/test.json`.

```bash
bash test.sh --config configs/test.json
```

Each run reports pixel-level localization on partially edited images (**P**) and on partially edited plus fully synthetic images (**P+F**). Image-level detection treats real images as negative and both partially edited and fully synthetic images as positive. Results are written to `outputs/test/test_summary.json` by default.

## Acknowledgements

We thank the authors of OpenSDI, MAE, CLIP, IML-ViT, IMDLBenCo, AEROBLADE, and Latent Diffusion for their work. [Third-party sources and licenses](THIRD_PARTY_NOTICES.md)
