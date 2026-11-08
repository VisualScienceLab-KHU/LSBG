# Pretrained Models

ReGFLoW uses two pretrained visual encoders to initialize its feature extraction network. Their roles are complementary:

- **MAE ViT-Base** provides the initialization for the localization-oriented ViT backbone that learns spatial, patch-level representations for fake-region localization.
- **OpenAI CLIP ViT-L/14** provides complementary multi-layer visual representations with semantic context. ReGFLoW keeps the CLIP visual encoder frozen and fuses its features with the MAE-based localization branch.

Together with the diffusion reconstruction-error maps described in the [data preprocessing guide](../data_preprocessing/README.md), these representations support reconstruction-guided feature learning and artifact-centric localization.

## Download checkpoints

The `weights/` directory is included in the LSBG repository for these instructions. Create the `mae/` and `clip/` subdirectories inside it, then download the following **official pretrained checkpoints** while keeping their filenames unchanged:

- **MAE ViT-Base:** Download `mae_pretrain_vit_base.pth` from the [official checkpoint](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and place it in `weights/mae/`. [MAE paper (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) · [Official repository](https://github.com/facebookresearch/mae).
- **OpenAI CLIP ViT-L/14:** Download `ViT-L-14.pt` from the [official checkpoint](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) and place it in `weights/clip/`. [CLIP paper (ICML 2021)](https://proceedings.mlr.press/v139/radford21a.html) · [Official repository](https://github.com/openai/CLIP).

Use the **OpenAI CLIP ViT-L/14** checkpoint, not an OpenCLIP variant or the ViT-L/14@336px checkpoint. These files initialize the encoders; they are **not** trained ReGFLoW checkpoints.

## Configure paths

The default locations are already specified in [`configs/base.json`](../configs/base.json):

| Field | Default | Description |
| --- | --- | --- |
| `paths.mae_checkpoint` | `./weights/mae/mae_pretrain_vit_base.pth` | Path to the MAE checkpoint file. |
| `paths.clip_cache_dir` | `./weights/clip` | Directory containing `ViT-L-14.pt`. |

You only need to change these fields if the files are stored elsewhere. Paths are resolved relative to the repository root. The pretrained checkpoints must be available locally before starting a new training run; the training script does not automatically download them.

Downloaded weight files are intentionally excluded from Git. Only this README should be committed from `weights/`. Trained ReGFLoW checkpoints are saved separately in the training output directory (default: `outputs/regflow/`).

Return to [Training](../README.md#training).
