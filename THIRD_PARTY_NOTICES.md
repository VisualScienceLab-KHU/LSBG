# Third-party sources and notices

This repository contains a reorganized and modified version of the supplied research code. Module names and file boundaries have changed; the acknowledgements below identify the upstream components. No third-party pretrained weights or datasets are bundled.

| Component | Source | Status / notice |
| --- | --- | --- |
| Original backbone integration and dataset implementation | [OpenSDI / MaskCLIP](https://github.com/iamwangyabin/OpenSDI) | Cite Wang, Huang, and Hong, CVPR 2025. The upstream repository inspected for this release does not provide a repository-level code license. Its dataset license must not be treated as a license for its code. |
| Windowed ViT backbone and feature-pyramid implementation | [IML-ViT](https://github.com/SunnyHaze/IML-ViT), especially `modules/window_attention_ViT.py` | Copyright (c) 2023 Xiaochen Ma. MIT license: [full text](licenses/IML-ViT-MIT.txt). The supplied source explicitly identifies this origin; unused components were removed and names/imports reorganized. |
| ViT window partitioning, relative/absolute positional embeddings, and related helpers inherited through IML-ViT | [Detectron2](https://github.com/facebookresearch/detectron2), including `detectron2/modeling/backbone/utils.py` | Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. Apache License 2.0: [full text](licenses/Detectron2-Apache-2.0.txt). The helpers are adapted in the ReGFLoW backbone. |
| CLIP implementation and pretrained visual encoder | [OpenAI CLIP](https://github.com/openai/CLIP) | Copyright (c) 2021 OpenAI. MIT license: [full text](licenses/CLIP-MIT.txt). Installed as a dependency; weights are obtained separately. |
| MAE ViT-Base initialization checkpoint | [MAE](https://github.com/facebookresearch/mae) / [official download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | He et al., *Masked Autoencoders Are Scalable Vision Learners*. The upstream MAE project is distributed under CC BY-NC 4.0: [full text](licenses/MAE-CC-BY-NC-4.0.txt). This notice concerns the external checkpoint/source, not a new license for all ReGFLoW code. |
| Reconstruction-error preprocessing concept | [AEROBLADE](https://github.com/jonasricker/aeroblade) | Ricker, Lukovnikov, and Fischer, CVPR 2024. The README recipe uses autoencoder reconstruction, with RGB L1 error and the supplied ReGFLoW normalization; it is not AEROBLADE's default LPIPS detector. |
| Reconstruction VAE | [CompVis/stable-diffusion-v1-1](https://huggingface.co/CompVis/stable-diffusion-v1-1) | Rombach et al., CVPR 2022. CreativeML OpenRAIL-M model license applies to the separately downloaded model. Only the VAE subfolder is needed. |
| OpenSDI datasets | [Training](https://huggingface.co/datasets/nebula/OpenSDI_train), [testing](https://huggingface.co/datasets/nebula/OpenSDI_test) | CC BY-SA 4.0, as stated by the dataset authors. Preserve their attribution and consult their dataset cards when using or redistributing the data. |

## Project license

The supplied archive did not specify a ReGFLoW project license. This cleanup does not invent one or extend third-party permissions. Before publicly licensing the combined repository, its owner needs to choose the terms for original contributions and establish the applicable permissions for retained OpenSDI-derived code. This does not affect local preparation, inspection, or testing of the supplied code.
