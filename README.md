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

## 1. Environment and pretrained weights

Run commands from the repository root. Create a Python 3.11 environment and install the example CUDA 12.1 build ([other PyTorch builds](https://pytorch.org/get-started/previous-versions/)):

```bash
conda create -n regflow python=3.11 -y
conda activate regflow
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Place the official checkpoints at these paths; existing files can be reused:

```text
weights/
├── mae/
│   └── mae_pretrain_vit_base.pth
└── clip/
    └── ViT-L-14.pt
```

**MAE:** [He et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html), [official repository](https://github.com/facebookresearch/mae), [ViT-Base checkpoint](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth). **CLIP:** [Radford et al., ICML 2021](https://proceedings.mlr.press/v139/radford21a.html), [official OpenAI CLIP ViT-L/14](https://github.com/openai/CLIP). Use OpenAI's `ViT-L-14.pt`, not an OpenCLIP or trained ReGFLoW checkpoint. Training reads local weights and does not download missing files.

## 2. OpenSDI data and paths

We use [OpenSDI](https://github.com/iamwangyabin/OpenSDI) (Wang et al., CVPR 2025): [training data](https://huggingface.co/datasets/nebula/OpenSDI_train) and [test data](https://huggingface.co/datasets/nebula/OpenSDI_test). Follow the authors' dataset terms, including CC BY-SA 4.0.

All settings are in JSON. `configs/train.json` and `configs/test.json` inherit `configs/base.json`. Relative paths resolve from the **repository root**.

**Set your actual cache directories in `configs/base.json`**, keeping its other fields. The following is an excerpt, not a replacement for the entire file:

```json
{
  "paths": {"hf_home": "/path/to/your/huggingface"},
  "data": {"cache_dir": "/path/to/your/huggingface/datasets"},
  "reconstruction": {"root": "/path/to/your/recon_maps"}
}
```

| Field | Location |
| --- | --- |
| `paths.hf_home` | Hugging Face cache root; default `./cache/huggingface`. |
| `data.cache_dir` | Prepared dataset cache root; `null` uses `<hf_home>/datasets`. |
| `reconstruction.root` | Parent of `sd1_v1/`; `null` uses `<hf_home>/recon_maps`. |
| `paths.mae_checkpoint` | `./weights/mae/mae_pretrain_vit_base.pth`. |
| `paths.clip_cache_dir` | `./weights/clip`, containing `ViT-L-14.pt`; independent of `hf_home`, including when `null`. |

Reuse a complete Arrow cache by preserving its configuration/version/hash directories, `dataset_info.json`, and shards. Default `data.local_files_only=true` reads prepared Arrow directly; it does not download data during training. To select a specific cache version, set `data.train_prepared_dir` / `data.test_prepared_dir` to its `dataset_info.json` directory. Dataset keys must remain unchanged because they identify reconstruction maps. Training uses image-level labels and drops ground-truth masks.

For a fresh cache, expand the recipe below. It downloads official data and prepares Arrow without exporting images; downloaded and prepared copies can both occupy disk space.

<details>
<summary>Download and verify the dataset cache</summary>

```bash
python - <<'PY'
import os
from pathlib import Path
from regflow.config import load_config, configure_environment
cfg = load_config('configs/train.json')
configure_environment(cfg)
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'

from datasets import load_dataset
from regflow.data import load_opensdi_split

cache = cfg['data']['cache_dir']
jobs = {
    cfg['data']['train_repo_id']: [cfg['data']['train_split']],
    cfg['data']['test_repo_id']: cfg['data']['test_splits'],
}
for repo, splits in jobs.items():
    prepared = load_dataset(repo, name=cfg['data']['cache_config'], cache_dir=cache)
    for split in splits:
        directory = str(Path(prepared[split].cache_files[0]['filename']).parent)
        local = load_opensdi_split(repo, split, cache, local_files_only=True,
                                  cache_config=cfg['data']['cache_config'], prepared_dir=directory)
        assert len(local) == len(prepared[split])
        print(repo, split, len(local), local.column_names)
PY
```

The cache contains `key`, `image`, `mask`, and binary `label` columns. The training loader drops the mask column and uses only image-level labels. Each `key` also identifies its reconstruction map. Keep these keys unchanged when moving data.

</details>

## 3. Reconstruction maps

Precompute maps for **training and all test domains/categories** before running the model. Reuse existing maps, or follow the recipe below using the VAE from [Stable Diffusion v1.1](https://huggingface.co/CompVis/stable-diffusion-v1-1) ([latent diffusion](https://arxiv.org/abs/2112.10752)). This follows the autoencoder-reconstruction idea of [AEROBLADE (Ricker et al., CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.html), but uses **RGB-mean L1 error and ReGFLoW normalization**, not AEROBLADE's default LPIPS detector. No diffusion denoising is performed.

Each map is a `[1,128,128]` tensor stored at:

```text
<reconstruction.root>/sd1_v1/vae_l1_rgbmean_zreal_128/<key>.pt
```

The loader transforms images and maps together. For another dataset, prepare equivalent maps before execution; no standalone raw-image inference script is included.

<details>
<summary>Generate maps from the dataset cache</summary>

Install optional preprocessing dependencies and download only the VAE files:

```bash
python -m pip install -r requirements-preprocess.txt
python - <<'PY'
import os
from pathlib import Path
from regflow.config import load_config, configure_environment
cfg = load_config('configs/train.json')
configure_environment(cfg)
os.environ['HF_HUB_OFFLINE'] = '0'
from huggingface_hub import snapshot_download
snapshot_download(
    'CompVis/stable-diffusion-v1-1',
    revision='574555a657f0c8d6e970a02c7e095e9d4220dcbf',
    cache_dir=str(Path(cfg['paths']['hf_home']) / 'hub'),
    allow_patterns=['vae/config.json', 'vae/diffusion_pytorch_model.fp16.safetensors'],
)
PY
```

Run on a CUDA device. This processes the training split and every configured test split.

```bash
python - <<'PY'
from pathlib import Path, PurePosixPath
from regflow.config import load_config, configure_environment
cfg = load_config('configs/train.json')
configure_environment(cfg)

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from diffusers import AutoencoderKL
from tqdm import tqdm
from regflow.data import load_opensdi_split

if not torch.cuda.is_available():
    raise RuntimeError('This recipe requires CUDA to preserve the original fp16 VAE computation.')
vae = AutoencoderKL.from_pretrained(
    'CompVis/stable-diffusion-v1-1', subfolder='vae',
    revision='574555a657f0c8d6e970a02c7e095e9d4220dcbf',
    variant='fp16', torch_dtype=torch.float16, use_safetensors=True,
    cache_dir=str(Path(cfg['paths']['hf_home']) / 'hub'), local_files_only=True,
).eval().requires_grad_(False).to('cuda')
recon = cfg['reconstruction']
maps = Path(recon['root']) / recon['alias'] / recon['name']
jobs = [(cfg['data']['train_repo_id'], cfg['data']['train_split'])]
jobs += [(cfg['data']['test_repo_id'], s) for s in cfg['data']['test_splits']]

def map_path(key):
    key = PurePosixPath(str(key).strip().replace('\\', '/'))
    if key.is_absolute() or '..' in key.parts:
        raise ValueError(f'Invalid dataset key: {key}')
    return maps / (str(key) + '.pt')

with torch.inference_mode():
    for repo, split in jobs:
        prepared = cfg['data']['train_prepared_dir'] if repo == cfg['data']['train_repo_id'] else cfg['data']['test_prepared_dir']
        data = load_opensdi_split(repo, split, cfg['data']['cache_dir'],
                                 local_files_only=True, cache_config=cfg['data']['cache_config'], prepared_dir=prepared)
        data = data.remove_columns([c for c in data.column_names if c not in {'key', 'image'}])
        generator = torch.Generator(device='cpu').manual_seed(1)
        for start in tqdm(range(0, len(data), 8), desc=f'{repo}/{split}'):
            images, paths = [], []
            for i in range(start, min(start + 8, len(data))):
                row = data[i]
                path = map_path(row['key'])
                if path.exists():
                    continue
                image = to_tensor(row['image'].convert('RGB')).float().clamp(0, 1)
                images.append(F.interpolate(image[None], (512, 512), mode='bilinear', align_corners=False)[0])
                paths.append(path)
            if not images:
                continue
            x = torch.stack(images).to('cuda')
            latent = vae.encode(x.to(vae.dtype) * 2 - 1).latent_dist.sample(generator=generator)
            reconstruction = (vae.decode(latent, return_dict=False)[0] / 2 + 0.5).clamp(0, 1).float()
            error = (x.float() - reconstruction).abs().mean(1, keepdim=True)
            normalized = ((error - 0.03222192) / 0.04853193).clamp(-5, 5)
            output = F.interpolate(normalized, (128, 128), mode='bilinear', align_corners=False)
            if not torch.isfinite(output).all():
                raise RuntimeError('Non-finite reconstruction map; maps were not saved.')
            for path, value in zip(paths, output.half().cpu()):
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.pt.tmp')
                torch.save(value.contiguous(), temporary)
                temporary.replace(path)
PY
```

Keep the same normalization across train/test: L1 error → normalize (`mean=0.03222192`, `std=0.04853193`) → clip to `[-5,5]` → resize to 128×128 → fp16 save. VAE posterior sampling is stochastic; reuse existing maps for the closest comparison with previous runs.

</details>

## 4. Training

**All reported ReGFLoW results use a single GPU for both training and inference.** To reproduce the training setup, set `runtime.num_processes` to `1` and `training.epochs` to `10` in `configs/train.json`. Use 8 images per batch and one batch per optimizer step (effective batch size 8). The remaining defaults are seed 42, base learning rate `5e-5`, weight decay `0.05`, augmentation enabled, and AMP disabled. Use a new `paths.output_dir` for each run.

```bash
bash train.sh --config configs/train.json --check
wandb login
bash train.sh --config configs/train.json
```

`--check` verifies local weights, caches, maps, and sample formats without training. W&B login is unnecessary when `wandb.enabled=false` or `wandb.mode="offline"`.

For [Slurm](https://slurm.schedmd.com/sbatch.html), activate the environment and submit from the repository root:

```bash
sbatch --nodes=1 --ntasks=1 --gres=gpu:1 train.sh --config configs/train.json
```

Add your site's partition, account, CPU, RAM, and time options before `train.sh`. The wrapper scripts contain no fixed resource requests. Use **one node and one task** because the Python launcher starts the workers. Set `PYTHON=/path/to/python` to select the interpreter, or run `python train.py --config configs/train.json` directly.

**Optional multi-GPU training.** The implementation supports PyTorch DistributedDataParallel (DDP). To train on multiple GPUs, set `runtime.num_processes` in `configs/train.json` to the number of GPUs you want to use. Under Slurm, request the same number with `--gres=gpu:<count>` and keep `--ntasks=1`. With 8 images per GPU and one batch per optimizer step, the effective batch size is `8 × runtime.num_processes`.

### Image-branch learning rate

These are configurable optimization hyperparameters, not architecture changes. The current values are in `configs/train.json`.

| Field | Meaning |
| --- | --- |
| `optimizer.image_lr_scale` | Image-branch LR multiplier from the first update. |
| `optimizer.image_decay_after_epoch` | Completed epochs before an extra drop; integer `1 <= value < training.epochs`, or `null` to disable. |
| `optimizer.image_decay_factor` | One-time multiplier in `(0,1]`, not compounded each epoch. |

Only `cls_aggregator`, `image_classifier`, and `loc_image_fusion` receive the extra scale; other groups retain their cosine schedule. Existing group multipliers, including `new_head_lr_mult=3.0`, remain active. For example, scale `1.0`, milestone `2`, factor `0.1` means normal image LR in epochs 1–2 and **0.1× the otherwise scheduled LR from epoch 3 onward**. Start a new run when changing the schedule.

### Validation, logging, and checkpoints

Each epoch evaluates SD1.5, SD2.1, SDXL, SD3, and FLUX.1:

| Metric | Evaluation |
| --- | --- |
| Pixel F1 | `partial` images; average the per-image pixel F1. |
| Image F1 | `real` + `partial` + `fully`; pool image-level TP/FP/FN, with fake as positive. |
| AVG / OOD Avg | Equal-weight domain means; OOD excludes SD1.5. |

The default threshold is `0.5`. Validation applies test calibration (`pred_calib_k_max=1.0`) and then restores training settings. `best.pth` is selected by SD1.5 Pixel F1; `last.pth`, `history.jsonl`, and `resolved_config.json` are also saved under `outputs/regflow/` by default. To resume the same run, keep its output directory and set `training.resume` to its `last.pth`.

W&B settings are in `configs/train.json`: `project` defaults to `ReGFLoW`; set `name` for a run, `entity` for an account/team, and optional `group`/`tags`. One training run is logged regardless of GPU count. Keep credentials outside config files.

Losses are averaged over each `runtime.log_every` interval (default 20 batches) and logged on a **fractional epoch** axis. During DDP training, the averages also include all participating GPUs. Validation logs once per epoch in separate **`val_pixel` / `val_image`** sections, ordered by numbered keys: SD1.5 → SD2.1 → SDXL → SD3 → FLUX.1 → AVG → OOD Avg. Enable [alphabetical panel sorting](https://docs.wandb.ai/models/track/workspaces#sort-panels-alphabetically) in the workspace. F1 values lie in `[0,1]`; no IoU or qualitative images are logged. Retain `wandb_run.json` for online resume; standalone testing does not create a W&B run.

## 5. Evaluation

For the reported single-GPU evaluation, set `runtime.num_processes` to `1` in `configs/test.json`. Set `paths.checkpoint` to the checkpoint to evaluate (default `./outputs/regflow/best.pth`).

```bash
bash test.sh --config configs/test.json --check
bash test.sh --config configs/test.json
```

For single-GPU evaluation under Slurm, run:

```bash
sbatch --nodes=1 --ntasks=1 --gres=gpu:1 test.sh --config configs/test.json
```

To evaluate on multiple GPUs, set `runtime.num_processes` in `configs/test.json` to the desired GPU count and request that many GPUs from Slurm if applicable. Results are saved to `outputs/test/test_summary.json` by default.

| `evaluation.pixel_classes` | Pixel protocol |
| --- | --- |
| `["partial"]` (default) | Partially edited images; mean per-image F1/recall/accuracy. |
| `["partial", "fully"]` | P+F: partially edited and fully synthetic; pooled-pixel F1/recall. |
| `["partial", "fully", "real"]` | All images; pooled-pixel F1/recall. |

With `evaluation.run_image_eval=true`, image F1/accuracy use **all three categories**, independently of the pixel subset. Do not compare results from different subsets or aggregation rules as the same protocol.

## Reproducibility and source release

Report the resolved configuration with results. The image-only milestone is an additional optimization setting, not a validated improvement or a claim of exact paper reproduction. The active `new_head_lr_mult` also corrects an overwrite in the original experimental scheduler. Changing GPU count or accumulation may change results even at the same effective batch size.

Epoch-wise validation uses **`nebula/OpenSDI_test`, including OOD domains**, not a held-out training split. Choosing hyperparameters from these curves is test-informed tuning; use a separate source-training validation set for final model selection.

Keep machine-specific path/run overrides in ignored `configs/*.local.json` files using `extends`. `.gitignore` excludes weights, caches, outputs, logs, and credentials; it does not untrack previously committed files or erase secrets from history. Review staged changes before publishing.

## Acknowledgements and licenses

We acknowledge [OpenSDI](https://github.com/iamwangyabin/OpenSDI), [MAE](https://github.com/facebookresearch/mae), [CLIP](https://github.com/openai/CLIP), [IML-ViT](https://github.com/SunnyHaze/IML-ViT), [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo), [AEROBLADE](https://github.com/jonasricker/aeroblade), and [latent diffusion](https://github.com/CompVis/latent-diffusion). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/` for component attribution and the unresolved project-wide licensing status. Weights and datasets are not bundled.

<details>
<summary>OpenSDI dataset and MAE citations</summary>

```bibtex
@InProceedings{wang2025opensdi,
  author = {Wang, Yabin and Huang, Zhiwu and Hong, Xiaopeng},
  title = {OpenSDI: Spotting Diffusion-Generated Images in the Open World},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025}
}

@InProceedings{He_2022_CVPR,
  author = {He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll\'ar, Piotr and Girshick, Ross},
  title = {Masked Autoencoders Are Scalable Vision Learners},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022},
  pages = {16000-16009}
}
```

</details>
