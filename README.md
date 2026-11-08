# Less Supervision, Better Generalization: Weakly Supervised Fake Region Localization in Diffusion-Edited Images [NeurIPS'26]

**LSBG** is the official repository for the NeurIPS 2026 paper, *Less Supervision, Better Generalization: Weakly Supervised Fake Region Localization in Diffusion-Edited Images*. The proposed model is **ReGFLoW**.

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

Run the commands below from the LSBG repository root. The example uses Python 3.11 and the CUDA 12.1 build of PyTorch. For a different CUDA setup, select an appropriate build from the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

```bash
conda create -n lsbg python=3.11 -y
conda activate lsbg
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Pretrained Models

ReGFLoW initializes its visual encoders from **MAE ViT-Base** and **OpenAI CLIP ViT-L/14**. The repository includes a `weights/` directory with instructions for downloading the official checkpoints and placing them in the expected locations.

**[Pretrained model setup and backbone roles](weights/README.md)**

## Data Preparation

Download the OpenSDI training and test splits, or configure an existing Hugging Face dataset cache. Training requires image-level real/fake labels; evaluation also uses segmentation masks.

**[Data preparation guide](data_preparation/README.md)**

## Data Preprocessing

Prepare a VAE reconstruction-error map for each training and evaluation image. Existing maps can be reused if they follow the expected format and paths.

**[Data preprocessing guide](data_preprocessing/README.md)**

## Training

The released configuration trains **ReGFLoW** on the OpenSDI SD1.5 training split. The reported experiments use a **single GPU**; the implementation also supports **single-node multi-GPU training with PyTorch DistributedDataParallel (DDP)**.

### Configure training

`configs/train.json` inherits shared settings from `configs/base.json`. Review the following fields before starting a run:

| Configuration | Field | What to set |
| --- | --- | --- |
| `configs/base.json` | `paths.hf_home`, `data.cache_dir` | Hugging Face cache and prepared OpenSDI dataset location. Keep the defaults if using `./cache/huggingface`. |
| `configs/base.json` | `reconstruction.root` | Parent directory of the precomputed reconstruction maps. Keep `null` to use `<hf_home>/recon_maps`. |
| `configs/base.json` | `paths.mae_checkpoint`, `paths.clip_cache_dir` | Locations of the pretrained MAE file and the directory containing the CLIP checkpoint, if different from the defaults in [weights/README.md](weights/README.md). |
| `configs/train.json` | `paths.output_dir` | Directory for this training run (default: `./outputs/regflow`). Use a new directory for a new run. |
| `configs/train.json` | `runtime.num_processes` | Number of GPUs to use: `1` for the reported single-GPU setting, or `N` to enable DDP on `N` visible GPUs. |
| `configs/train.json` | `wandb.enabled`, `wandb.mode` | Choose online logging, offline logging, or disable W&B. With the default online settings, run `wandb login` once. |

The default training configuration uses **10 epochs**, a **per-GPU batch size of 8**, and one gradient-accumulation step. On multiple GPUs, the effective batch size grows with `runtime.num_processes` unless the batch/accumulation settings are adjusted. No separate `torchrun` command is needed for the built-in single-node launcher.

Check the inputs, then start training from the repository root:

```bash
bash train.sh --config configs/train.json --check
wandb login  # Only when using online W&B logging
bash train.sh --config configs/train.json
```

The `--check` command verifies pretrained weights, the required dataset caches and reconstruction maps, and representative sample formats without training.

For **multi-GPU training**, set `runtime.num_processes` in `configs/train.json` to the number of GPUs allocated to the job (for example, `4`). The same training command then launches four workers with DDP. If using Slurm, request the matching GPU count while keeping one node and one Slurm task:

```bash
sbatch --nodes=1 --ntasks=1 --gres=gpu:4 train.sh --config configs/train.json
```

Adjust the GPU count and any partition, account, CPU, memory, or time options for your cluster. To reproduce the single-GPU setup with Slurm, use `--gres=gpu:1` and `runtime.num_processes=1` instead.

### Checkpoints and validation

By default, training saves `best.pth`, `last.pth`, `history.jsonl`, and `resolved_config.json` under `./outputs/regflow/`. The best checkpoint is selected by the SD1.5 **partial-image Pixel F1**. Evaluation during training covers the configured SD1.5, SD2.1, SDXL, SD3, and FLUX.1 test domains.

To resume an interrupted run, set `training.resume` in `configs/train.json` to the existing `last.pth` and keep the same `paths.output_dir`. For a different experiment, use a new output directory.

## Evaluation

Evaluate a trained **ReGFLoW** checkpoint with `configs/test.json`, which also inherits the shared data and reconstruction paths from `configs/base.json`.

### Configure evaluation

| Configuration | Field | What to set |
| --- | --- | --- |
| `configs/test.json` | `paths.checkpoint` | Trained ReGFLoW checkpoint (default: `./outputs/regflow/best.pth`). |
| `configs/test.json` | `paths.output_dir` | Evaluation output directory (default: `./outputs/test`). |
| `configs/test.json` | `runtime.num_processes` | `1` for single-GPU evaluation, or `N` to evaluate with `N` visible GPUs. |
| `configs/test.json` | `evaluation.pixel_classes` | Pixel-level evaluation subset; default `["partial"]` evaluates partially edited images (**P**). Use `["partial", "fully"]` for **P+F**. |
| `configs/test.json` | `evaluation.run_image_eval` | Set to `true` to also evaluate image-level detection over real, partially edited, and fully synthetic images. |

Make sure the paths in `configs/base.json` point to the same prepared data and reconstruction maps used for evaluation. Run a preflight check, followed by evaluation:

```bash
bash test.sh --config configs/test.json --check
bash test.sh --config configs/test.json
```

The evaluation script supports multi-GPU execution using `runtime.num_processes` in `configs/test.json`, in the same way as training. Under Slurm, request the same number of GPUs with `--gres=gpu:<count>` and keep `--ntasks=1`.

Results are written to **`<paths.output_dir>/test_summary.json`** (by default, `outputs/test/test_summary.json`). The result file records the selected pixel-level protocol, image-level metrics when enabled, and per-domain and averaged results. **P and P+F require separate runs with their respective `evaluation.pixel_classes` settings**; they are not both calculated automatically in the default run.

## Acknowledgements

We thank the authors of [OpenSDI](https://github.com/iamwangyabin/OpenSDI), [MAE](https://github.com/facebookresearch/mae), [CLIP](https://github.com/openai/CLIP), [IML-ViT](https://github.com/SunnyHaze/IML-ViT), [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo), [AEROBLADE](https://github.com/jonasricker/aeroblade), and [Latent Diffusion](https://github.com/CompVis/latent-diffusion) for their work. [Third-party sources and licenses](THIRD_PARTY_NOTICES.md)
