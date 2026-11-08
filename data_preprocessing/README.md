# Data Preprocessing

Prepare one reconstruction-error map per OpenSDI image. Train and test use the same preprocessing. Include real test images because image-level detection also uses their reconstruction maps. Start with the [prepared OpenSDI cache](../data_preparation/README.md).

## Generate reconstruction maps

Run the following commands from the repository root on a CUDA machine. The first run downloads the VAE configuration and fp16 weights for Stable Diffusion v1.1. The script processes the training split and all five test splits. For evaluation alone, pass `--config configs/test.json`.

```bash
python -m pip install -r requirements-preprocess.txt
python -m data_preprocessing.generate_maps --config configs/train.json
```

For a GPU machine without network access, first run the same module with `--download-vae-only` on a machine that shares `paths.hf_home`. Then run the generation command on the GPU machine with `HF_HUB_OFFLINE=1`.

Each RGB image is resized to 512 × 512 and reconstructed with one VAE encode/decode pass using a sample from the posterior. The map is the mean absolute reconstruction error across the three RGB channels. The script subtracts `0.03222192`, divides by `0.04853193`, clips to `[-5, 5]`, and resizes bilinearly to 128 × 128. It saves one fp16 tensor of shape `[1, 128, 128]` per image. [Stable Diffusion v1.1 VAE](https://huggingface.co/CompVis/stable-diffusion-v1-1/tree/main/vae) · [AEROBLADE](https://github.com/jonasricker/aeroblade)

## Reuse maps or resume generation

If all maps are present, proceed to [training or evaluation](../README.md). If generation was interrupted, rerun the same command. It fills missing files and preserves the posterior noise sequence across skipped batches.

ReGFLoW expects one map for each original OpenSDI key, including its subdirectories and filename extension, followed by `.pt`:

```text
<reconstruction.root>/sd1_v1/vae_l1_rgbmean_zreal_128/<key>.pt
```

## Configure preprocessing paths

Set `reconstruction.root` in [`configs/base.json`](../configs/base.json) to the directory **containing** `sd1_v1/`. Leave it as `null` to use `<paths.hf_home>/recon_maps`. The default `reconstruction.alias` (`sd1_v1`) and `reconstruction.name` (`vae_l1_rgbmean_zreal_128`) must match the names of your prepared map directories. The same configuration is inherited by both training and evaluation.
