"""Generate ReGFLoW reconstruction-error maps from a prepared OpenSDI cache."""

import argparse
from pathlib import Path

from data_preparation.prepare_cache import configured_datasets
from regflow.config import configure_environment, load_config


VAE_REPO = "CompVis/stable-diffusion-v1-1"
VAE_REVISION = "574555a657f0c8d6e970a02c7e095e9d4220dcbf"


def generate_split(dataset, vae, map_dir, description):
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import to_tensor
    from tqdm import tqdm
    from regflow.data import reconstruction_path

    dataset = dataset.select_columns(["key", "image"])
    keys = dataset["key"]
    generator = torch.Generator(device="cpu").manual_seed(1)

    with torch.inference_mode():
        for start in tqdm(range(0, len(keys), 8), desc=description):
            paths = [reconstruction_path(map_dir, key) for key in keys[start:start + 8]]
            # One 4 x 64 x 64 posterior draw per 512px image from the SD v1.1 VAE.
            # Draw for the full batch so resuming preserves the fresh-run noise sequence.
            noise = torch.randn((len(paths), 4, 64, 64), generator=generator, dtype=vae.dtype)
            if all(path.is_file() for path in paths):
                continue

            images = []
            for image in dataset[start:start + len(paths)]["image"]:
                rgb = to_tensor(image.convert("RGB"))
                images.append(F.interpolate(rgb[None], (512, 512), mode="bilinear", align_corners=False)[0])
            x = torch.stack(images).to(vae.device)
            posterior = vae.encode(x.to(vae.dtype) * 2 - 1).latent_dist
            latent = posterior.mean + posterior.std * noise.to(vae.device)
            decoded = vae.decode(latent, return_dict=False)[0]
            reconstructed = (decoded / 2 + 0.5).clamp(0, 1).float()
            error = (x - reconstructed).abs().mean(1, keepdim=True)
            normalized = ((error - 0.03222192) / 0.04853193).clamp(-5, 5)
            maps = F.interpolate(normalized, (128, 128), mode="bilinear", align_corners=False)
            if not torch.isfinite(maps).all():
                raise ValueError(f"Non-finite reconstruction map in {description} at batch offset {start}")
            for path, value in zip(paths, maps.half().cpu()):
                if path.is_file():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".pt.tmp")
                torch.save(value.clone(), temporary)
                temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--download-vae-only", action="store_true", help="Cache VAE files for offline preprocessing")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_environment(config)

    import torch
    from huggingface_hub import snapshot_download

    if not args.download_vae_only and not torch.cuda.is_available():
        raise RuntimeError("Generating reconstruction maps requires a CUDA device.")
    snapshot = snapshot_download(
        VAE_REPO,
        revision=VAE_REVISION,
        cache_dir=str(Path(config["paths"]["hf_home"]) / "hub"),
        allow_patterns=["vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"],
    )
    if args.download_vae_only:
        print(f"Cached VAE: {Path(snapshot) / 'vae'}")
        return

    # Diffusers 0.30.3 looks up the PyTorch backend on the safetensors package.
    import safetensors.torch
    from diffusers import AutoencoderKL
    from regflow.data import load_opensdi_split

    vae = AutoencoderKL.from_pretrained(
        Path(snapshot) / "vae", variant="fp16", torch_dtype=torch.float16, use_safetensors=True,
    ).eval().requires_grad_(False).to("cuda")
    recon = config["reconstruction"]
    map_dir = Path(recon["root"]) / recon["alias"] / recon["name"]
    data = config["data"]
    for repo, splits, prepared_dir in configured_datasets(config):
        for split in splits:
            dataset = load_opensdi_split(
                repo, split, data["cache_dir"], local_files_only=True,
                cache_config=data["cache_config"], prepared_dir=prepared_dir,
            )
            generate_split(dataset, vae, map_dir, f"{repo}/{split}")


if __name__ == "__main__":
    main()
