from pathlib import Path, PurePosixPath, PureWindowsPath

import torch


def reconstruction_path(root, key):
    key = str(key).strip().replace("\\", "/")
    relative = PurePosixPath(key)
    if (
        not key
        or relative.is_absolute()
        or PureWindowsPath(key).drive
        or ".." in relative.parts
        or key.endswith("/")
    ):
        raise ValueError(f"Reconstruction key must be a safe relative path: {key!r}")
    root = Path(root).expanduser().resolve()
    path = (root / f"{key}.pt").resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Reconstruction key escapes the map directory: {key!r}")
    return path


def load_reconstruction_map(path, dtype="float32"):
    if dtype not in {"float16", "float32"}:
        raise ValueError("Reconstruction dtype must be 'float16' or 'float32'.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing reconstruction map: {path}. Prepare maps as described in README.md."
        )
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(tensor, dict):
        for key in ("recon_map", "map", "recon_error", "recon_error_map"):
            if key in tensor:
                tensor = tensor[key]
                break
        else:
            raise ValueError(f"No reconstruction tensor found in {path}.")
    if not torch.is_tensor(tensor):
        raise TypeError(f"Reconstruction map must contain a torch.Tensor: {path}")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1 or min(tensor.shape[-2:]) <= 0:
        raise ValueError(
            f"Expected a [1,H,W] reconstruction map, got {tuple(tensor.shape)}: {path}"
        )
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError(f"Reconstruction map must contain finite floating-point values: {path}")
    tensor = tensor.to(dtype=getattr(torch, dtype))
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Reconstruction values overflow {dtype}: {path}")
    return tensor
