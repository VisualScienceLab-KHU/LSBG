import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import optimizer_with_defaults


def read_checkpoint(path):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path!r}. Set paths.checkpoint or training.resume in JSON."
        )
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("A checkpoint must be a state_dict or a dict containing model weights.")
    return checkpoint


def restore_model(model, checkpoint, config, runtime):
    from .models import load_model_weights

    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    ignored = load_model_weights(model, state)
    if ignored:
        runtime.print(
            f"Loaded legacy weights; discarded {len(ignored)} tensors from unused text/decoder modules."
        )
    previous = checkpoint.get("config", {}).get("model", {})
    changes = {
        key: {"checkpoint": previous[key], "current": value}
        for key, value in config["model"].items()
        if key in previous and previous[key] != value
    }
    if changes:
        runtime.print(f"Explicit model settings differing from the checkpoint: {changes}")
    if not previous:
        runtime.print(
            "Legacy checkpoint has no complete model config; current JSON settings are authoritative."
        )


def _rng_state(device):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def resume_training(checkpoint, optimizer, scaler, config, runtime):
    if checkpoint.get("format_version") != 1:
        raise ValueError(
            "Optimizer resume requires a checkpoint produced by this release. "
            "Legacy model weights can be evaluated with test.py, but their optimizer layout is not migrated."
        )
    previous = checkpoint["config"]
    if previous["model"] != config["model"] or (
        optimizer_with_defaults(previous["optimizer"]) != optimizer_with_defaults(config["optimizer"])
    ):
        raise ValueError(
            "Resume requires unchanged model/optimizer settings, including the image LR schedule. "
            "Start a new run to compare schedules; use the checkpoint's resolved_config.json to resume."
        )
    guarded = {
        "training": (
            "epochs",
            "batch_size",
            "accumulation_steps",
            "augmentation",
            "data_scale",
            "amp",
            "max_batches",
        ),
        "data": ("image_size", "train_repo_id", "train_split", "test_repo_id", "test_splits"),
        "runtime": ("seed",),
        "reconstruction": ("alias", "name", "dtype"),
        "validation": (
            "batch_size",
            "threshold",
            "max_batches",
            "selection_split",
            "model_overrides",
        ),
    }
    changed = [
        f"{section}.{key}"
        for section, keys in guarded.items()
        for key in keys
        if key not in previous.get(section, {}) or previous[section][key] != config[section][key]
    ]
    if changed:
        raise ValueError(
            f"Resume requires the same training schedule/data and validation settings; changed: {changed}"
        )
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    states = checkpoint["rng_states"]
    if len(states) == runtime.world_size:
        _restore_rng(states[runtime.rank], runtime.device)
    else:
        runtime.print(
            "Process count changed on resume; weights/optimizer restored, but RNG trajectory will differ."
        )
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_metric"])


def save_training_checkpoint(
    model, optimizer, scaler, epoch, best_metric, config, runtime, *, is_best=False
):
    state = _rng_state(runtime.device)
    states = [None] * runtime.world_size
    if dist.is_initialized():
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if not runtime.is_main:
        return
    checkpoint = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "rng_states": states,
    }
    output = Path(config["paths"]["output_dir"])
    destinations = [output / "last.pth"]
    if is_best:
        destinations.append(output / "best.pth")
    period = config["training"]["save_every"]
    if period and (epoch + 1) % period == 0:
        destinations.append(output / f"epoch-{epoch + 1:03d}.pth")
    for path in destinations:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
