import math
import time
from contextlib import nullcontext
from itertools import islice

import torch

from .config import optimizer_with_defaults
from .metrics import Metrics
from .runtime import all_reduce_sum, check_finite


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def adjust_learning_rate(optimizer, progress, config):
    schedule, epochs = config["optimizer"], config["training"]["epochs"]
    warmup = schedule["warmup_epochs"]
    if progress < warmup:
        lr = schedule["lr"] * progress / warmup
    else:
        phase = min(1.0, max(0.0, (progress - warmup) / (epochs - warmup)))
        lr = (
            schedule["min_lr"]
            + (schedule["lr"] - schedule["min_lr"]) * (1 + math.cos(math.pi * phase)) / 2
        )
    settings = optimizer_with_defaults(schedule)
    image_scale = settings["image_lr_scale"]
    after = settings["image_decay_after_epoch"]
    if after is not None and progress >= after:
        image_scale *= settings["image_decay_factor"]
    for group in optimizer.param_groups:
        scale = image_scale if group.get("is_image", False) else 1.0
        group["lr"] = lr * group.get("lr_scale", 1.0) * scale
    return lr


def _mean_losses(totals, count, device):
    keys = sorted(totals)
    values = torch.stack(
        [totals[key] for key in keys]
        + [torch.tensor(float(count), device=device, dtype=torch.float64)]
    )
    values = all_reduce_sum(values)
    return {key: (values[index] / values[-1]).item() for index, key in enumerate(keys)}


def train_epoch(model, loader, optimizer, scaler, epoch, config, runtime, *, log_train=None):
    model.train()
    training = config["training"]
    steps = min(len(loader), training["max_batches"]) if training["max_batches"] else len(loader)
    if steps == 0:
        raise ValueError("The training loader is empty.")
    accumulation = training["accumulation_steps"]
    amp = training["amp"]
    if amp and runtime.device.type != "cuda":
        raise ValueError("training.amp=true requires a CUDA device in this release.")
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    window = {}
    window_steps = 0
    start = time.monotonic()
    for step, batch in enumerate(islice(loader, steps)):
        batch = move_batch(batch, runtime.device)
        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, steps - group_start)
        end_group = step + 1 == group_start + group_size
        if step == group_start:
            adjust_learning_rate(optimizer, epoch + step / steps, config)
        if step == 0:
            learning_rates_start = {
                group.get("name", f"group_{index}"): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            }
            runtime.print(f"Epoch {epoch + 1} starting learning rates: {learning_rates_start}")
        sync = model.no_sync() if hasattr(model, "no_sync") and not end_group else nullcontext()
        with sync:
            with torch.autocast(device_type=runtime.device.type, dtype=torch.float16, enabled=amp):
                output = model(
                    image=batch["image"], label=batch["label"], recon_map=batch["recon_map"]
                )
                loss = output["backward_loss"]
            check_finite(loss)
            scaler.scale(loss / group_size).backward()
        if end_group:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        for name, value in output["visual_loss"].items():
            if name not in totals:
                totals[name] = torch.zeros((), device=runtime.device, dtype=torch.float64)
            detached = value.detach().double()
            totals[name] += detached
            if log_train is not None:
                if name not in window:
                    window[name] = torch.zeros((), device=runtime.device, dtype=torch.float64)
                window[name] += detached
        window_steps += 1
        if (step + 1) % config["runtime"]["log_every"] == 0 or step + 1 == steps:
            if log_train is not None:
                log_train(
                    _mean_losses(window, window_steps, runtime.device),
                    epoch=epoch,
                    step=step + 1,
                    steps=steps,
                )
                window.clear()
                window_steps = 0
            runtime.print(
                f"Epoch {epoch + 1}/{training['epochs']} | step {step + 1}/{steps} | "
                f"loss={loss.detach().item():.6f} | lr={optimizer.param_groups[0]['lr']:.3g}"
            )
    stats = _mean_losses(totals, steps, runtime.device)
    stats.update(
        steps_per_process=steps,
        seconds=time.monotonic() - start,
        learning_rates_start=learning_rates_start,
        learning_rates_end={
            group.get("name", f"group_{index}"): group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        },
    )
    return stats


@torch.no_grad()
def evaluate(
    model, loader, runtime, *, pixel_mode=None, image_metrics=False, threshold=0.5, max_batches=0
):
    model.eval()
    metrics = Metrics(
        runtime.device, pixel_mode=pixel_mode, image=image_metrics, threshold=threshold
    )
    steps = min(len(loader), max_batches) if max_batches else len(loader)
    for batch in islice(loader, steps):
        batch = move_batch(batch, runtime.device)
        output = model(image=batch["image"], recon_map=batch["recon_map"])
        metrics.update(output, batch)
    return metrics.compute()
