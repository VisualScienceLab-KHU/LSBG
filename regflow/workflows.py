import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import read_checkpoint, restore_model, resume_training, save_training_checkpoint
from .config import save_json, separate_image_lr
from .data import (
    BalancedBatchSampler,
    DistributedEvalSampler,
    OpenSDIDataset,
)
from .engine import accumulate_metrics, train_epoch
from .metrics import Metrics, average_domains
from .models import ReGFLoW
from .runtime import seed_worker
from .tracking import WandbTracker


TEST_CATEGORIES = ("partial", "fully", "real")


def build_dataset(config, split, *, training=False, classes=None, load_masks=True):
    data, recon = config["data"], config["reconstruction"]
    train = config.get("training", {})
    return OpenSDIDataset(
        data["train_repo_id"] if training else data["test_repo_id"],
        split,
        data["cache_dir"],
        recon_root=recon["root"],
        image_size=data["image_size"],
        augmentation=training and train["augmentation"],
        classes=classes,
        data_scale=train["data_scale"] if training else 1.0,
        seed=config["runtime"]["seed"],
        local_files_only=data["local_files_only"],
        recon_alias=recon["alias"],
        recon_name=recon["name"],
        recon_dtype=recon["dtype"],
        load_masks=load_masks,
        cache_config=data["cache_config"],
        prepared_dir=data["train_prepared_dir"] if training else data["test_prepared_dir"],
    )


def build_loader(dataset, batch_size, config, runtime, *, training=False):
    generator = torch.Generator().manual_seed(config["runtime"]["seed"] + runtime.rank)
    kwargs = dict(
        num_workers=config["runtime"]["num_workers"],
        pin_memory=config["runtime"]["pin_memory"],
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if training:
        sampler = BalancedBatchSampler(
            dataset,
            batch_size,
            rank=runtime.rank,
            world_size=runtime.world_size,
            seed=config["runtime"]["seed"],
        )
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    sampler = DistributedEvalSampler(dataset, rank=runtime.rank, world_size=runtime.world_size)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False, **kwargs)


def build_model(config, runtime, checkpoint=None):
    model = ReGFLoW(
        **config["model"],
        mae_checkpoint=config["paths"]["mae_checkpoint"],
        clip_cache_dir=config["paths"]["clip_cache_dir"],
        initialize_pretrained=checkpoint is None,
    )
    if checkpoint is not None:
        restore_model(model, checkpoint, config, runtime)
    return model.to(runtime.device)


def _prepare_training_output(config, runtime):
    output = Path(config["paths"]["output_dir"])
    errors = [None]
    if runtime.is_main:
        try:
            existing = any(
                (output / name).exists()
                for name in ("last.pth", "best.pth", "resolved_config.json", "history.jsonl")
            ) or bool(list(output.glob("epoch-*.pth")))
            resume = config["training"]["resume"]
            same_run = bool(resume) and Path(resume).resolve().parent == output.resolve()
            if existing and not same_run:
                raise FileExistsError(
                    f"Run already exists at {output}. Choose a new paths.output_dir, "
                    "or resume from a checkpoint inside this run directory."
                )
            output.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as error:
            errors[0] = str(error)
    if dist.is_initialized():
        dist.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise RuntimeError(errors[0])


def _save_training_config(config, runtime):
    errors = [None]
    if runtime.is_main:
        try:
            save_json(Path(config["paths"]["output_dir"]) / "resolved_config.json", config)
        except (OSError, ValueError) as error:
            errors[0] = str(error)
    if dist.is_initialized():
        dist.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise RuntimeError(errors[0])


def validate_domains(model, loaders, config, runtime):
    """Evaluate P, P+F and image-level metrics using one model pass per image.

    P uses mean per-image F1 on partially edited images. P+F pools TP/FP/FN
    over partially edited and fully synthetic images, as in test().
    """
    settings = config["validation"]
    overrides = settings["model_overrides"]
    previous = {name: getattr(model, name) for name in overrides}
    results = {}
    try:
        for name, value in overrides.items():
            setattr(model, name, value)
        for split in config["data"]["test_splits"]:
            pixel_p = Metrics(runtime.device, pixel_mode="per_image", threshold=settings["threshold"])
            pixel_pf = Metrics(runtime.device, pixel_mode="entire_pixel", threshold=settings["threshold"])
            image = Metrics(runtime.device, image=True, threshold=settings["threshold"])
            metrics_by_class = {
                "partial": (pixel_p, image),
                "fully": (pixel_pf, image),
                "real": (image,),
            }
            for category in TEST_CATEGORIES:
                accumulate_metrics(
                    model,
                    loaders[split][category],
                    runtime,
                    *metrics_by_class[category],
                    max_batches=settings["max_batches"],
                )
            pixel_pf.include_pixels(pixel_p)
            results[split] = {
                "pixel_p": pixel_p.compute(),
                "pixel_pf": pixel_pf.compute(),
                "image": image.compute(),
            }
            runtime.print(f"Validation {split}: {json.dumps(results[split], sort_keys=True)}")
    finally:
        for name, value in previous.items():
            setattr(model, name, value)
    splits = config["data"]["test_splits"]
    results["avg_stats"] = average_domains(results, splits)
    results["avg_without_sd15"] = average_domains(
        results, [split for split in splits if split != "sd15"]
    )
    runtime.print(f"Validation Avg: {json.dumps(results['avg_stats'], sort_keys=True)}")
    runtime.print(f"Validation OOD Avg: {json.dumps(results['avg_without_sd15'], sort_keys=True)}")
    return results


def train(config, runtime):
    dataset = build_dataset(config, config["data"]["train_split"], training=True, load_masks=False)
    validation = config["validation"]
    loader = build_loader(dataset, config["training"]["batch_size"], config, runtime, training=True)
    validation_loaders = {}
    for split in config["data"]["test_splits"]:
        validation_loaders[split] = {}
        for category in TEST_CATEGORIES:
            subset = build_dataset(
                config, split, classes=[category], load_masks=category != "real"
            )
            validation_loaders[split][category] = build_loader(
                subset, validation["batch_size"], config, runtime
            )
    resume_path = config["training"]["resume"]
    checkpoint = read_checkpoint(resume_path) if resume_path else None
    model = build_model(config, runtime, checkpoint)
    groups, names = model.get_param_groups(
        config["optimizer"]["lr"], config["optimizer"]["weight_decay"],
        separate_image_groups=separate_image_lr(config),
    )
    optimizer = torch.optim.AdamW(groups, betas=tuple(config["optimizer"]["betas"]))
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"])
    first_epoch, best_metric = 0, -1.0
    if checkpoint is not None:
        first_epoch, best_metric = resume_training(checkpoint, optimizer, scaler, config, runtime)
        del checkpoint
    if first_epoch >= config["training"]["epochs"]:
        raise ValueError("This checkpoint has already completed training. No epochs remain.")
    parallel_model = model
    if runtime.world_size > 1:
        ratio = config["model"]["contrast_top_ratio"]
        patches = config["model"]["mil_pool"] ** 2
        selected = model._resolve_k(patches, ratio)
        allow_unused = (config["training"]["batch_size"] // 2) * selected < 2
        parallel_model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=allow_unused,
        )
    _prepare_training_output(config, runtime)
    runtime.print(
        f"Train: {len(dataset)} images ({len(dataset.real_indices)} real, {len(dataset.fake_indices)} fake); "
        f"validation images by domain (pixel P/P+F; image=all): "
        f"{ {split: {kind: len(item.dataset) for kind, item in loaders.items()} for split, loaders in validation_loaders.items()} }."
    )
    effective_batch = (
        config["training"]["batch_size"]
        * runtime.world_size
        * config["training"]["accumulation_steps"]
    )
    runtime.print(
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters "
        f"in {len(names)} tensors; effective batch={effective_batch}; "
        f"head LR multiplier={config['model']['new_head_lr_mult']:g}."
    )
    if separate_image_lr(config):
        schedule = config["optimizer"]
        after = schedule["image_decay_after_epoch"]
        timing = (
            "no milestone drop"
            if after is None
            else f"one-time drop after {after} completed epochs (first affected epoch: {after + 1})"
        )
        runtime.print(
            "Image-only optimizer groups: CLS aggregator, image classifier, localization-to-image fusion. "
            f"LR scale={schedule['image_lr_scale']:g}; {timing}; "
            f"decay factor={schedule['image_decay_factor']:g}. "
            "Other groups retain their cosine schedule. "
            "This is an optimizer tuning setting, not a verified benchmark improvement."
        )
    if config["training"]["max_batches"] or validation["max_batches"]:
        runtime.print(
            "Limited-batch smoke run: metrics from this run are not full-dataset benchmark results."
        )
    start = time.monotonic()
    with WandbTracker(config, runtime, first_epoch=first_epoch) as tracker:
        _save_training_config(config, runtime)
        for epoch in range(first_epoch, config["training"]["epochs"]):
            loader.batch_sampler.set_epoch(epoch)
            loader.generator.manual_seed(
                config["runtime"]["seed"] + runtime.rank + epoch * runtime.world_size
            )
            stats = {
                "epoch": epoch + 1,
                "train": train_epoch(
                    parallel_model, loader, optimizer, scaler, epoch, config, runtime,
                    log_train=tracker.log_train if tracker.active else None,
                ),
            }
            stats["validation"] = validate_domains(model, validation_loaders, config, runtime)
            ood = stats["validation"]["avg_without_sd15"]
            if "pixel_p" not in ood:
                raise ValueError("Checkpoint selection requires at least one non-SD1.5 test split.")
            selected_f1 = float(ood["pixel_p"]["pixel_f1"])
            if not math.isfinite(selected_f1):
                raise ValueError("OOD Avg Pixel P F1 must be finite for checkpoint selection.")
            is_best = selected_f1 > best_metric
            if is_best:
                best_metric = selected_f1
            stats["best_ood_pixel_p_f1"] = best_metric
            if config["training"]["save_checkpoints"]:
                save_training_checkpoint(
                    model, optimizer, scaler, epoch, best_metric, config, runtime, is_best=is_best
                )
            if runtime.is_main:
                path = Path(config["paths"]["output_dir"]) / "history.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(stats, allow_nan=False) + "\n")
            tracker.log_epoch(stats)
            runtime.print(
                f"Completed epoch {epoch + 1}; OOD Avg Pixel P F1={selected_f1:.6f}, "
                f"best={best_metric:.6f}."
            )
    runtime.print(
        f"Training completed in {(time.monotonic() - start) / 3600:.2f} hours. Output: {config['paths']['output_dir']}"
    )


def test(config, runtime):
    checkpoint = read_checkpoint(config["paths"]["checkpoint"])
    model = build_model(config, runtime, checkpoint)
    del checkpoint
    settings = config["evaluation"]
    threshold = settings["threshold"]
    results = {}
    for split in config["data"]["test_splits"]:
        pixel_p = Metrics(runtime.device, pixel_mode="per_image", threshold=threshold)
        pixel_pf = Metrics(runtime.device, pixel_mode="entire_pixel", threshold=threshold)
        image = Metrics(runtime.device, image=True, threshold=threshold)
        metrics_by_class = {
            "partial": (pixel_p, image),
            "fully": (pixel_pf, image),
            "real": (image,),
        }
        for category in TEST_CATEGORIES:
            dataset = build_dataset(
                config, split, classes=[category], load_masks=category != "real"
            )
            loader = build_loader(dataset, settings["batch_size"], config, runtime)
            accumulate_metrics(model, loader, runtime, *metrics_by_class[category])
        pixel_pf.include_pixels(pixel_p)
        results[split] = {
            "pixel_p": pixel_p.compute(),
            "pixel_pf": pixel_pf.compute(),
            "image": image.compute(),
        }
        runtime.print(f"{split}: {json.dumps(results[split], sort_keys=True)}")
    split_names = list(config["data"]["test_splits"])
    results["avg_stats"] = average_domains(results, split_names)
    results["avg_without_sd15"] = average_domains(
        results, [name for name in split_names if name != "sd15"]
    )
    results["protocol"] = {
        "pixel_p": {"classes": ["partial"], "mode": "per_image"},
        "pixel_pf": {"classes": ["partial", "fully"], "mode": "entire_pixel"},
        "image_classes": list(TEST_CATEGORIES),
        "threshold": threshold,
        "model_calibration": {
            key: value for key, value in config["model"].items() if key.startswith("pred_calib_")
        },
    }
    if runtime.is_main:
        output = Path(config["paths"]["output_dir"])
        save_json(output / "resolved_config.json", config)
        save_json(output / "test_summary.json", results)
        runtime.print(json.dumps(results, indent=2))
        runtime.print(f"Saved {output / 'test_summary.json'}")
    return results


def check_inputs(config, *, inspect_data=False):
    mode, paths = config["mode"], config["paths"]
    if mode == "train" and not config["training"]["resume"]:
        for path, hint in (
            (paths["mae_checkpoint"], "See 'Pretrained models' in README.md for the MAE weight."),
            (
                str(Path(paths["clip_cache_dir"]) / "ViT-L-14.pt"),
                "See 'Pretrained models' in README.md for the CLIP weight.",
            ),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing pretrained weight: {path}. {hint}")
    else:
        path = config["training"]["resume"] if mode == "train" else paths["checkpoint"]
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path!r}. Set its path in JSON.")
    if not inspect_data:
        return
    requests = []
    if mode == "train":
        requests.append((config["data"]["train_split"], True, None, False))
        for split in config["data"]["test_splits"]:
            requests.append((split, False, ["partial"], True))
            requests.append((split, False, ["fully"], True))
            requests.append((split, False, ["real"], False))
    else:
        for split in config["data"]["test_splits"]:
            for category in TEST_CATEGORIES:
                requests.append((split, False, [category], category != "real"))
    from .data import reconstruction_path

    for split, training, classes, masks in requests:
        dataset = build_dataset(config, split, training=training, classes=classes, load_masks=masks)
        missing = []
        missing_count = 0
        for key in dataset.dataset["key"]:
            path = reconstruction_path(dataset.recon_map_dir, key)
            if not path.is_file():
                missing_count += 1
                if len(missing) < 3:
                    missing.append(str(path))
        if missing_count:
            raise FileNotFoundError(
                f"{split}: {missing_count} reconstruction maps are missing. Examples: {missing}"
            )
        indices = {0, len(dataset) - 1}
        for category in {"real", "fully", "partial"}:
            if category in dataset.categories:
                indices.add(dataset.categories.index(category))
        for index in sorted(indices):
            dataset[index]
        if training:
            BalancedBatchSampler(
                dataset,
                config["training"]["batch_size"],
                world_size=config["runtime"]["num_processes"],
            )
        print(
            f"{split} ({'train' if training else 'test'}, {sorted(dataset.classes)}): {len(dataset)} samples; every map exists; sampled tensor contracts OK.",
            flush=True,
        )
    print(
        "Preflight passed. This checks files and sampled data, not a full model forward or benchmark reproduction."
    )
