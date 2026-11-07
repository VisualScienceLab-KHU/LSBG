import copy
import json
import math
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECTION_KEYS = {
    "paths": {"hf_home", "mae_checkpoint", "clip_cache_dir", "output_dir", "checkpoint"},
    "runtime": {"device", "num_processes", "num_workers", "pin_memory", "seed", "log_every"},
    "data": {
        "image_size",
        "cache_dir",
        "cache_config",
        "local_files_only",
        "train_repo_id",
        "train_split",
        "test_repo_id",
        "test_splits",
        "train_prepared_dir",
        "test_prepared_dir",
    },
    "reconstruction": {"root", "alias", "name", "dtype"},
    "training": {
        "epochs",
        "batch_size",
        "accumulation_steps",
        "augmentation",
        "data_scale",
        "amp",
        "resume",
        "max_batches",
        "save_checkpoints",
        "save_every",
    },
    "optimizer": {
        "lr", "min_lr", "weight_decay", "betas", "warmup_epochs",
        "image_lr_scale", "image_decay_after_epoch", "image_decay_factor",
    },
    "validation": {"batch_size", "max_batches", "threshold", "selection_split", "model_overrides"},
    "evaluation": {"batch_size", "pixel_classes", "run_image_eval", "threshold", "max_batches"},
    "wandb": {"enabled", "mode", "project", "entity", "name", "group", "tags"},
}


IMAGE_OPTIMIZER_DEFAULTS = {
    "image_lr_scale": 1.0,
    "image_decay_after_epoch": None,
    "image_decay_factor": 1.0,
}


def optimizer_with_defaults(settings):
    return {**IMAGE_OPTIMIZER_DEFAULTS, **settings}


def separate_image_lr(config):
    settings = optimizer_with_defaults(config["optimizer"])
    return settings["image_lr_scale"] != 1.0 or (
        settings["image_decay_after_epoch"] is not None
        and settings["image_decay_factor"] != 1.0
    )


def _merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_json(path, stack=()):
    path = path.expanduser().resolve()
    if path in stack:
        raise ValueError(f"Circular config inheritance: {path}")
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    parent = config.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ValueError("extends must be a relative or absolute JSON path.")
        config = _merge(_read_json(path.parent / parent, (*stack, path)), config)
    return config


def _path(value, allow_empty=False):
    if value is None or value == "":
        if allow_empty:
            return ""
        raise ValueError("A required path is empty.")
    if not isinstance(value, str):
        raise ValueError("Paths must be strings or null for documented defaults.")
    expanded = Path(os.path.expandvars(value)).expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return str(expanded.resolve())


def _number(value, name, minimum=0, integer=False, strictly=False):
    expected = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'}.")
    if value < minimum or (strictly and value == minimum):
        symbol = ">" if strictly else ">="
        raise ValueError(f"{name} must be {symbol} {minimum}.")


def validate_config(config):
    expected_top = set(SECTION_KEYS) | {"model", "mode"}
    unknown = set(config) - expected_top
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    mode = config.get("mode")
    if mode not in {"train", "test"}:
        raise ValueError("mode must be train or test.")
    required = {"paths", "runtime", "data", "reconstruction", "model"}
    required |= {
        "train": {"training", "optimizer", "validation", "wandb"},
        "test": {"evaluation"},
    }[mode]
    if required - set(config):
        raise ValueError(f"Missing config sections: {sorted(required - set(config))}")
    for section, allowed in SECTION_KEYS.items():
        if section not in config:
            continue
        values = config[section]
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a JSON object.")
        if set(values) - allowed:
            raise ValueError(f"Unknown {section} keys: {sorted(set(values) - allowed)}")
        required_keys = allowed - (set(IMAGE_OPTIMIZER_DEFAULTS) if section == "optimizer" else set())
        if required_keys - set(values):
            raise ValueError(f"Missing {section} keys: {sorted(required_keys - set(values))}")
    common = _read_json(PROJECT_ROOT / "configs" / "base.json")
    model_keys = set(common["model"])
    if not isinstance(config["model"], dict) or set(config["model"]) != model_keys:
        raise ValueError(
            "model keys must match configs/base.json; check for a missing key or typo."
        )
    runtime, data, model = config["runtime"], config["data"], config["model"]
    if runtime["device"] not in {"cuda", "cpu"}:
        raise ValueError("runtime.device must be cuda or cpu.")
    for key in ("num_processes", "log_every"):
        _number(runtime[key], f"runtime.{key}", integer=True, strictly=True)
    for key in ("num_workers", "seed"):
        _number(runtime[key], f"runtime.{key}", integer=True)
    for section, keys in ((runtime, ("pin_memory",)), (data, ("local_files_only",))):
        if any(not isinstance(section[key], bool) for key in keys):
            raise ValueError("Use JSON true/false for boolean options.")
    if data["image_size"] != 512 or isinstance(data["image_size"], bool):
        raise ValueError("This released ViT-L/14 + MAE configuration requires image_size=512.")
    if not isinstance(data["test_splits"], list) or not data["test_splits"]:
        raise ValueError("data.test_splits must be a nonempty list.")
    if any(not isinstance(split, str) or not split for split in data["test_splits"]):
        raise ValueError("Every test split must be a nonempty string.")
    if len(set(data["test_splits"])) != len(data["test_splits"]):
        raise ValueError("data.test_splits must not contain duplicates.")
    if data["cache_config"] is not None and (
        not isinstance(data["cache_config"], str) or not data["cache_config"]
    ):
        raise ValueError("data.cache_config must be null or a nonempty string.")
    for key in ("train_repo_id", "test_repo_id", "train_split"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"data.{key} must be a nonempty string.")
    for key in ("alias", "name"):
        value = config["reconstruction"][key]
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or any(char in value for char in "/\\")
        ):
            raise ValueError(f"reconstruction.{key} must be a single directory name.")
    if config["reconstruction"]["dtype"] not in {"float16", "float32"}:
        raise ValueError("reconstruction.dtype must be float16 or float32.")
    if model["model_setting_name"] != "ViTL" or model["mil_pool"] not in (8, 16):
        raise ValueError("Use model_setting_name=ViTL and mil_pool=8 or 16.")
    for key, value in model.items():
        if key == "model_setting_name":
            continue
        _number(value, f"model.{key}")
    for key in (
        "use_imagewise_pred_calibration",
        "use_loc_to_img",
        "loc_to_img_detach_score",
        "loc_to_img_detach_feature",
        "loc_to_img_use_score_stats",
        "use_recon_fusion",
        "use_prior_logit_bias",
    ):
        if model[key] not in (0, 1):
            raise ValueError(f"model.{key} must be 0 or 1.")
    for key in (
        "score_hidden_dim",
        "embed_dim",
        "image_hidden_dim",
        "loc_to_img_hidden_dim",
        "recon_dim",
        "recon_proj_dim",
        "rgb_proj_dim",
        "recon_bias_hidden_dim",
        "recon_bias_recon_hidden_dim",
    ):
        _number(model[key], f"model.{key}", integer=True, strictly=True)
    for key in ("contrast_temp", "pred_calib_eps", "new_head_lr_mult", "loc_to_img_attn_temp"):
        _number(model[key], f"model.{key}", strictly=True)
    for key in (
        "contrast_top_ratio",
        "real_hardneg_ratio",
        "peak_top_ratio",
        "loc_to_img_top_ratio",
    ):
        if not 0 < model[key] <= 1:
            raise ValueError(f"model.{key} must be in (0, 1].")
    for key in ("image_dropout", "loc_to_img_dropout", "recon_dropout"):
        if not 0 <= model[key] < 1:
            raise ValueError(f"model.{key} must be in [0, 1).")
    if not 0 <= model["peak_rest_min_ratio"] < model["peak_rest_max_ratio"] <= 1:
        raise ValueError("peak_rest_min_ratio must be smaller than peak_rest_max_ratio in [0, 1].")
    if model["pred_calib_k_min"] > model["pred_calib_k_max"]:
        raise ValueError("pred_calib_k_min must not exceed pred_calib_k_max.")
    if model["use_prior_logit_bias"] and not model["use_recon_fusion"]:
        raise ValueError("use_prior_logit_bias requires use_recon_fusion=1.")
    if mode == "train":
        training, optimizer, validation = (
            config[key] for key in ("training", "optimizer", "validation")
        )
        for key in ("epochs", "batch_size", "accumulation_steps"):
            _number(training[key], f"training.{key}", integer=True, strictly=True)
        if training["batch_size"] % 2:
            raise ValueError("training.batch_size must be even for real/fake balanced batches.")
        for key in ("augmentation", "amp", "save_checkpoints"):
            if not isinstance(training[key], bool):
                raise ValueError(f"training.{key} must be true or false.")
        _number(training["data_scale"], "training.data_scale", strictly=True)
        if training["data_scale"] > 1:
            raise ValueError("training.data_scale is a fraction in (0, 1].")
        for key in ("max_batches", "save_every"):
            _number(training[key], f"training.{key}", integer=True)
        for key in ("lr", "min_lr", "weight_decay", "warmup_epochs"):
            _number(optimizer[key], f"optimizer.{key}")
        if optimizer["lr"] <= 0 or optimizer["min_lr"] > optimizer["lr"]:
            raise ValueError("Use lr > 0 and min_lr <= lr.")
        if optimizer["warmup_epochs"] >= training["epochs"]:
            raise ValueError("optimizer.warmup_epochs must be smaller than training.epochs.")
        image_schedule = optimizer_with_defaults(optimizer)
        for key in ("image_lr_scale", "image_decay_factor"):
            _number(image_schedule[key], f"optimizer.{key}", strictly=True)
        if image_schedule["image_decay_factor"] > 1:
            raise ValueError("optimizer.image_decay_factor must be in (0, 1].")
        after = image_schedule["image_decay_after_epoch"]
        if after is not None:
            _number(after, "optimizer.image_decay_after_epoch", integer=True, strictly=True)
            if after >= training["epochs"]:
                raise ValueError(
                    "optimizer.image_decay_after_epoch must be smaller than training.epochs; "
                    "it counts completed epochs. Use null to disable the image LR drop."
                )
        if not isinstance(optimizer["betas"], list) or len(optimizer["betas"]) != 2:
            raise ValueError("optimizer.betas must contain two values in [0, 1).")
        for beta in optimizer["betas"]:
            _number(beta, "optimizer.betas")
            if beta >= 1:
                raise ValueError("optimizer.betas must contain two values in [0, 1).")
        selection = validation["selection_split"]
        if not isinstance(selection, str) or selection not in data["test_splits"]:
            raise ValueError("validation.selection_split must be included in data.test_splits.")
        _number(validation["batch_size"], "validation.batch_size", integer=True, strictly=True)
        _number(validation["max_batches"], "validation.max_batches", integer=True)
        overrides = validation["model_overrides"]
        calibration_keys = {key for key in model if key.startswith("pred_calib_")} | {
            "use_imagewise_pred_calibration"
        }
        if not isinstance(overrides, dict) or set(overrides) - calibration_keys:
            raise ValueError(
                "validation.model_overrides may contain only prediction calibration keys."
            )
        for key, value in overrides.items():
            _number(value, f"validation.model_overrides.{key}")
        calibrated = {**model, **overrides}
        if calibrated["use_imagewise_pred_calibration"] not in (0, 1):
            raise ValueError(
                "validation calibration must use use_imagewise_pred_calibration=0 or 1."
            )
        if calibrated["pred_calib_eps"] <= 0:
            raise ValueError("validation calibration requires pred_calib_eps > 0.")
        if calibrated["pred_calib_k_min"] > calibrated["pred_calib_k_max"]:
            raise ValueError(
                "validation calibration requires pred_calib_k_min <= pred_calib_k_max."
            )
    if "wandb" in config:
        tracking = config["wandb"]
        if not isinstance(tracking["enabled"], bool):
            raise ValueError("wandb.enabled must be true or false.")
        if not isinstance(tracking["mode"], str) or tracking["mode"] not in {
            "online",
            "offline",
            "disabled",
        }:
            raise ValueError("wandb.mode must be online, offline or disabled.")
        if not isinstance(tracking["project"], str) or not tracking["project"].strip():
            raise ValueError("wandb.project must be a nonempty string.")
        for key in ("entity", "name", "group"):
            value = tracking[key]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"wandb.{key} must be null or a nonempty string.")
        tags = tracking["tags"]
        if (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
            or len(set(tags)) != len(tags)
        ):
            raise ValueError("wandb.tags must be a list of unique nonempty strings.")
    if mode == "test":
        evaluation = config["evaluation"]
        classes = evaluation["pixel_classes"]
        if (
            not isinstance(classes, list)
            or not classes
            or any(not isinstance(item, str) for item in classes)
        ):
            raise ValueError("evaluation.pixel_classes must be a nonempty list of class names.")
        if len(set(classes)) != len(classes) or set(classes) - {"partial", "fully", "real"}:
            raise ValueError(
                "evaluation.pixel_classes must select partial, fully and/or real without duplicates."
            )
        _number(evaluation["batch_size"], "evaluation.batch_size", integer=True, strictly=True)
        _number(evaluation["max_batches"], "evaluation.max_batches", integer=True)
        if not isinstance(evaluation["run_image_eval"], bool):
            raise ValueError("evaluation.run_image_eval must be true or false.")
    for section in ("validation", "evaluation"):
        if section in config:
            _number(config[section]["threshold"], f"{section}.threshold")
            if config[section]["threshold"] > 1:
                raise ValueError(f"{section}.threshold must be in [0, 1].")


def load_config(path, expected_mode=None):
    config = _read_json(Path(path))
    if isinstance(config.get("optimizer"), dict):
        config["optimizer"] = optimizer_with_defaults(config["optimizer"])
    validate_config(config)
    if expected_mode is not None and config["mode"] != expected_mode:
        raise ValueError(f"This entry point requires mode={expected_mode}, got {config['mode']}.")
    paths, data, recon = config["paths"], config["data"], config["reconstruction"]
    paths["hf_home"] = _path(paths["hf_home"])
    paths["mae_checkpoint"] = _path(paths["mae_checkpoint"])
    paths["clip_cache_dir"] = _path(paths["clip_cache_dir"] or "./weights/clip")
    paths["output_dir"] = _path(paths["output_dir"])
    paths["checkpoint"] = _path(paths["checkpoint"], allow_empty=True)
    data["cache_dir"] = _path(data["cache_dir"] or str(Path(paths["hf_home"]) / "datasets"))
    for key in ("train_prepared_dir", "test_prepared_dir"):
        data[key] = _path(data[key], allow_empty=True) or None
    recon["root"] = _path(recon["root"] or str(Path(paths["hf_home"]) / "recon_maps"))
    if "training" in config:
        config["training"]["resume"] = _path(config["training"]["resume"], allow_empty=True)
    return config


def configure_environment(config):
    paths = config["paths"]
    os.environ["HF_HOME"] = paths["hf_home"]
    os.environ["HF_DATASETS_CACHE"] = config["data"]["cache_dir"]
    os.environ["HF_HUB_CACHE"] = str(Path(paths["hf_home"]) / "hub")
    os.environ["CLIP_CACHE_DIR"] = paths["clip_cache_dir"]


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
