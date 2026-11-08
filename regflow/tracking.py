import copy
import json
import math
import random
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch.distributed as dist

from .config import save_json


DOMAIN_NAMES = {
    "sd15": "01_SD1.5",
    "sd2": "02_SD2.1",
    "sdxl": "03_SDXL",
    "sd3": "04_SD3",
    "flux": "05_FLUX.1",
    "avg_stats": "06_AVG",
    "avg_without_sd15": "07_OOD_Avg",
}
VALIDATION_SECTIONS = ("val_pixel_P", "val_pixel_PF", "val_image")
LOSS_WEIGHTS = {
    "loss_ce": "ce_w",
    "loss_rank": "rank_w",
    "loss_real_hardneg": "real_hardneg_w",
    "loss_peak": "peak_w",
    "loss_contrast": "contrast_w",
    "loss_smooth": "smooth_w",
    "loss_sparse": "sparse_w",
}


@contextmanager
def _preserve_python_rng():
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


def _finite_scalars(values):
    if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("W&B metrics must be finite numbers.")
    return {key: float(value) for key, value in values.items()}


def train_scalars(losses, config, *, epoch, step, steps):
    if steps <= 0 or not 1 <= step <= steps or epoch < 0:
        raise ValueError("Training progress requires epoch >= 0 and 1 <= step <= steps.")
    values = {
        "epoch": epoch + step / steps,
        "global_step": epoch * steps + step,
        "train/loss": losses["combined_loss"],
    }
    for loss, weight in LOSS_WEIGHTS.items():
        if config["model"][weight] > 0 and loss in losses:
            values[f"train/{loss}"] = losses[loss]
    return _finite_scalars(values)


def epoch_scalars(stats):
    values = {"epoch": stats["epoch"]}
    results = stats["validation"]
    ordered = [split for split in DOMAIN_NAMES if split in results]
    extras = sorted(set(results) - set(DOMAIN_NAMES))
    labels = {**DOMAIN_NAMES, **{split: f"{index:02d}_{split}" for index, split in enumerate(extras, 8)}}
    for section, kind, metric in (
        ("val_pixel_P", "pixel_p", "pixel_f1"),
        ("val_pixel_PF", "pixel_pf", "entire_pixel_f1"),
        ("val_image", "image", "image_f1"),
    ):
        for split in (*ordered, *extras):
            values[f"{section}/{labels[split]}_F1"] = results[split][kind][metric]
    return _finite_scalars(values)


class WandbTracker:
    def __init__(self, config, runtime, *, first_epoch=0):
        self.config = config
        self.runtime = runtime
        self.settings = config["wandb"]
        self.active = self.settings["enabled"] and self.settings["mode"] != "disabled"
        self.first_epoch = first_epoch
        self.run = None
        self.metadata = None
        self.path = Path(config["paths"]["output_dir"]) / "wandb_run.json"

    def _on_main(self, action, phase):
        errors = [None]
        if self.runtime.is_main:
            try:
                with _preserve_python_rng():
                    action()
            except Exception as error:
                errors[0] = f"{type(error).__name__}: {error}"
        if dist.is_initialized():
            dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise RuntimeError(f"W&B {phase} failed: {errors[0]}")

    def _previous_run(self):
        if not self.path.exists():
            return None
        resume = self.config["training"]["resume"]
        same_output = resume and Path(resume).resolve().parent == self.path.parent.resolve()
        if not same_output:
            raise ValueError(
                "This output already contains a W&B run. Choose a new output directory."
            )
        previous = json.loads(self.path.read_text(encoding="utf-8"))
        required = {"id", "mode", "project", "entity", "last_epoch"}
        if not isinstance(previous, dict) or required - previous.keys():
            raise ValueError("Invalid wandb_run.json; restore it or use a new output directory.")
        for key in ("mode", "project"):
            if previous[key] != self.settings[key]:
                raise ValueError(f"W&B {key} changed on resume. Use a new output directory.")
        if self.settings["entity"] is not None and previous["entity"] != self.settings["entity"]:
            raise ValueError("W&B entity changed on resume. Use a new output directory.")
        if self.settings["mode"] == "online" and previous["last_epoch"] > self.first_epoch:
            raise ValueError(
                "This checkpoint predates the W&B history. Resume the latest checkpoint, "
                "or choose a new output directory to start a separate run."
            )
        return previous

    def _start(self):
        previous = self._previous_run()
        try:
            import wandb
        except ImportError as error:
            raise ImportError("Install requirements.txt to enable W&B logging.") from error
        self.path.parent.mkdir(parents=True, exist_ok=True)
        online_resume = previous is not None and self.settings["mode"] == "online"
        parameters = {
            "project": self.settings["project"],
            "entity": previous["entity"] if online_resume else self.settings["entity"],
            "name": self.settings["name"],
            "group": self.settings["group"],
            "tags": self.settings["tags"],
            "mode": self.settings["mode"],
            "job_type": "train",
            "dir": str(self.path.parent),
            "config": copy.deepcopy(self.config),
            "allow_val_change": True,
            "id": previous["id"] if online_resume else uuid.uuid4().hex,
            "settings": wandb.Settings(
                console="off",
                disable_code=True,
                disable_git=True,
                save_code=False,
                x_disable_stats=True,
                x_disable_meta=True,
                x_save_requirements=False,
                init_timeout=60,
                login_timeout=30,
            ),
        }
        if self.settings["mode"] == "online":
            parameters["resume"] = "must" if online_resume else "never"
        elif previous is not None:
            self.runtime.print("W&B offline resume starts a separate run segment with a new ID.")
        self.run = wandb.init(**parameters)
        self.run.define_metric("epoch", hidden=True)
        self.run.define_metric("global_step", hidden=True)
        self.run.define_metric("train/*", step_metric="epoch", step_sync=False)
        for section in VALIDATION_SECTIONS:
            self.run.define_metric(f"{section}/*", step_metric="epoch", step_sync=False)
            for label in DOMAIN_NAMES.values():
                self.run.define_metric(
                    f"{section}/{label}_F1", step_metric="epoch", step_sync=False
                )
        self.metadata = {
            "id": self.run.id,
            "mode": self.settings["mode"],
            "project": self.settings["project"],
            "entity": self.run.entity or self.settings["entity"],
            "last_epoch": previous["last_epoch"] if online_resume else self.first_epoch,
            "last_train_epoch": (
                previous.get("last_train_epoch", previous["last_epoch"])
                if online_resume
                else self.first_epoch
            ),
        }
        if previous is not None and not online_resume:
            self.metadata["previous_segment_id"] = previous["id"]
        save_json(self.path, self.metadata)
        self.runtime.print(
            f"W&B: {self.settings['project']} / {self.run.id} ({self.settings['mode']})."
        )

    def __enter__(self):
        if self.active:
            try:
                self._on_main(self._start, "initialization")
            except Exception:
                self._finish_after_failure()
                raise
        return self

    def log_train(self, losses, *, epoch, step, steps):
        if not self.active:
            return

        def log():
            values = train_scalars(losses, self.config, epoch=epoch, step=step, steps=steps)
            if values["epoch"] <= self.metadata["last_train_epoch"]:
                return
            self.run.log(values)
            self.metadata["last_train_epoch"] = values["epoch"]
            save_json(self.path, self.metadata)

        self._on_main(log, "training loss logging")

    def log_epoch(self, stats):
        if not self.active:
            return

        def log():
            values = epoch_scalars(stats)
            if values["epoch"] <= self.metadata["last_epoch"]:
                raise ValueError("Epochs must increase within one W&B run.")
            self.run.log(values)
            self.metadata["last_epoch"] = stats["epoch"]
            save_json(self.path, self.metadata)

        self._on_main(log, "epoch logging")

    def _finish_after_failure(self):
        if self.runtime.is_main and self.run is not None:
            try:
                with _preserve_python_rng():
                    self.run.finish(exit_code=1)
            except Exception as error:
                self.runtime.print(f"W&B cleanup also failed: {error}")
            finally:
                self.run = None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            if exc_type is not None:
                self._finish_after_failure()
            else:
                try:
                    self._on_main(lambda: self.run.finish(exit_code=0), "finalization")
                finally:
                    self.run = None
        return False
