from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import load_opensdi_split
from .reconstruction import load_reconstruction_map, reconstruction_path
from .transforms import SampleTransform


VALID_CLASSES = frozenset({"real", "fully", "partial"})


def sample_class(key, label):
    if int(label) not in {0, 1}:
        raise ValueError(f"Expected a binary image label, got {label!r}.")
    if int(label) == 0:
        return "real"
    return "partial" if "partial" in str(key) and "fake" in str(key) else "fully"


class OpenSDIDataset(Dataset):
    def __init__(
        self,
        repo_id,
        split,
        cache_dir,
        *,
        recon_root,
        image_size=512,
        augmentation=False,
        classes=None,
        data_scale=1.0,
        seed=42,
        local_files_only=True,
        recon_alias="sd1_v1",
        recon_name="vae_l1_rgbmean_zreal_128",
        recon_dtype="float32",
        load_masks=True,
        cache_config=None,
        prepared_dir=None,
    ):
        if not (0 < data_scale <= 1):
            raise ValueError("data_scale must be in (0, 1].")
        classes = VALID_CLASSES if classes is None else frozenset(classes)
        if not classes or not classes.issubset(VALID_CLASSES):
            raise ValueError("classes must be a nonempty subset of real, fully, partial.")
        dataset = load_opensdi_split(
            repo_id,
            split,
            cache_dir,
            local_files_only=local_files_only,
            cache_config=cache_config,
            prepared_dir=prepared_dir,
        )
        missing = {"image", "label", "key"} - set(dataset.column_names)
        if missing:
            raise ValueError(f"Dataset {repo_id}:{split} is missing columns: {sorted(missing)}")
        keys, labels = dataset["key"], dataset["label"]
        categories = [sample_class(key, label) for key, label in zip(keys, labels)]
        selected = [index for index, category in enumerate(categories) if category in classes]
        if data_scale < 1:
            rng = np.random.RandomState(int(seed))
            sampled = []
            for category in ("real", "fully", "partial"):
                indices = [index for index in selected if categories[index] == category]
                if indices:
                    count = max(1, int(np.floor(len(indices) * data_scale)))
                    sampled.extend(rng.choice(indices, size=count, replace=False).tolist())
            selected = sorted(sampled)
        if not selected:
            raise ValueError(f"No samples match classes={sorted(classes)} in {repo_id}:{split}.")
        self.labels = [int(labels[index]) for index in selected]
        self.categories = [categories[index] for index in selected]
        self.dataset = dataset.select(selected)
        self.load_masks = bool(load_masks)
        if not self.load_masks and "mask" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("mask")
        self.real_indices = [index for index, label in enumerate(self.labels) if label == 0]
        self.fake_indices = [index for index, label in enumerate(self.labels) if label == 1]
        self.classes = classes
        self.image_size = int(image_size)
        for name, value in (("recon_alias", recon_alias), ("recon_name", recon_name)):
            if (
                not isinstance(value, str)
                or not value
                or any(c in value for c in "/\\")
                or value in {".", ".."}
            ):
                raise ValueError(f"{name} must be a single nonempty directory name.")
        self.recon_map_dir = Path(recon_root).expanduser().resolve() / recon_alias / recon_name
        self.recon_dtype = recon_dtype
        self.transform = SampleTransform(image_size=image_size, augmentation=augmentation)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample["image"].convert("RGB")
        label = self.labels[index]
        mask = None
        if self.load_masks:
            height, width = image.height, image.width
            if label == 0:
                mask = np.zeros((height, width), dtype=np.float32)
            elif sample.get("mask") is None:
                if self.categories[index] == "partial":
                    raise ValueError(f"Partial image {sample['key']!r} has no evaluation mask.")
                mask = np.ones((height, width), dtype=np.float32)
            else:
                rgb_mask = np.asarray(sample["mask"].convert("RGB"))
                mask = (rgb_mask.mean(axis=2) > 127.5).astype(np.float32)
        recon_map = load_reconstruction_map(
            reconstruction_path(self.recon_map_dir, sample["key"]), self.recon_dtype
        )
        result = self.transform(image, recon_map, mask=mask)
        result.update(
            label=label,
            name=sample["key"],
            shape=torch.tensor((self.image_size, self.image_size)),
        )
        return result

    def __len__(self):
        return len(self.dataset)
