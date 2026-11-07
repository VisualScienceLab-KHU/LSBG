import json
from pathlib import Path

from datasets import Dataset, DatasetInfo, DownloadConfig, concatenate_datasets, load_dataset
from datasets.naming import camelcase_to_snakecase, filenames_for_dataset_split


def _prepared_files(directory, split, expected_name):
    directory = Path(directory).expanduser().resolve()
    metadata_path = directory / "dataset_info.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Hugging Face cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_name = metadata.get("dataset_name")
    if dataset_name != expected_name:
        raise ValueError(
            f"Cache {directory} belongs to {dataset_name!r}; expected {expected_name!r}."
        )
    split_info = metadata.get("splits", {}).get(split)
    if split_info is None:
        raise ValueError(f"Split {split!r} is not prepared in {directory}.")
    filenames = filenames_for_dataset_split(
        str(directory),
        dataset_name=dataset_name,
        split=split,
        filetype_suffix="arrow",
        shard_lengths=split_info.get("shard_lengths"),
    )
    missing = [filename for filename in filenames if not Path(filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Arrow cache; missing shard: {missing[0]}")
    return directory, metadata, filenames


def load_opensdi_split(
    repo_id,
    split,
    cache_dir,
    *,
    local_files_only=True,
    cache_config=None,
    prepared_dir=None,
):
    if not isinstance(repo_id, str) or len(repo_id.split("/")) != 2:
        raise ValueError("repo_id must have the form 'owner/dataset'.")
    if not isinstance(split, str) or not split or any(c in split for c in "/\\[]+:"):
        raise ValueError("split must be one complete split name, such as 'sd15'.")
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not local_files_only and prepared_dir is None:
        return load_dataset(
            repo_id,
            name=cache_config,
            split=split,
            cache_dir=str(cache_dir),
            download_config=DownloadConfig(local_files_only=False),
            download_mode="reuse_dataset_if_exists",
        )

    owner, name = repo_id.split("/")
    normalized_name = camelcase_to_snakecase(name)
    if prepared_dir is not None:
        directory, metadata, filenames = _prepared_files(prepared_dir, split, normalized_name)
    else:
        repo_cache = cache_dir / f"{owner}___{normalized_name}"
        candidates = []
        errors = []
        for metadata_path in sorted(repo_cache.glob("*/*/*/dataset_info.json")):
            try:
                record = _prepared_files(metadata_path.parent, split, normalized_name)
                if cache_config is None or record[1].get("config_name") == cache_config:
                    candidates.append(record)
            except (ValueError, OSError, KeyError, TypeError) as error:
                errors.append(str(error))
        if not candidates:
            detail = f"\n{errors[0]}" if errors else ""
            raise FileNotFoundError(
                f"No complete prepared Arrow cache for {repo_id}:{split} in {repo_cache}. "
                "Prepare it with datasets.load_dataset as shown in README.md, or set "
                "prepared_dir to its directory containing dataset_info.json." + detail
            )
        if len(candidates) > 1:
            directories = "\n".join(str(item[0]) for item in candidates)
            raise ValueError(
                f"Multiple prepared caches match {repo_id}:{split}. Set prepared_dir "
                f"explicitly instead of selecting a revision implicitly:\n{directories}"
            )
        directory, metadata, filenames = candidates[0]

    info = DatasetInfo.from_directory(str(directory))
    shards = [Dataset.from_file(filename, info=info, split=split) for filename in filenames]
    dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    expected_rows = metadata["splits"][split]["num_examples"]
    if len(dataset) != expected_rows:
        raise ValueError(
            f"Incomplete cache for {repo_id}:{split}: metadata specifies {expected_rows} "
            f"rows, but Arrow shards contain {len(dataset)}."
        )
    return dataset
