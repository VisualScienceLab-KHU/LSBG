"""Prepare the OpenSDI Arrow caches used by ReGFLoW."""

import argparse
from pathlib import Path

from regflow.config import configure_environment, load_config


def configured_datasets(config):
    data = config["data"]
    if config["mode"] == "train":
        yield data["train_repo_id"], [data["train_split"]], data["train_prepared_dir"]
    yield data["test_repo_id"], data["test_splits"], data["test_prepared_dir"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    config = load_config(parser.parse_args().config)
    configure_environment(config)

    from datasets import load_dataset
    from regflow.data import load_opensdi_split

    data = config["data"]
    for repo, splits, prepared_dir in configured_datasets(config):
        if prepared_dir is None:
            load_dataset(repo, name=data["cache_config"], cache_dir=data["cache_dir"])
        for split in splits:
            dataset = load_opensdi_split(
                repo, split, data["cache_dir"], local_files_only=True,
                cache_config=data["cache_config"], prepared_dir=prepared_dir,
            )
            directory = Path(dataset.cache_files[0]["filename"]).parent
            print(f"{repo}/{split}: {len(dataset):,} samples ({directory})", flush=True)


if __name__ == "__main__":
    main()
