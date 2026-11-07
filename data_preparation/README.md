# Data Preparation

ReGFLoW uses OpenSDI. Prepare the `sd15` training split and the five test splits listed below. Training uses the binary image label. The evaluation loader also reads masks for localization. [OpenSDI paper and dataset repository](https://github.com/iamwangyabin/OpenSDI)

| Dataset | Splits |
| --- | --- |
| [OpenSDI train](https://huggingface.co/datasets/nebula/OpenSDI_train) | `sd15` |
| [OpenSDI test](https://huggingface.co/datasets/nebula/OpenSDI_test) | `sd15`, `sd2`, `sdxl`, `sd3`, `flux` |

## Prepare a new cache

Activate the project environment and run this command from the repository root with network access. It downloads the datasets, prepares the Hugging Face Arrow cache, and prints the row count and cache directory for each split. Allow space for both the download and the prepared cache.

```bash
python -m data_preparation.prepare_cache --config configs/train.json
```

The default location is `./cache/huggingface/datasets`. To use another disk, set `data.cache_dir` in the shared configuration before running the command. Relative paths are resolved from the repository root. For evaluation alone, use `--config configs/test.json` to prepare the test dataset. [Base configuration](../configs/base.json)

## Reuse a prepared cache

Point `data.cache_dir` to the cache root and preserve the directory hierarchy, including `dataset_info.json` and all Arrow shards. If several versions are present, set `data.train_prepared_dir` and `data.test_prepared_dir` to the respective directories containing `dataset_info.json`. The default training and test loaders read these files offline.

Keep each OpenSDI `key` intact. ReGFLoW uses it to pair the image with its reconstruction map. Continue with [Data Preprocessing](../data_preprocessing/README.md).
