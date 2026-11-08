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

The default cache location is `./cache/huggingface/datasets`. For evaluation alone, use `--config configs/test.json` to prepare the test dataset.

## Reuse a prepared cache

Preserve the complete Hugging Face Arrow cache hierarchy, including `dataset_info.json` and all shards. The default training and test loaders read prepared files offline.

Keep each OpenSDI `key` intact. ReGFLoW uses it to pair the image with its reconstruction map.

## Configure data paths

Set the following fields in [`configs/base.json`](../configs/base.json), which is shared by training and evaluation. Paths are resolved relative to the repository root.

| Field | What to set |
| --- | --- |
| `paths.hf_home` | Hugging Face cache root; default `./cache/huggingface`. |
| `data.cache_dir` | Prepared dataset cache root. Leave `null` to use `<paths.hf_home>/datasets`. |
| `data.train_prepared_dir`, `data.test_prepared_dir` | Optional paths to specific prepared cache directories containing `dataset_info.json`, if multiple versions are available. |

Once the cache is ready, continue with [Data Preprocessing](../data_preprocessing/README.md).
