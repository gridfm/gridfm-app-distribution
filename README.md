# GridFM App Distribution

GridFM Distribution appkit`gridfm-app-distribution` is a Python library to train, finetune, and predict a foundation model of an electrical power grid. 
Specialised training tasks can be developed for specific distribution applications.  


# Installation

Create and activate a virtual environment (make sure you use the right python version = 3.10, 3.11, 3.12 or 3.13.)
```
python -m venv venv
source venv/bin/activate
```

Install gridfm-app-distribution from PyPI
```
pip install gridfm-app-distribution
```

**`torch-scatter` is a required dependency.** It cannot be bundled in `pyproject.toml` because the correct wheel depends on your PyTorch and CUDA versions, so it must be installed separately.

Get PyTorch + CUDA version for torch-scatter
```
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.__version__ + ('+cpu' if torch.version.cuda is None else ''))")
```

Install the correct torch-scatter wheel
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
```


For documentation generation and unit testing, install with the optional `dev` and `test` extras:

```
pip install "gridfm-app-distribution[dev,test]"
```


# CLI commands

Interface to train, fine-tune, evaluate, and run inference on GridFM models of distribution networks using YAML configs and MLflow tracking.

```
gridfm_dist_app <command> [OPTIONS]
```

Available commands:

* `train` - Train a new model from scratch
* `finetune` - Fine-tune an existing pre-trained model
* `evaluate` - Evaluate model performance on a dataset
* `predict` - Run inference and save predictions

---

## Training Models

```
gridfm_dist_app --config path/to/config.yaml
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Path to the training configuration YAML file. | `None` |
| `--exp_name` | `str` | MLflow experiment name. | `timestamp` |
| `--run_name` | `str` | MLflow run name. | `run` |
| `--log_dir` | `str` | MLflow tracking/logging directory. | `mlruns` |
| `--data_path` | `str` | Root dataset directory. | `data` |
| `--compile [MODE]` | `str` | Enable `torch.compile` mode. Valid values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. If flag is passed without a value, mode is `default`. | `None` |
| `--bfloat16` | `flag` | Cast model to `torch.bfloat16` (`model.to(torch.bfloat16)`). | `False` |
| `--tf32` | `flag` | Enable TF32 on Ampere+ GPUs via `torch.set_float32_matmul_precision("high")`. | `False` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration, e.g. `gridfm_graphkit_ee`. | `[]` |
| `--num_workers` | `int` | Override `data.workers` from YAML. Use `0` to debug worker crashes. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Disk cache directory for dataset wrapper; cache is loaded from here when present and saved after first population. | `None` |
| `--profiler` | `str` | Enable Lightning profiler (`simple`, `advanced`, `pytorch`). | `None` |
| `--compute_dc_ac_metrics` | `flag` | Compute ground-truth AC/DC power balance metrics on the test split. | `False` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |

### Examples

**Standard Training:**

```
gridfm_dist_app train --config examples/config/rural_MVLV_test.yaml --data_path examples/data
```

---

## Fine-Tuning Models

```
gridfm_dist_app finetune --config path/to/config.yaml --model_path path/to/model.pt
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Fine-tuning configuration file. | `None` |
| `--model_path` | `str` | **Required**. Path to a pre-trained model state dict. | `None` |
| `--exp_name` | `str` | MLflow experiment name. | `timestamp` |
| `--run_name` | `str` | MLflow run name. | `run` |
| `--log_dir` | `str` | MLflow logging directory. | `mlruns` |
| `--data_path` | `str` | Root dataset directory. | `data` |
| `--compile [MODE]` | `str` | Enable `torch.compile` mode. Valid values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. If flag is passed without a value, mode is `default`. | `None` |
| `--bfloat16` | `flag` | Cast model to `torch.bfloat16` (`model.to(torch.bfloat16)`). | `False` |
| `--tf32` | `flag` | Enable TF32 on Ampere+ GPUs via `torch.set_float32_matmul_precision("high")`. | `False` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration, e.g. `gridfm_graphkit_ee`. | `[]` |
| `--num_workers` | `int` | Override `data.workers` from YAML. Use `0` to debug worker crashes. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Disk cache directory for dataset wrapper; cache is loaded from here when present and saved after first population. | `None` |
| `--profiler` | `str` | Enable Lightning profiler (`simple`, `advanced`, `pytorch`). | `None` |
| `--compute_dc_ac_metrics` | `flag` | Compute ground-truth AC/DC power balance metrics on the test split. | `False` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |


---

## Evaluating Models

```
gridfm_dist_app evaluate --config path/to/eval.yaml --model_path path/to/model.pt
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Path to evaluation config. | `None` |
| `--model_path` | `str` | Path to the trained model state dict. | `None` |
| `--normalizer_stats` | `str` | Path to `normalizer_stats.pt` from a training run. Restores `fit_on_train` normalizers from saved statistics instead of re-fitting on current split. | `None` |
| `--exp_name` | `str` | MLflow experiment name. | `timestamp` |
| `--run_name` | `str` | MLflow run name. | `run` |
| `--log_dir` | `str` | MLflow logging directory. | `mlruns` |
| `--data_path` | `str` | Dataset directory. | `data` |
| `--compile [MODE]` | `str` | Enable `torch.compile` mode. Valid values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. If flag is passed without a value, mode is `default`. | `None` |
| `--bfloat16` | `flag` | Cast model to `torch.bfloat16` (`model.to(torch.bfloat16)`). | `False` |
| `--tf32` | `flag` | Enable TF32 on Ampere+ GPUs via `torch.set_float32_matmul_precision("high")`. | `False` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration, e.g. `gridfm_graphkit_ee`. | `[]` |
| `--num_workers` | `int` | Override `data.workers` from YAML. Use `0` to debug worker crashes. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Disk cache directory for dataset wrapper; cache is loaded from here when present and saved after first population. | `None` |
| `--profiler` | `str` | Enable Lightning profiler (`simple`, `advanced`, `pytorch`). | `None` |
| `--compute_dc_ac_metrics` | `flag` | Compute ground-truth AC/DC power balance metrics on the test split. | `False` |
| `--save_output` | `flag` | Save predictions as `<grid_name>_predictions.parquet` under MLflow artifacts (`.../artifacts/test`). | `False` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |

### Example with saved normalizer stats

When evaluating a model on a dataset, you can pass the normalizer statistics from the original training run to ensure the same normalization parameters are used:

```
gridfm_dist_app evaluate \
  --config examples/config/HGNS_PF_datakit_case118.yaml \
  --model_path mlruns/<experiment_id>/<run_id>/artifacts/model/best_model_state_dict.pt \
  --normalizer_stats mlruns/<experiment_id>/<run_id>/artifacts/stats/normalizer_stats.pt \
  --data_path data
```

> **Note:** The `--normalizer_stats` flag only affects normalizers with `fit_strategy = "fit_on_train"` (e.g. `HeteroDataMVANormalizer`). Per-sample normalizers (`HeteroDataPerSampleMVANormalizer`) always recompute their statistics from the current dataset regardless of this flag.

---

## Running Predictions

```
gridfm_dist_app predict --config path/to/config.yaml --model_path path/to/model.pt
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Path to prediction config file. | `None` |
| `--model_path` | `str` | Path to trained model state dict. Optional; may be defined in config. | `None` |
| `--normalizer_stats` | `str` | Path to `normalizer_stats.pt` from a training run. Restores `fit_on_train` normalizers from saved statistics. | `None` |
| `--exp_name` | `str` | MLflow experiment name. | `timestamp` |
| `--run_name` | `str` | MLflow run name. | `run` |
| `--log_dir` | `str` | MLflow logging directory. | `mlruns` |
| `--data_path` | `str` | Dataset directory. | `data` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration, e.g. `gridfm_graphkit_ee`. | `[]` |
| `--num_workers` | `int` | Override `data.workers` from YAML. Use `0` to debug worker crashes. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Disk cache directory for dataset wrapper; cache is loaded from here when present and saved after first population. | `None` |
| `--output_path` | `str` | Directory where predictions are saved as `<grid_name>_predictions.parquet`. | `data` |
| `--compile [MODE]` | `str` | Enable `torch.compile` mode. Valid values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. If flag is passed without a value, mode is `default`. | `None` |
| `--bfloat16` | `flag` | Cast model to `torch.bfloat16` (`model.to(torch.bfloat16)`). | `False` |
| `--tf32` | `flag` | Enable TF32 on Ampere+ GPUs via `torch.set_float32_matmul_precision("high")`. | `False` |
| `--profiler` | `str` | Enable Lightning profiler (`simple`, `advanced`, `pytorch`). | `None` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |

---

## Benchmarking Dataloader Throughput

```
gridfm_dist_app benchmark --config path/to/config.yaml
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Path to configuration YAML file. | `None` |
| `--data_path` | `str` | Root dataset directory. | `data` |
| `--epochs` | `int` | Number of epochs to iterate through the train dataloader. | `3` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Directory for dataset wrapper disk cache. | `None` |
| `--num_workers` | `int` | Override `data.workers` from YAML. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration. | `[]` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |

Use built-in help for full command details:

```
gridfm_dist_app --help
gridfm_dist_app <command> --help
```

---

## Running Tests

### Unit and Integration Tests

Install the test dependencies first (if not already done):

```
pip install -e .[dev,test]
```

Run the full unit test suite:

```
pytest ./tests
```

Run the base set integration tests:

```
pytest ./integrationtests/test_base_set.py
```

### Running Base Set Tests on an LSF Cluster (GPU)

To submit the base set integration tests as an interactive LSF job with GPU access, use `bsub`. Adjust the paths to match your environment:

```
bsub -gpu "num=1" \
     -n 16 \
     -R "rusage[mem=32GB] span[hosts=1]" \
     -Is \
     -J gridfm_base_set_tests \
     /bin/bash -c "
       cd /path/to/gridfm-graphkit && \
       export PATH=/path/to/cuda/bin:\$PATH \
               CUDA_HOME=/path/to/cuda \
               LD_LIBRARY_PATH=/path/to/cuda/lib64:\$LD_LIBRARY_PATH && \
       source /path/to/venv/bin/activate && \
       pytest ./integrationtests/test_base_set.py
     "
```

Key `bsub` options used above:

| Option | Description |
| ------ | ----------- |
| `-gpu "num=1"` | Request 1 GPU |
| `-n 16` | Request 16 CPU slots |
| `-R "rusage[mem=32GB] span[hosts=1]"` | Reserve 32 GB of memory on a single host |
| `-Is` | Run as an interactive shell session |
| `-J <job_name>` | Assign a name to the job |

**Concrete example** (adapt paths to your cluster setup):

```
bsub -gpu "num=1" -n 16 -R "rusage[mem=32GB] span[hosts=1]" -Is -J hpo_trial_190 /bin/bash -c "cd /dccstor/terratorch/users/rkie/gitco/gridfm-graphkit && export PATH=/opt/share/cuda-12.8.1/bin:\$PATH CUDA_HOME=/opt/share/cuda-12.8.1 LD_LIBRARY_PATH=/opt/share/cuda-12.8.1/lib64:\$LD_LIBRARY_PATH && source /u/rkie/venvs/venv_gridfm-graphkit/bin/activate && pytest ./integrationtests/test_base_set.py"
```

---