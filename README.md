# GroupMask

GroupMask contains training and evaluation utilities for group-sparsity masking of Hugging Face causal language models. The code focuses on learning and applying hypernetwork-generated masks for Llama/Qwen-style transformer blocks.

## Repository Layout

- `recipe_train_groupsparsity.py`: main group-sparsity training recipe.
- `hf_ppl.py`: perplexity evaluation and optional Hugging Face-readable model export.
- `flashlm/`: model, data loading, distributed training, and compression helpers.
- `run_group.sh`: example training launcher.
- `run_transfer_ppl.sh`: example evaluation/export launcher.

## Setup

Create an environment with PyTorch, Transformers, Datasets, jsonargparse, tqdm, and any CUDA-specific packages required by your machine.

```bash
pip install torch transformers datasets jsonargparse tqdm
```

Optional dependencies used by some configurations:

```bash
pip install bitsandbytes
```

## Configuration

Private paths are intentionally not stored in the repository. Use environment variables to point the code at local data, checkpoints, and output locations.

```bash
export FLASHLM_DATA_ROOT=/path/to/datasets
export FLASHLM_HF_CACHE_DIR=/path/to/hf_cache
export GROUPMASK_OUTPUT_ROOT=/path/to/outputs
```

For shell launchers, these variables are also supported:

```bash
export HF_MODEL=Qwen/Qwen3-14B
export HN_PATH=/path/to/hn-checkpoint.pt
export SAVE_HF_DIR=/path/to/exported-model
export NPROC_PER_NODE=1
export MASTER_PORT=29503
```

Notifications are disabled by default. To enable ntfy notifications, set only the topic name:

```bash
export NTFY_TOPIC=your_topic_name
```

## Train

Run the example launcher:

```bash
./run_group.sh
```

Or call the recipe directly:

```bash
torchrun --nproc_per_node=1 --master_port=29503 recipe_train_groupsparsity.py \
  --exp_name=groupsparsity \
  --hf_model="$HF_MODEL" \
  --dataset_list=['wiki'] \
  --total_n_step=30000
```

Checkpoints are written under `GROUPMASK_OUTPUT_ROOT` when that environment variable is set, otherwise under `./outputs`.

## Evaluate And Export

Run the example evaluation/export launcher:

```bash
HN_PATH=./checkpoints/hn-ckpt-iter-030000.pt \
SAVE_HF_DIR=./outputs/exported-model \
./run_transfer_ppl.sh
```

The launcher uses `hf_ppl.py` by default. Override `EVAL_SCRIPT` if you maintain a separate evaluation entrypoint.

## Data

The data helpers use public Hugging Face dataset names where possible. Local datasets are resolved relative to `FLASHLM_DATA_ROOT`. For example, a local dataset referenced as `CodeAlpaca-20k.hf` should be available at:

```text
$FLASHLM_DATA_ROOT/CodeAlpaca-20k.hf
```

Hugging Face cache location can be controlled with `FLASHLM_HF_CACHE_DIR`.

## Public Release Notes

This repository should not contain personal paths, private notification topics, API keys, tokens, or machine-specific absolute paths. Keep those values in environment variables or local untracked config files.
