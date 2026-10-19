# Pudding supervision ablation from a common original π0 initialization

Three arms: `single_task_ft`, `multi_task_ft`, `multi_wrist_fixed_task_ft`.
This experiment starts from **original `pi0_base_pt/model.safetensors`**, not either previously fine-tuned 120000-step checkpoint.

`prepare_common.py` loads original π0 weights, creates ControlAttention once with seed 42 (copy_weights=False, zero output projections), and saves a full shared initialization. Each arm verifies its SHA256 and every loaded tensor, then creates a fresh AdamW optimizer. All arms use the same 30 training demonstrations (7547 frames), five held-out demonstrations, and train-only normalization; images/actions and split are unchanged. Only supervision masks differ. The subset export contains 35 demos total.

## Configuration

- Four GPUs, global batch **16**, local batch **4**, no gradient accumulation.
- **6000 updates**, checkpoints at **3000 and 6000**; validation disabled.
- Fresh GPU AdamW: beta=(0.9,0.95), eps=1e-8, weight decay=1e-10, clipping=1.
- LR: 300-update warmup, peak 2e-5, cosine decay to 2e-6 at 6000.
- FP32 parameters/optimizer, BF16 forward autocast, gradient checkpointing, DDP static graph.
- Object loss weight 0.001, no skill/depth supervision, seed 42.
- No ZeRO, CPU optimizer offload, or microbatch workaround. Requires sufficient GPU memory; 4×40GB is not a supported memory target for this configuration.
- Eager execution by default. Optional `PUDDING_TORCH_COMPILE=1` enables compilation for **all arms**; this is not performance-validated. Do not change execution settings between arms.

## Run from the repository root

Install the repository's training dependencies and modified transformers as documented in its main README. Download the private dataset `BlackNier/pudding-wrist-ablation` so DATA_ROOT contains `single`, `multi`, and `multi_fixed`. Provide your own authenticated Hugging Face session; no credentials are included.

```bash
export PYTHON_BIN=/path/to/environment/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_ROOT=/path/to/pudding-wrist-ablation
export BASE_CKPT=/path/to/pi0_base_pt
export TOKENIZER=/path/to/paligemma_tokenizer.model
export PREP_ROOT=/path/to/new_pudding_preparation
export OUTPUT_ROOT=/path/to/desired_checkpoints
bash experiments/pudding_base/run_all.sh
```

Preparation runs on CPU once. Each arm then trains sequentially. Final checkpoints:

```
$OUTPUT_ROOT/pudding_common_init/single_task_ft_b16_s6000_seed42/6000
$OUTPUT_ROOT/pudding_common_init/multi_task_ft_b16_s6000_seed42/6000
$OUTPUT_ROOT/pudding_common_init/multi_wrist_fixed_task_ft_b16_s6000_seed42/6000
```

Each includes `model.safetensors`, standard `optimizer.pt`, `metadata.pt`, and `assets/pudding_train30/norm_stats.json`. Use that norm for inference. Existing training output directories are rejected rather than overwritten. This launcher deliberately does not promise exact restart/resume (RNG and data iterator states are not serialized). Reuse PREP_ROOT only for this unchanged base/data configuration; use a fresh directory if either changes. Logs are written under OUTPUT_ROOT/logs.

CPU-only configuration/data check (no training):

```bash
"$PYTHON_BIN" experiments/pudding_base/run_finetune.py \
  --arm multi_wrist_fixed_task_ft --check-data \
  --data-root "$DATA_ROOT" --common-checkpoint "$PREP_ROOT/common_init" \
  --norm-root "$PREP_ROOT/norm_assets" --tokenizer "$TOKENIZER" \
  --output "$OUTPUT_ROOT"
```

## Local `/gs/bs/...` paths

`src/openpi/training/config.py` now retains local asset paths as `pathlib.Path` until they reach the downloader. Only explicit URIs such as `gs://openpi-assets/...` use cloud path handling. This prevents etils from interpreting a local HPC `/gs/bs` mount as a GCS bucket called `bs`. Checkpoint, preparation, dataset and tokenizer arguments use local paths. A local tokenizer is mandatory here, avoiding an implicit GCS download.

Regression test: `python experiments/pudding_base/test_paths.py`.

Full large-GPU training has not been run for this new branch; CPU checks do not establish throughput or peak-memory capacity.
