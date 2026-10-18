# Pudding wrist-supervision ablation

Three sequential fine-tuning arms for task 9: “put the white mug on the plate and put the chocolate pudding to the right of the plate”. No inference-time attention guidance.

| Arm | Starting checkpoint | Data folder |
| --- | --- | --- |
| single_task_ft | original single-object checkpoint | single |
| multi_task_ft | original multi-object checkpoint | multi |
| multi_wrist_fixed_task_ft | same multi-object checkpoint | multi_fixed |

Each dataset contains the same 35 complete demonstrations / 8,726 frames. `episode_split.json` freezes 30 training demos (7,547 frames) and 5 validation demos (1,179 frames). Original RGB, state and actions are unchanged. The repaired multi version changes wrist mask/weight/valid supervision on 1,374 frames in 33 demos (1,214 training frames, 160 validation frames). It does not fill invisible/unreliable regions indiscriminately. These validation demos may have appeared in the original checkpoint training.

## Training configuration

- Four GPUs, **global batch 16**, batch 4 per rank, **6,000 optimizer steps**.
- Original FP32 model parameters / AdamW, existing BF16 forward autocast, gradient checkpointing and torch.compile.
- **No low-memory variant, gradient accumulation or optimizer sharding.**
- LR 5e-6, warmup 100 steps, cosine decay to 5e-7 at step 6000.
- Object loss coefficient 0.001; same trained ControlAttention settings as each starting checkpoint.
- Fresh optimizer and schedule, seed 42; preserve original normalization and delta-action transform.
- Validation every 250 steps; checkpoints every 1000 steps. Existing trainer additionally saves step5999; the required final endpoint is **6000**.
- B vs C isolates the repair; A is a reference, not an equivalent-initialization causal control (checkpoint and original supervision formulation differ).

**Validation status:** CPU dataset loading, exact selected-demo membership, 50×7 action chunks and episode-boundary padding checks passed. Model/checkpoint loading passed on the previous four-GPU attempt. The original batch4/GPU profile ran out of memory during compile warmup on 4×A10040GB, before any optimizer step. Larger-GPU end-to-end training is still unverified. This branch is runnable training code, not a claim of completed GPU validation or improved success rate.

## Setup and inputs

Use this repository's documented environment and matching PyTorch/transformers setup. Provide:

1. The three exported subset folders under one `--data-root`.
2. Single and multi checkpoint directories, each containing `model.safetensors`.
3. Original normalization assets under `SINGLE_CHECKPOINT/assets/ybwowen/libero`.
4. A local `paligemma_tokenizer.model` file.

No checkpoint weights, datasets, credentials, or machine-specific absolute paths are stored in this branch. The code reads the original model settings from `checkpoint_metadata.json` (model fields only).

```bash
export PYTHON_BIN=/path/to/environment/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3  # four GPUs allocated to you

# CPU-only configuration and dataset check; does not train.
"$PYTHON_BIN" experiments/pudding_wrist/run_finetune.py \
  --arm multi_wrist_fixed_task_ft \
  --data-root /path/to/pudding_ablation_subsets \
  --single-checkpoint /path/to/single/120000 \
  --multi-checkpoint /path/to/multi/120000 \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --check-data

# Sequential single -> multi -> repaired multi; stop immediately on failure.
bash experiments/pudding_wrist/run_three.sh \
  --data-root /path/to/pudding_ablation_subsets \
  --single-checkpoint /path/to/single/120000 \
  --multi-checkpoint /path/to/multi/120000 \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --output /path/to/new-checkpoints
```

The queue refuses occupied GPUs, uses a checkout lock, and never terminates another user's processes. `--output` must be new/empty for each arm; partial checkpoints are not silently overwritten or resumed. Logs are under `experiments/pudding_wrist/logs/<timestamp>` or `$LOG_DIR`. The queue verifies final step6000 model/optimizer/metadata files before proceeding. No scheduled monitoring is installed.

## Necessary implementation fixes

- Inject ControlAttention **before** loading the already fine-tuned checkpoint; reject unexpected/missing trainable weights instead of silently reinitializing the branch.
- Protect execution with `if __name__ == '__main__'` so DataLoader spawn workers cannot recursively launch training.
- Explicitly filter selected episodes: this installed LeRobot version's local-data loader does not itself honor `episodes` filtering.
- Preserve source episode IDs while remapping temporal lookup to selected-row positions; assert that padded future actions stay inside the current episode.

## Subset format and export

`export_subsets.py` creates Parquet subsets with embedded RGB bytes, preserving original IDs, all row columns and source metadata. `subset_manifest.json` is authoritative for the actual 35-demo subset; source `meta/` describes the original dataset for provenance. **Use the included selected-episode loader, not a generic unfiltered LeRobot loader.** This is a custom sparse-episode export, not a reindexed native LeRobot dataset.

```bash
"$PYTHON_BIN" experiments/pudding_wrist/export_subsets.py \
  --repo /path/to/original/GuidedVLA \
  --out /path/to/new/pudding_ablation_subsets \
  --manifest experiments/pudding_wrist/task9_rows.json
```

Export expects the original datasets under `data/datasets/ybwowen_libero`, `libero_sam2_weighted`, and the independently repaired `libero_sam2_weighted_pudding_wrist_fixed_20261018`. It does not generate the repair from scratch; the repaired source or already exported three subsets must be supplied. It refuses overwriting existing exports.

Measured serialized size (all files, decimal GB): single 1.030230207 GB; multi 1.030377578 GB; multi_fixed 1.030454609 GB; total **3.091062394 GB / 2.878776187 GiB**. RGB is duplicated across arms for independent loading. Hugging Face deduplication/compression/storage accounting may differ.
