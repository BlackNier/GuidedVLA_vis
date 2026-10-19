#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT containing single, multi, multi_fixed}"
: "${BASE_CKPT:?Set BASE_CKPT to original pi0_base_pt directory}"
: "${TOKENIZER:?Set TOKENIZER to local paligemma_tokenizer.model}"
: "${PREP_ROOT:?Set PREP_ROOT for shared initialization and norm}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for checkpoints}"
PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OPENPI_PALIGEMMA_TOKENIZER_PATH="$TOKENIZER"
if [[ ! -f "$PREP_ROOT/common_init/manifest.json" ]]; then
  "$PYTHON_BIN" experiments/pudding_base/prepare_common.py --base-checkpoint "$BASE_CKPT" --data-root "$DATA_ROOT" --output "$PREP_ROOT"
fi
mkdir -p "$OUTPUT_ROOT/logs"
for arm in single_task_ft multi_task_ft multi_wrist_fixed_task_ft; do
  if [[ -e "$OUTPUT_ROOT/logs/${arm}.log" ]]; then
    echo "Existing log for $arm; choose a fresh OUTPUT_ROOT" >&2; exit 1
  fi
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 experiments/pudding_base/run_finetune.py \
    --arm "$arm" --execute --data-root "$DATA_ROOT" --common-checkpoint "$PREP_ROOT/common_init" \
    --norm-root "$PREP_ROOT/norm_assets" --tokenizer "$TOKENIZER" --output "$OUTPUT_ROOT" \
    > "$OUTPUT_ROOT/logs/${arm}.log" 2>&1
done
