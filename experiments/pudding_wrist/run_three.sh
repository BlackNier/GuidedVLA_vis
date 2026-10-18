#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/../../src:$HERE:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-$HERE/logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
exec 9>"$HERE/.training.lock"
flock -n 9 || { echo 'Another queue holds this checkout lock'; exit 73; }
for arm in single_task_ft multi_task_ft multi_wrist_fixed_task_ft; do
  "$PYTHON_BIN" - <<'PY'
import os,subprocess
ids=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
assert len(ids)==4 and all(ids),'Set CUDA_VISIBLE_DEVICES to four available GPUs'
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
mapping={a.strip():b.strip() for a,b in (r.split(',') for r in rows.splitlines())}
selected={mapping.get(i,i) for i in ids}; assert len(selected)==4
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert not any(r.split(',')[0].strip() in selected for r in procs.splitlines()),'Selected GPU occupied; refusing to start'
PY
  echo "START $arm" | tee -a "$LOG_DIR/queue.log"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    "$HERE/run_finetune.py" --arm "$arm" --execute "$@" > "$LOG_DIR/$arm.log" 2>&1
  echo "DONE $arm" | tee -a "$LOG_DIR/queue.log"
done
