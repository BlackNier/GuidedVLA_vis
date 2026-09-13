import argparse
import dataclasses
import http.client
import json
import os
import pathlib
import queue
import shlex
import signal
import subprocess
import threading
import time
from contextlib import suppress
from typing import Optional, get_args, get_origin
from tqdm import tqdm


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCAL_PATH_ARG_NAMES = (
    "client_python",
    "libero_plus_path",
    "results_base_dir",
    "video_base_dir",
    "log_dir",
)


@dataclasses.dataclass
class Args:
    checkpoint_dir: Optional[str] = None
    policy_config: str = "pi0_libero_object_depth_skill"

    client_python: str = str(REPO_ROOT / "examples" / "libero_plus" / ".venv" / "bin" / "python")
    libero_plus_path: str = str(REPO_ROOT / "third_party" / "LIBERO-plus")

    results_base_dir: str = str(REPO_ROOT / "data" / "libero_plus")
    video_base_dir: str = str(REPO_ROOT / "data" / "libero_plus")
    log_dir: str = str(REPO_ROOT / "logs" / "libero_plus")

    gpu_ids: Optional[str] = None
    start_port: int = 10000

    task_suites: str = "libero_spatial,libero_object,libero_goal,libero_10"
    categories: str = (
        "Objects Layout,Camera Viewpoints,Robot Initial States,"
        "Language Instructions,Light Conditions,Background Textures,Sensor Noise"
    )
    task_ids: Optional[str] = None
    num_trials_per_task: int = 1

    client_host: str = "127.0.0.1"
    resize_size: int = 224
    replan_steps: int = 5
    attention_dump_interval: int = 1
    export_stage_data: bool = True
    stage_plan_path: Optional[str] = None

    client_use_xvfb: bool = False
    client_mujoco_gl: Optional[str] = None

    # --- VRAM / scheduling ---
    # Maximum concurrent workers per GPU, where each worker owns one server/client pair.
    # For 24GB RTX 4090 cards, 1-2 workers is usually appropriate.
    max_workers_per_gpu: int = 2
    # Estimated peak VRAM per worker in GB, used for reservation accounting.
    # pi0 usually needs about 8-10GB, and MuJoCo EGL may add another 1-3GB.
    estimated_worker_vram_gb: float = 12.0
    # Hard cap: avoid spawning when measured nvidia-smi usage exceeds this ratio,
    # even if the reservation tracker says capacity is available.
    vram_safe_threshold: float = 0.85
    enable_conservative_cuda_mem_cap: bool = True
    server_headroom_reserve_gb: float = 1.0
    # Treat newly spawned workers as unstable for this long before reconciling
    # reservation accounting against measured usage.
    spawn_warmup_sec: float = 45.0
    spawn_cooldown_sec: int = 20
    check_interval_sec: int = 5
    server_ready_timeout_sec: int = 600

    client_timeout_sec: int = 72000


# ---------------------------------------------------------------------------
# Global process registry and shutdown coordination
# ---------------------------------------------------------------------------
active_processes: list[subprocess.Popen] = []
process_lock = threading.Lock()
shutdown_event = threading.Event()
shutdown_in_progress = threading.Event()
task_queue: "queue.Queue[dict]" = queue.Queue()

port_lock = threading.Lock()
current_port = 0

# Per-GPU bookkeeping for VRAM reservation.
gpu_state_lock = threading.Lock()
# gpu_id -> list of (reserved_gb, spawn_time). We drop entries as workers finish.
gpu_reservations: dict[int, list[list[float]]] = {}


def _parse_csv(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _str2bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _unwrap_optional(annotation):
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _is_remote_path(path_str: str) -> bool:
    return "://" in path_str


def _parse_gpu_id_list(value: str) -> list[int]:
    gpu_ids = []
    for item in _parse_csv(value):
        try:
            gpu_ids.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid GPU id '{item}'. Expected a comma-separated list of integers.") from exc
    if not gpu_ids:
        raise ValueError("No GPU ids were provided.")
    return gpu_ids


def _discover_gpu_ids(requested: Optional[str]) -> list[int]:
    if requested:
        return _parse_gpu_id_list(requested)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        try:
            return _parse_gpu_id_list(visible)
        except ValueError:
            pass

    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        encoding="utf-8",
    )
    gpu_ids = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if not gpu_ids:
        raise RuntimeError("No GPUs were found via nvidia-smi.")
    return gpu_ids


def _resolve_path(path_str: str) -> pathlib.Path:
    return pathlib.Path(path_str).expanduser().resolve()


def _absolute_path_preserve_symlinks(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str).expanduser()
    return pathlib.Path(os.path.abspath(path))


def _normalize_local_paths(args: Args) -> None:
    for field_name in LOCAL_PATH_ARG_NAMES:
        raw_path = getattr(args, field_name)
        if field_name == "client_python":
            normalized = _absolute_path_preserve_symlinks(raw_path)
        else:
            normalized = _resolve_path(raw_path)
        setattr(args, field_name, str(normalized))


def _build_task_list(args: Args) -> list[dict]:
    suites = _parse_csv(args.task_suites)
    if not suites:
        raise ValueError("At least one task suite is required.")

    categories = _parse_csv(args.categories)
    tasks = []
    for suite in suites:
        if categories:
            for category in categories:
                tasks.append(
                    {
                        "suite": suite,
                        "category": category,
                        "slug": f"{suite}_{_slugify(category)}",
                    }
                )
        else:
            tasks.append(
                {
                    "suite": suite,
                    "category": None,
                    "slug": suite,
                }
            )
    return tasks


def get_next_port() -> int:
    global current_port  # noqa: PLW0603
    with port_lock:
        port = current_port
        current_port += 1
        return port


# ---------------------------------------------------------------------------
# Process lifecycle helpers
# ---------------------------------------------------------------------------
def register_process(proc: subprocess.Popen) -> None:
    with process_lock:
        active_processes.append(proc)


def unregister_process(proc: subprocess.Popen) -> None:
    with process_lock:
        if proc in active_processes:
            active_processes.remove(proc)


def _pgid_or_none(proc: subprocess.Popen) -> Optional[int]:
    with suppress(Exception):
        return os.getpgid(proc.pid)
    return None


def _signal_process_group(proc: subprocess.Popen, sig: signal.Signals) -> None:
    """Signal the process group. Silent on already-dead processes."""
    if proc.poll() is not None:
        return
    pgid = _pgid_or_none(proc)
    if pgid is None:
        # Process likely already gone; fall back to direct kill just in case.
        with suppress(ProcessLookupError, PermissionError):
            os.kill(proc.pid, sig)
        return
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _wait_for_process_exit(proc: subprocess.Popen, timeout_sec: float) -> bool:
    if proc.poll() is not None:
        return True
    try:
        proc.wait(timeout=max(timeout_sec, 0.0))
        return True
    except subprocess.TimeoutExpired:
        return proc.poll() is not None


def _kill_process_tree(proc: subprocess.Popen, *, grace_sec: float = 0.5) -> None:
    """Kill a subprocess and its entire process group, hard and fast.

    We do NOT attempt SIGTERM graceful shutdown here — the user wants immediate
    teardown. A very short grace period lets cooperative children (like uv run
    relaying signals) exit cleanly, then SIGKILL catches the rest.
    """
    if proc.poll() is not None:
        return

    # SIGINT first gives python-level KeyboardInterrupt handlers a split second
    # to run, which helps CUDA/MuJoCo release contexts cleanly. Then SIGKILL.
    _signal_process_group(proc, signal.SIGINT)
    if _wait_for_process_exit(proc, grace_sec):
        return

    _signal_process_group(proc, signal.SIGKILL)
    _wait_for_process_exit(proc, 2.0)


def _snapshot_active_processes() -> list[subprocess.Popen]:
    with process_lock:
        return [proc for proc in active_processes if proc.poll() is None]


def kill_all_processes(sig: int = signal.SIGINT) -> None:
    """Signal handler entry point. Hard teardown of everything."""
    signame = signal.Signals(sig).name
    exit_code = 130 if sig == signal.SIGINT else 143

    # Second signal: skip niceties, SIGKILL everything, exit immediately.
    if shutdown_in_progress.is_set():
        print(f"\n[!] {signame} received again — SIGKILL all remaining process groups.")
        for proc in _snapshot_active_processes():
            _signal_process_group(proc, signal.SIGKILL)
        os._exit(exit_code)

    shutdown_in_progress.set()
    shutdown_event.set()

    processes = _snapshot_active_processes()
    print(f"\n[!] {signame} received — tearing down {len(processes)} worker process group(s) immediately.")

    # First pass: SIGINT (gives CUDA/MuJoCo a chance to release contexts cleanly).
    for proc in processes:
        _signal_process_group(proc, signal.SIGINT)

    # Short grace window.
    deadline = time.time() + 1.0
    for proc in processes:
        _wait_for_process_exit(proc, max(0.0, deadline - time.time()))

    # Anything still alive: SIGKILL.
    survivors = [p for p in processes if p.poll() is None]
    if survivors:
        print(f"[!] {len(survivors)} group(s) ignored SIGINT — SIGKILL now.")
        for proc in survivors:
            _signal_process_group(proc, signal.SIGKILL)
        kill_deadline = time.time() + 3.0
        for proc in survivors:
            _wait_for_process_exit(proc, max(0.0, kill_deadline - time.time()))

    # Final sweep via ps in case uv/xvfb-run spawned detached grandchildren.
    _orphan_sweep()

    still_alive = len([p for p in _snapshot_active_processes() if p.poll() is None])
    if still_alive:
        print(f"[!] Shutdown finished with {still_alive} process(es) still alive — OS will reap them.")
    else:
        print("[!] All worker process groups exited.")

    os._exit(exit_code)


def _orphan_sweep() -> None:
    """Find any children still attached to our session and SIGKILL them.

    uv run and xvfb-run occasionally spawn grandchildren that detach from the
    immediate process group. A pgrep sweep catches those before we exit.
    """
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["pgrep", "-P", str(my_pid)],
            encoding="utf-8",
        ).strip()
    except subprocess.CalledProcessError:
        return  # No children.
    except FileNotFoundError:
        return  # pgrep unavailable; best effort only.

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            child_pid = int(line)
        except ValueError:
            continue
        with suppress(ProcessLookupError, PermissionError):
            pgid = os.getpgid(child_pid)
            os.killpg(pgid, signal.SIGKILL)


signal.signal(signal.SIGINT, lambda sig, frame: kill_all_processes(sig))
signal.signal(signal.SIGTERM, lambda sig, frame: kill_all_processes(sig))


# ---------------------------------------------------------------------------
# GPU VRAM utilities
# ---------------------------------------------------------------------------
def get_gpu_vram_stats(gpu_id: int) -> tuple[Optional[float], Optional[float]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
                "-i",
                str(gpu_id),
            ],
            encoding="utf-8",
        ).strip()
        used, total = map(float, output.splitlines()[0].split(","))
        return used, total
    except Exception as exc:
        print(f"[Warning] Failed to query GPU {gpu_id} VRAM: {exc}")
        return None, None


def _reserve_gpu_slot(args: Args, gpu_id: int) -> bool:
    """Atomically check capacity and reserve a slot on this GPU.

    We combine three constraints:
      - Per-GPU worker cap (max_workers_per_gpu).
      - Measured VRAM usage (nvidia-smi) under vram_safe_threshold.
      - Reserved VRAM + measured used + estimated new worker <= safe budget.

    The reservation is held until `_release_gpu_slot` is called in finally.
    """
    used_mib, total_mib = get_gpu_vram_stats(gpu_id)
    if used_mib is None or total_mib is None or total_mib <= 0:
        return False

    used_frac = used_mib / total_mib
    safe_budget_mib = args.vram_safe_threshold * total_mib

    with gpu_state_lock:
        reservations = gpu_reservations.setdefault(gpu_id, [])
        # Count how much VRAM we've promised but nvidia-smi might not yet reflect
        # (workers still in warmup window).
        now = time.time()
        promised_mib = sum(r[0] * 1024 for r in reservations if (now - r[1]) < args.spawn_warmup_sec)

        if len(reservations) >= args.max_workers_per_gpu:
            return False

        if used_frac >= args.vram_safe_threshold:
            return False

        need_mib = args.estimated_worker_vram_gb * 1024
        projected_mib = used_mib + promised_mib + need_mib
        if projected_mib > safe_budget_mib:
            return False

        reservations.append([args.estimated_worker_vram_gb, now])
        return True


def _release_gpu_slot(gpu_id: int, reserved_at: Optional[float] = None) -> None:
    """Release one reservation slot. If reserved_at given, match that entry."""
    with gpu_state_lock:
        reservations = gpu_reservations.get(gpu_id, [])
        if not reservations:
            return
        if reserved_at is None:
            reservations.pop(0)
        else:
            for i, r in enumerate(reservations):
                if abs(r[1] - reserved_at) < 1e-6:
                    reservations.pop(i)
                    return
            reservations.pop(0)  # fallback


def compute_server_mem_fraction(args: Args, gpu_id: int) -> Optional[float]:
    """Return the PyTorch CUDA memory fraction cap for a new server on this GPU.

    Note: this only constrains PyTorch's own allocator. MuJoCo/EGL allocations
    from the client process are NOT counted here, which is exactly why we also
    gate spawning via `_reserve_gpu_slot` with an estimated per-worker budget.
    """
    used_mib, total_mib = get_gpu_vram_stats(gpu_id)
    if used_mib is None or total_mib is None or total_mib <= 0:
        return None

    safe_budget_mib = args.vram_safe_threshold * total_mib
    raw_headroom_mib = safe_budget_mib - used_mib
    reserve_mib = args.server_headroom_reserve_gb * 1024
    allocatable_mib = max(0.0, raw_headroom_mib - reserve_mib)
    if allocatable_mib <= 0:
        return None
    mem_fraction = min(allocatable_mib / total_mib, args.vram_safe_threshold)
    return max(mem_fraction, 0.01)


# ---------------------------------------------------------------------------
# Server/client wiring
# ---------------------------------------------------------------------------
def _wait_for_server(port: int, server_process: subprocess.Popen, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline and not shutdown_event.is_set():
        if server_process.poll() is not None:
            return False

        conn = None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
            conn.request("GET", "/healthz")
            response = conn.getresponse()
            response.read()
            if response.status == 200:
                return True
        except (OSError, http.client.HTTPException):
            time.sleep(1)
        finally:
            if conn is not None:
                with suppress(Exception):
                    conn.close()

    return False


def _prepend_env_path(path_value: "pathlib.Path | str", existing: Optional[str]) -> str:
    if existing:
        return os.pathsep.join([str(path_value), existing])
    return str(path_value)


def _compose_pythonpath(extra_path: pathlib.Path, existing: Optional[str]) -> str:
    return _prepend_env_path(extra_path, existing)


def _detect_virtual_env_root(python_path: pathlib.Path) -> Optional[pathlib.Path]:
    if python_path.parent.name != "bin":
        return None
    venv_root = python_path.parent.parent
    if (venv_root / "pyvenv.cfg").exists():
        return venv_root
    return None


def _build_client_env(args: Args) -> dict[str, str]:
    client_env = os.environ.copy()
    client_env["PYTHONPATH"] = _compose_pythonpath(pathlib.Path(args.libero_plus_path), client_env.get("PYTHONPATH"))
    client_env.pop("PYTHONHOME", None)
    venv_root = _detect_virtual_env_root(pathlib.Path(args.client_python))
    if venv_root is not None:
        client_env["VIRTUAL_ENV"] = str(venv_root)
        client_env["PATH"] = _prepend_env_path(venv_root / "bin", client_env.get("PATH"))
    else:
        client_env.pop("VIRTUAL_ENV", None)
    if args.client_mujoco_gl:
        client_env["MUJOCO_GL"] = args.client_mujoco_gl
    return client_env


def _build_server_env(args: Args, gpu_id: int) -> dict[str, str]:
    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return server_env


def _run_client_preflight(args: Args, client_env: dict[str, str]) -> dict[str, object]:
    check_script = """
import importlib.util
import json
import os
import sys

import imageio

summary = {
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "pythonhome": os.environ.get("PYTHONHOME"),
    "virtual_env": os.environ.get("VIRTUAL_ENV"),
    "imageio": getattr(imageio, "__file__", None),
    "openpi_client": importlib.util.find_spec("openpi_client") is not None,
    "libero": importlib.util.find_spec("libero") is not None,
}
print(json.dumps(summary, ensure_ascii=True))
missing = [name for name in ("openpi_client", "libero") if not summary[name]]
if missing:
    raise SystemExit("Missing imports: " + ", ".join(missing))
"""
    result = subprocess.run(
        [args.client_python, "-c", check_script],
        env=client_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        raise RuntimeError(
            "LIBERO-plus client preflight failed.\n"
            f"Interpreter: {args.client_python}\n"
            f"VIRTUAL_ENV: {client_env.get('VIRTUAL_ENV')!r}\n"
            f"PYTHONHOME: {client_env.get('PYTHONHOME')!r}\n"
            f"PYTHONPATH: {client_env.get('PYTHONPATH')!r}\n"
            f"PATH head: {client_env.get('PATH', '').split(os.pathsep)[:3]}\n"
            f"stdout:\n{stdout or '<empty>'}\n"
            f"stderr:\n{stderr or '<empty>'}"
        )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("LIBERO-plus client preflight succeeded but produced no diagnostic output.")
    return json.loads(lines[-1])


def _start_logged_process(cmd: list[str], env: dict[str, str], log_path: pathlib.Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise
    # Keep the file handle open for the lifetime of the subprocess — closing it
    # is fine too since Popen dup'd the fd, but we let GC handle it.
    log_file.close()
    register_process(proc)

    if shutdown_event.is_set() and proc.poll() is None:
        _kill_process_tree(proc)

    return proc


def _build_server_cmd(args: Args, checkpoint_dir: str, port: int, mem_fraction: Optional[float]) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "scripts/serve_policy.py",
        "--env",
        "LIBERO",
        "--port",
        str(port),
    ]
    if mem_fraction is not None:
        cmd.extend(["--max-cuda-mem-fraction", f"{mem_fraction:.4f}"])

    cmd.extend(
        [
            "policy:checkpoint",
            "--policy.config",
            args.policy_config,
            "--policy.dir",
            checkpoint_dir,
        ]
    )
    return cmd


def _build_client_cmd(args: Args, task: dict, port: int) -> tuple[list[str], pathlib.Path]:
    task_slug = task["slug"]
    if task["category"] is None:
        result_json = pathlib.Path(args.results_base_dir) / f"{task_slug}.json"
    else:
        result_json = pathlib.Path(args.results_base_dir) / task["suite"] / "results.json"

    video_path = pathlib.Path(args.video_base_dir) / task_slug
    cmd = []
    if args.client_use_xvfb:
        cmd.extend(["xvfb-run", "-a"])

    cmd.extend(
        [
            args.client_python,
            str(REPO_ROOT / "examples" / "libero_plus" / "main.py"),
            "--args.num_trials_per_task",
            str(args.num_trials_per_task),
            "--args.task-suite-name",
            task["suite"],
            "--args.video_out_path",
            str(video_path),
            "--image-out-path",
            str(pathlib.Path(args.results_base_dir) / task_slug / "rollout_images"),
            "--attention-out-path",
            str(pathlib.Path(args.results_base_dir) / task_slug / "rollout_attention"),
            "--attention-dump-interval",
            str(args.attention_dump_interval),
            "--args.results-json-path",
            str(result_json),
            "--args.port",
            str(port),
            "--args.host",
            args.client_host,
            "--args.resize_size",
            str(args.resize_size),
            "--args.replan_steps",
            str(args.replan_steps),
        ]
    )

    if not args.export_stage_data:
        cmd.append("--no-export-stage-data")
    if args.stage_plan_path:
        cmd.extend(["--stage-plan-path", str(pathlib.Path(args.stage_plan_path).resolve())])
    if task["category"] is not None:
        cmd.extend(["--args.category", task["category"]])
    if args.task_ids:
        cmd.extend(["--args.task_ids", args.task_ids])

    return cmd, result_json


def run_worker_daemon(
    args: Args,
    checkpoint_dir: str,
    gpu_id: int,
    port: int,
    pbar: tqdm,
    reservation_time: float,
) -> None:
    """One worker: spin up a server, then pull tasks from the queue until empty.

    Whatever happens, this must:
      - Release the GPU reservation slot.
      - Tear down the server process tree, hard.
    """
    server_process: Optional[subprocess.Popen] = None
    try:
        if shutdown_event.is_set():
            return

        server_env = _build_server_env(args, gpu_id)
        server_mem_fraction = None
        if args.enable_conservative_cuda_mem_cap:
            server_mem_fraction = compute_server_mem_fraction(args, gpu_id)
            if server_mem_fraction is None:
                print(f"[Worker GPU {gpu_id}:{port}] No safe headroom for PyTorch cap; abort spawn.")
                return
            print(
                f"[Worker GPU {gpu_id}:{port}] Starting server with PyTorch mem cap "
                f"{server_mem_fraction:.1%} (safe threshold={args.vram_safe_threshold:.0%})."
            )

        client_env = _build_client_env(args)
        log_dir = pathlib.Path(args.log_dir)
        server_log_path = log_dir / f"server_gpu_{gpu_id}_port_{port}.log"
        server_cmd = _build_server_cmd(args, checkpoint_dir, port, server_mem_fraction)
        server_process = _start_logged_process(server_cmd, server_env, server_log_path)

        if not _wait_for_server(port, server_process, args.server_ready_timeout_sec):
            if shutdown_event.is_set():
                return
            try:
                tail = subprocess.check_output(["tail", "-n", "30", str(server_log_path)], text=True)
            except Exception:
                tail = "<tail failed>"
            print(f"[Worker GPU {gpu_id}:{port}] Server failed to become ready.\n{tail}")
            return

        _mark_gpu_ready(gpu_id)

        # Drain tasks from the shared queue.
        while not shutdown_event.is_set():
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                break

            task_slug = task["slug"]
            client_log_path = log_dir / f"client_{task_slug}_port_{port}.log"
            client_cmd, result_json = _build_client_cmd(args, task, port)
            result_json.parent.mkdir(parents=True, exist_ok=True)

            client_process: Optional[subprocess.Popen] = None
            try:
                client_process = _start_logged_process(client_cmd, client_env, client_log_path)
                try:
                    client_process.wait(timeout=args.client_timeout_sec)
                except subprocess.TimeoutExpired:
                    print(f"[Worker GPU {gpu_id}:{port}] Client timeout on {task_slug}; killing.")
                    _kill_process_tree(client_process)
                # Verify the server still responds. If the client crashed in a
                # way that took the server with it (rare but seen with MuJoCo
                # OOM cascades), stop this worker so the monitor can respawn.
                if server_process.poll() is not None:
                    print(f"[Worker GPU {gpu_id}:{port}] Server died during client run; stopping worker.")
                    task_queue.task_done()
                    pbar.update(1)
                    break
            except Exception as exc:
                print(f"[Worker GPU {gpu_id}:{port}] Client error on {task_slug}: {exc}")
            finally:
                if client_process is not None:
                    # Belt-and-suspenders: if still alive, kill its tree.
                    if client_process.poll() is None:
                        _kill_process_tree(client_process)
                    unregister_process(client_process)

            pbar.update(1)
            task_queue.task_done()

    except Exception as exc:
        print(f"[Worker GPU {gpu_id}:{port}] Worker exception: {exc}")
    finally:
        # Always tear down the server — regardless of shutdown state. The
        # original code guarded this with `if not shutdown_event.is_set()`
        # which leaked server processes precisely when we most wanted them
        # dead.
        if server_process is not None:
            _kill_process_tree(server_process)
            unregister_process(server_process)
        _release_gpu_slot(gpu_id, reservation_time)


# Tracks whether each GPU has at least one worker that passed /healthz.
gpu_ready_lock = threading.Lock()
gpu_first_ready: dict[int, bool] = {}


def _mark_gpu_ready(gpu_id: int) -> None:
    """Worker calls this once its server passes /healthz."""
    with gpu_ready_lock:
        gpu_first_ready[gpu_id] = True


def _gpu_has_ready_worker(gpu_id: int) -> bool:
    with gpu_ready_lock:
        return gpu_first_ready.get(gpu_id, False)


def _gpu_active_reservations(gpu_id: int) -> int:
    with gpu_state_lock:
        return len(gpu_reservations.get(gpu_id, []))


def monitor_and_dispatch(args: Args, checkpoint_dir: str, gpu_ids: list[int], pbar: tqdm) -> None:
    last_spawn_time = {gpu_id: 0.0 for gpu_id in gpu_ids}
    # Initialize first-ready state for every GPU we plan to use.
    with gpu_ready_lock:
        for gpu_id in gpu_ids:
            gpu_first_ready.setdefault(gpu_id, False)

    while not task_queue.empty() and not shutdown_event.is_set():
        spawned_this_round = False
        for gpu_id in gpu_ids:
            if task_queue.empty() or shutdown_event.is_set():
                break

            if time.time() - last_spawn_time[gpu_id] < args.spawn_cooldown_sec:
                continue

            # Cold-start guard: if this GPU has never had a worker reach
            # /healthz, only allow ONE in-flight reservation. This prevents
            # two simultaneous cold loads (pi0 + JAX compile) from competing
            # for VRAM / PCIe and tripping the server_ready_timeout for both.
            if not _gpu_has_ready_worker(gpu_id) and _gpu_active_reservations(gpu_id) >= 1:
                continue

            if not _reserve_gpu_slot(args, gpu_id):
                continue

            with gpu_state_lock:
                reservations = gpu_reservations.get(gpu_id, [])
                reservation_time = reservations[-1][1] if reservations else time.time()

            port = get_next_port()
            threading.Thread(
                target=run_worker_daemon,
                args=(args, checkpoint_dir, gpu_id, port, pbar, reservation_time),
                daemon=True,
            ).start()
            last_spawn_time[gpu_id] = time.time()
            spawned_this_round = True

        time.sleep(args.check_interval_sec if not spawned_this_round else 1.0)


def _validate_args(args: Args) -> str:
    if not args.checkpoint_dir:
        raise ValueError(
            "checkpoint_dir is required. Example: "
            "--checkpoint-dir checkpoints/pi0_libero_object_depth_skill/<exp_name>/<step>"
        )

    _normalize_local_paths(args)

    checkpoint_dir = args.checkpoint_dir
    if not _is_remote_path(checkpoint_dir) and not _resolve_path(checkpoint_dir).exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not _is_remote_path(checkpoint_dir):
        checkpoint_dir = str(_resolve_path(checkpoint_dir))
        args.checkpoint_dir = checkpoint_dir

    client_python = pathlib.Path(args.client_python)
    if not client_python.exists():
        raise FileNotFoundError(f"LIBERO-plus client Python was not found: {client_python}")

    libero_plus_path = pathlib.Path(args.libero_plus_path)
    if not libero_plus_path.exists():
        raise FileNotFoundError(f"LIBERO-plus checkout was not found: {libero_plus_path}")

    if args.max_workers_per_gpu < 1:
        raise ValueError("max_workers_per_gpu must be >= 1")
    if args.estimated_worker_vram_gb <= 0:
        raise ValueError("estimated_worker_vram_gb must be > 0")

    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.results_base_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.video_base_dir).mkdir(parents=True, exist_ok=True)

    return checkpoint_dir


def _parse_cli_args() -> Args:
    parser = argparse.ArgumentParser(description="Run LIBERO-plus evaluation tasks in parallel workers.")
    for field in dataclasses.fields(Args):
        option = f"--{field.name.replace('_', '-')}"
        default = field.default
        field_type = _unwrap_optional(field.type)
        kwargs = {"default": default}

        if field_type is bool:
            kwargs["type"] = _str2bool
            kwargs["metavar"] = "BOOL"
        elif field_type in (int, float, str):
            kwargs["type"] = field_type

        parser.add_argument(option, **kwargs)

    namespace = parser.parse_args()
    return Args(**vars(namespace))


def main(args: Args) -> None:
    global current_port  # noqa: PLW0603
    checkpoint_dir = _validate_args(args)
    client_summary = _run_client_preflight(args, _build_client_env(args))
    gpu_ids = _discover_gpu_ids(args.gpu_ids)
    current_port = args.start_port

    all_tasks = _build_task_list(args)
    for task in all_tasks:
        task_queue.put(task)

    print(f"Policy config: {args.policy_config}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"GPU ids: {gpu_ids}")
    print(f"Task count: {len(all_tasks)}")
    print(f"Task suites: {_parse_csv(args.task_suites)}")
    print(f"Categories: {_parse_csv(args.categories) or ['ALL']}")
    print(
        f"Scheduling: max_workers_per_gpu={args.max_workers_per_gpu}, "
        f"estimated_worker_vram={args.estimated_worker_vram_gb:.1f}GB, "
        f"safe_threshold={args.vram_safe_threshold:.0%}"
    )
    print(
        "Client python: "
        f"{client_summary['executable']} "
        f"(prefix={client_summary['prefix']}, imageio={client_summary['imageio']})"
    )
    print(
        "Client command prefix: "
        f"{' '.join(shlex.quote(part) for part in _build_client_cmd(args, all_tasks[0], args.start_port)[0][:4])}"
    )

    with tqdm(total=len(all_tasks), unit="task", desc="Total Progress") as pbar:
        monitor_thread = threading.Thread(
            target=monitor_and_dispatch,
            args=(args, checkpoint_dir, gpu_ids, pbar),
            daemon=True,
        )
        monitor_thread.start()

        while not task_queue.empty() and not shutdown_event.is_set():
            time.sleep(2)

        # Wait for all in-flight tasks to complete (or shutdown).
        if not shutdown_event.is_set():
            task_queue.join()

    # Clean up any workers still alive (e.g. their server is still running
    # waiting to be torn down even though the queue is empty).
    print("\nDraining remaining worker processes...")
    for proc in _snapshot_active_processes():
        _kill_process_tree(proc)

    print("All benchmarks completed.")


if __name__ == "__main__":
    main(_parse_cli_args())
