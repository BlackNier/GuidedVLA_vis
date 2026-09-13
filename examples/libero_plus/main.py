import collections
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

try:
    from .stage_export import StageRecorder, export_is_complete, revision
except ImportError:
    from stage_export import StageRecorder, export_is_complete, revision

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
ATTENTION_CAPTURE_IMPL = "fp32_masked_softmax_v2"
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_TASK_CLASSIFICATION_PATH = (
    REPO_ROOT / "third_party" / "LIBERO-plus" / "libero" / "libero" / "benchmark" / "task_classification.json"
)
ALL_TASK_SUITE_NAMES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
}


@dataclasses.dataclass(frozen=True)
class TaskClassificationSelection:
    task_ids_0based: Set[int]
    task_names: Set[str]

    def matches(self, task_id: int, task_name: str) -> bool:
        return task_id in self.task_ids_0based or task_name in self.task_names


@dataclasses.dataclass
class Args:
    """Evaluation arguments."""

    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    category: Optional[str] = None
    task_classification_path: Optional[str] = None
    task_ids: Optional[str] = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    # Global cap on newly executed rollouts across all selected tasks/suites.
    # Existing completed rollouts are skipped and do not consume this budget.
    num_rollouts: Optional[int] = None

    video_out_path: str = "data/libero/videos"
    image_out_path: str = "data/libero_plus/rollout_images"
    attention_out_path: str = "data/libero_plus/rollout_attention"
    # Save one image/PT pair every N exported rollout frames.  This is the
    # client-side counterpart of the server-side attention dump interval.
    attention_dump_interval: int = 1

    # Per-control-step state export is independent of PNG / attention sampling.
    export_stage_data: bool = True
    stage_plan_path: Optional[str] = None

    seed: int = 7

    results_json_path: str = "data/libero/results.json"

    prompt_strip_trailing_id_with_prev: bool = True
    prompt_strip_trailing_word_ending_with_digit: bool = True


def _resolve_results_json_path(args: Args) -> str:
    if args.category is None:
        return args.results_json_path

    base_path = pathlib.Path(args.results_json_path)
    safe_category = args.category.replace(" ", "_").replace("/", "_")
    return str(base_path.parent / f"{base_path.stem}_{safe_category}{base_path.suffix}")


def _get_suite_names(task_suite_name: str) -> List[str]:
    if task_suite_name == "all":
        return list(ALL_TASK_SUITE_NAMES)
    return [task_suite_name]


def _resolve_task_classification_path(args: Args) -> pathlib.Path:
    if args.task_classification_path:
        return pathlib.Path(args.task_classification_path).expanduser()
    return DEFAULT_TASK_CLASSIFICATION_PATH


def _normalize_category_name(value: str) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _load_classification_by_suite(args: Args, suite_names: List[str]) -> Dict[str, TaskClassificationSelection]:
    if args.category is None:
        return {}

    classification_path = _resolve_task_classification_path(args)
    try:
        with open(classification_path, encoding="utf-8") as f:
            classification = json.load(f)
    except Exception as e:
        logging.warning(
            f"Failed to load task classification from {classification_path}: {e}. Proceeding without category filter."
        )
        return {}

    requested_category = _normalize_category_name(args.category)
    classification_by_suite: Dict[str, TaskClassificationSelection] = {}
    for suite_name in suite_names:
        suite_entries = classification.get(suite_name)
        if suite_entries is None:
            logging.warning(
                "No task classification entries found for suite '%s' in %s. Category filter will be disabled for this suite.",
                suite_name,
                classification_path,
            )
            continue

        matched_entries = [
            entry
            for entry in suite_entries
            if _normalize_category_name(entry.get("category", "")) == requested_category
        ]
        task_ids_0based = {
            int(entry["id"]) - 1
            for entry in matched_entries
            if isinstance(entry.get("id"), int) and int(entry["id"]) > 0
        }
        task_names = {str(entry["name"]) for entry in matched_entries if entry.get("name")}
        classification_by_suite[suite_name] = TaskClassificationSelection(
            task_ids_0based=task_ids_0based,
            task_names=task_names,
        )
        logging.info(
            "[%s] category '%s' matched %d classification entries.",
            suite_name,
            args.category,
            len(matched_entries),
        )

    logging.info(f"Category filter enabled: '{args.category}'. Using classification at {classification_path}")
    return classification_by_suite


def _get_max_steps_for_suite(suite_name: str) -> int:
    try:
        return MAX_STEPS_BY_SUITE[suite_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task suite: {suite_name}") from exc


class ArtifactPaths:
    @staticmethod
    def _truncate_utf8(value: str, max_bytes: int) -> str:
        if max_bytes <= 0:
            return ""
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value
        out = []
        total = 0
        for ch in value:
            b = ch.encode("utf-8")
            if total + len(b) > max_bytes:
                break
            out.append(ch)
            total += len(b)
        return "".join(out)

    @classmethod
    def safe_video_filename(
        cls,
        task_description: str,
        episode_index: int,
        suffix: str,
        *,
        prefix: str = "rollout_",
    ) -> str:
        """Create a filename that stays within common filesystem limits."""
        task_segment = task_description.replace(" ", "_")
        base = f"{prefix}{task_segment}_ep{episode_index:02d}_{suffix}"
        ext = ".mp4"
        filename = f"{base}{ext}"

        if len(filename.encode("utf-8")) <= 255:
            return filename

        hash_suffix = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
        reserved = len(f"_{hash_suffix}{ext}".encode())
        max_base_bytes = 255 - reserved
        base_short = cls._truncate_utf8(base, max_base_bytes)

        if base_short:
            return f"{base_short}_{hash_suffix}{ext}"
        return f"{hash_suffix}{ext}"

    @classmethod
    def build(
        cls,
        base_dir: str,
        task_description: str,
        episode_index: int,
        suffix: str,
        *,
        prefix: str,
    ) -> pathlib.Path:
        return pathlib.Path(base_dir) / cls.safe_video_filename(
            task_description,
            episode_index,
            suffix,
            prefix=prefix,
        )

    @classmethod
    def build_rollout_dir(
        cls,
        base_dir: str,
        task_description: str,
        episode_index: int,
        suffix: str,
    ) -> pathlib.Path:
        filename = cls.safe_video_filename(task_description, episode_index, suffix, prefix="rollout_")
        return pathlib.Path(base_dir) / pathlib.Path(filename).stem


@dataclasses.dataclass(frozen=True)
class EpisodeArtifacts:
    rollout_video: pathlib.Path
    image_dir: pathlib.Path
    attention_dir: pathlib.Path

    @property
    def stage_dir(self) -> pathlib.Path:
        return self.image_dir / "annotation"

    @property
    def wrist_video(self) -> pathlib.Path:
        return self.image_dir / "wristview.mp4"

    @classmethod
    def from_args(cls, args: Args, task_description: str, episode_index: int, suffix: str) -> "EpisodeArtifacts":
        return cls(
            rollout_video=ArtifactPaths.build(
                args.video_out_path,
                task_description,
                episode_index,
                suffix,
                prefix="rollout_",
            ),
            image_dir=ArtifactPaths.build_rollout_dir(
                args.image_out_path,
                task_description,
                episode_index,
                suffix,
            ),
            attention_dir=ArtifactPaths.build_rollout_dir(
                args.attention_out_path,
                task_description,
                episode_index,
                suffix,
            )
            / "attention",
        )

    @classmethod
    def from_result(
        cls,
        args: Args,
        task_description: str,
        episode_index: int,
        *,
        success: bool,
    ) -> "EpisodeArtifacts":
        return cls.from_args(args, task_description, episode_index, "success" if success else "failure")

    @staticmethod
    def _write_video(path: pathlib.Path, frames, *, fps: int) -> pathlib.Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path, [np.asarray(x) for x in frames], fps=fps)
        return path

    def write_rollout(self, frames, *, fps: int) -> pathlib.Path:
        return self._write_video(self.rollout_video, frames, fps=fps)

    def write_images(self, agentview_frames, wristview_frames, frame_indices: List[int]) -> pathlib.Path:
        """Write the two LIBERO camera streams as per-step PNG images."""
        for view_name in ("agentview", "wristview"):
            view_dir = self.image_dir / view_name
            if view_dir.exists():
                for old_image_file in view_dir.glob("*.png"):
                    old_image_file.unlink()
        for view_name, frames in (("agentview", agentview_frames), ("wristview", wristview_frames)):
            view_dir = self.image_dir / view_name
            view_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in frame_indices:
                frame = frames[frame_index]
                imageio.imwrite(view_dir / f"{frame_index:06d}.png", np.asarray(frame))
        return self.image_dir

    @staticmethod
    def _tensorize_attention(value):
        import torch

        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(value))
        if isinstance(value, dict):
            return {key: EpisodeArtifacts._tensorize_attention(item) for key, item in value.items()}
        if isinstance(value, list):
            return [EpisodeArtifacts._tensorize_attention(item) for item in value]
        if isinstance(value, tuple):
            return tuple(EpisodeArtifacts._tensorize_attention(item) for item in value)
        return value

    def write_attention(
        self,
        attention_per_frame,
        *,
        frame_indices: List[int],
        task_description: str,
        episode_index: int,
    ) -> pathlib.Path:
        """Write one torch-loadable attention file for every saved image frame.

        The file name is the image frame index, so for example
        ``attention/000012.pt`` corresponds to ``agentview/000012.png`` and
        ``wristview/000012.png``.  With ``replan_steps > 1`` the same model
        attention is intentionally reused for the action chunk; the payload
        records the original policy-inference frame in ``source_env_step``.
        """
        import torch

        self.attention_dir.mkdir(parents=True, exist_ok=True)
        for old_attention_file in self.attention_dir.glob("*.pt"):
            old_attention_file.unlink()
        for frame_index in frame_indices:
            item = attention_per_frame[frame_index]
            payload = {
                "format_version": 2,
                "task_description": str(task_description),
                "episode_index": int(episode_index),
                "frame_index": int(frame_index),
                "env_step": None if item is None else int(item["env_step"]),
                "source_env_step": None if item is None else int(item["source_env_step"]),
                "attention_reused": False if item is None else bool(item["attention_reused"]),
                "attention": None if item is None else self._tensorize_attention(item["attention"]),
            }
            if os.environ.get("OPENPI_ATTENTION_DEBUG", "0") == "1" and payload["attention"] is not None:
                snapshots = payload["attention"].get("snapshots", [])
                if snapshots:
                    origin = snapshots[0].get("origin", {})
                    for layer in ("9", "10", "11", "12"):
                        if layer in origin:
                            value = origin[layer]
                            print(
                                "client attention before torch.save:",
                                frame_index,
                                layer,
                                "finite=",
                                torch.isfinite(value).all().item(),
                                "shape=",
                                tuple(value.shape),
                                "dtype=",
                                value.dtype,
                                flush=True,
                            )
            torch.save(payload, self.attention_dir / f"{frame_index:06d}.pt")
        return self.attention_dir

    @classmethod
    def _attention_capture_is_current(cls, attention_dir: pathlib.Path) -> bool:
        """Reject stale attention files produced before the numerical fixes."""
        attention_files = sorted(attention_dir.glob("*.pt"), key=lambda path: path.name)
        if not attention_files:
            return False
        import torch

        try:
            payload = torch.load(attention_files[0], map_location="cpu", weights_only=False)
        except TypeError as exc:
            if "weights_only" not in str(exc):
                return False
            payload = torch.load(attention_files[0], map_location="cpu")
        return (payload.get("attention") or {}).get("capture_impl") == ATTENTION_CAPTURE_IMPL

    @classmethod
    def find_existing(
        cls,
        args: Args,
        task_description: str,
        episode_index: int,
    ) -> Optional["ExistingEpisodeResult"]:
        for success in (True, False):
            artifacts = cls.from_result(args, task_description, episode_index, success=success)
            agentview_dir = artifacts.image_dir / "agentview"
            wristview_dir = artifacts.image_dir / "wristview"
            agentview_frames = {path.stem for path in agentview_dir.glob("*.png")}
            wristview_frames = {path.stem for path in wristview_dir.glob("*.png")}
            attention_frames = {path.stem for path in artifacts.attention_dir.glob("*.pt")}
            image_complete = bool(agentview_frames) and agentview_frames == wristview_frames
            attention_complete = (
                image_complete
                and agentview_frames == attention_frames
                and cls._attention_capture_is_current(artifacts.attention_dir)
            )
            stage_complete = not args.export_stage_data or (
                artifacts.wrist_video.exists()
                and export_is_complete(artifacts.stage_dir, args.stage_plan_path)
            )
            if stage_complete and args.export_stage_data:
                try:
                    metadata = json.loads((artifacts.stage_dir / "metadata.json").read_text())
                    frames = json.loads((artifacts.stage_dir / "frame_map.json").read_text())
                    expected_frames = {f"{row['obs_id']:06d}" for row in frames if row["agentview_png"] is not None}
                    stage_complete = (
                        agentview_frames == expected_frames
                        and metadata["image_interval"] == args.attention_dump_interval
                        and metadata["replan_steps"] == args.replan_steps
                        and metadata["num_steps_wait"] == args.num_steps_wait
                        and metadata["episode_seed"] == args.seed + 1000 * metadata["task_id"] + episode_index
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    stage_complete = False
            if artifacts.rollout_video.exists() and attention_complete and stage_complete:
                return ExistingEpisodeResult(success=success, video_path=artifacts.rollout_video)
        return None


@dataclasses.dataclass(frozen=True)
class ExistingEpisodeResult:
    success: bool
    video_path: pathlib.Path


class PolicyIO:
    @staticmethod
    def quat2axisangle(quat):
        """Convert a quaternion to axis-angle."""
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    @classmethod
    def build_input(
        cls,
        obs: Dict[str, Any],
        img: np.ndarray,
        wrist_img: np.ndarray,
        prompt: str,
    ) -> Dict[str, Any]:
        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    cls.quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": prompt,
        }

    @staticmethod
    def prepare_images(obs: Dict[str, Any], resize_size: int) -> Tuple[np.ndarray, np.ndarray]:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
        return img, wrist_img


def _planned_actions(response: Dict[str, Any], replan_steps: int):
    action_chunk = response["actions"]
    if len(action_chunk) < replan_steps:
        raise RuntimeError(
            f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
        )
    return action_chunk[:replan_steps]


def _episode_extra(
    args: Args, suite_name: str, max_steps: int, model_prompt: str, *, skipped: bool = False
) -> Dict[str, Any]:
    extra: Dict[str, Any] = {
        "max_steps": max_steps,
        "num_steps_wait": args.num_steps_wait,
        "suite": suite_name,
        "prompt_sent": model_prompt,
    }
    if skipped:
        extra["skipped"] = True
    return extra


@dataclasses.dataclass
class EpisodeFrameCollector:
    replay_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    wrist_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    attention_per_frame: List[Optional[Dict[str, Any]]] = dataclasses.field(default_factory=list)

    def append_step(
        self,
        img: np.ndarray,
        wrist_img: np.ndarray,
        *,
        attention: Optional[Dict[str, Any]],
        env_step: int,
        source_env_step: int,
    ) -> None:
        self.replay_images.append(img)
        self.wrist_images.append(wrist_img)
        self.attention_per_frame.append(
            None
            if attention is None
            else {
                "env_step": int(env_step),
                "source_env_step": int(source_env_step),
                "attention_reused": int(env_step) != int(source_env_step),
                "attention": attention,
            }
        )

    def write_artifacts(
        self,
        artifacts: EpisodeArtifacts,
        *,
        task_description: str,
        episode_index: int,
        dump_interval: int,
        include_terminal: bool = False,
    ) -> pathlib.Path:
        frame_indices = list(range(0, len(self.replay_images), dump_interval))
        if include_terminal:
            frame_indices = sorted(set(frame_indices + [len(self.replay_images) - 1]))
            artifacts._write_video(artifacts.wrist_video, self.wrist_images, fps=10)
        artifacts.write_rollout(self.replay_images, fps=10)
        artifacts.write_images(self.replay_images, self.wrist_images, frame_indices)
        artifacts.write_attention(
            self.attention_per_frame,
            frame_indices=frame_indices,
            task_description=task_description,
            episode_index=episode_index,
        )
        return artifacts.rollout_video


@dataclasses.dataclass
class RunStats:
    episodes: int = 0
    successes: int = 0

    def record(self, *, success: bool) -> None:
        self.episodes += 1
        if success:
            self.successes += 1

    @property
    def rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return float(self.successes) / float(self.episodes)


@dataclasses.dataclass(frozen=True)
class EpisodeRunResult:
    success: bool
    steps_taken: int
    video_path: pathlib.Path
    image_dir: pathlib.Path
    attention_path: pathlib.Path
    last_error: Optional[str] = None


class EpisodeRunner:
    def __init__(
        self,
        args: Args,
        client,
        env,
        *,
        task_id: int,
        task_description: str,
        initial_states,
        model_prompt: str,
        max_steps: int,
        suite_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_bddl_file: Optional[str] = None,
    ):
        self.args = args
        self.client = client
        self.env = env
        self.task_id = task_id
        self.task_description = task_description
        self.initial_states = initial_states
        self.model_prompt = model_prompt
        self.max_steps = max_steps
        self.suite_name = suite_name or args.task_suite_name
        self.task_name = task_name
        self.task_bddl_file = task_bddl_file

    def _episode_seed(self, episode_idx: int) -> int:
        return int(self.args.seed + 1000 * int(self.task_id) + int(episode_idx))

    def _reset_episode(self, episode_idx: int):
        if len(self.initial_states) == 0:
            raise RuntimeError(f"No initial states for task {self.task_id}")

        episode_seed = self._episode_seed(episode_idx)
        self.env.seed(episode_seed)
        self.env.reset()

        rng = np.random.RandomState(episode_seed)
        init_idx = int(rng.randint(len(self.initial_states)))
        self.init_state_index = init_idx
        return self.env.set_init_state(self.initial_states[init_idx])

    def _infer_actions(self, obs: Dict[str, Any], img: np.ndarray, wrist_img: np.ndarray) -> Dict[str, Any]:
        element = PolicyIO.build_input(obs, img, wrist_img, self.model_prompt)
        return self.client.infer(element)

    def run(self, episode_idx: int) -> EpisodeRunResult:
        obs = self._reset_episode(episode_idx)
        action_plan = collections.deque()
        frame_collector = EpisodeFrameCollector()
        done, last_error = False, None
        t, executed = 0, 0
        current_attention = None
        attention_source_env_step = None
        policy_call_id, chunk_index = -1, 0
        recorder = None
        termination_reason = "timeout"

        # Preserve the original warm-up before the first policy observation.
        for _ in range(self.args.num_steps_wait):
            obs, _, done, _ = self.env.step(LIBERO_DUMMY_ACTION)
            t += 1

        if self.args.export_stage_data:
            recorder = StageRecorder(self.env, {
                "suite": self.suite_name, "task_id": self.task_id, "task_name": self.task_name,
                "task_bddl_reference": self.task_bddl_file,
                "instruction": self.task_description, "prompt_sent": self.model_prompt,
                "category": self.args.category, "episode_index": episode_idx,
                "episode_seed": self._episode_seed(episode_idx), "init_state_index": self.init_state_index,
                "initial_state_sha256": hashlib.sha256(
                    np.asarray(self.initial_states[self.init_state_index]).tobytes()).hexdigest(),
                "num_steps_wait": self.args.num_steps_wait, "replan_steps": self.args.replan_steps,
                "repository_revision": revision(REPO_ROOT),
                "libero_plus_revision": revision(REPO_ROOT / "third_party" / "LIBERO-plus"),
                "image_transform": {"flip_axes": [0, 1], "resize_with_pad": self.args.resize_size,
                                    "source_shape": list(obs["agentview_image"].shape)},
                "action_semantics": "post-policy-transform action exactly passed to env.step; see action_spec and controller_config",
            }, self.args.stage_plan_path)

        def append_observation():
            img, wrist = PolicyIO.prepare_images(obs, self.args.resize_size)
            # StageRecorder can fail without adding a mismatched video frame.
            recorder.capture(obs, t)
            frame_collector.append_step(img, wrist, attention=None, env_step=t, source_env_step=t)

        if recorder is not None:
            append_observation()

        while executed < self.max_steps:
            try:
                if recorder is not None:
                    img, wrist_img = frame_collector.replay_images[-1], frame_collector.wrist_images[-1]
                else:
                    img, wrist_img = PolicyIO.prepare_images(obs, self.args.resize_size)
                if not action_plan:
                    response = self._infer_actions(obs, img, wrist_img)
                    planned = _planned_actions(response, self.args.replan_steps)
                    current_attention = response.get("attention")
                    attention_source_env_step = t
                    action_plan.extend(planned)
                    policy_call_id = recorder.record_policy_call(response, self.args.replan_steps) if recorder else policy_call_id + 1
                    chunk_index = 0
                action = action_plan.popleft()
                if recorder is not None:
                    frame_collector.attention_per_frame[-1] = None if current_attention is None else {
                        "attention": current_attention, "env_step": t,
                        "source_env_step": attention_source_env_step,
                        "attention_reused": t != attention_source_env_step,
                    }
                else:
                    frame_collector.append_step(img, wrist_img, attention=current_attention, env_step=t,
                                                source_env_step=attention_source_env_step)
                obs, _, done, _ = self.env.step(action.tolist())
                executed += 1
                t += 1
                if recorder is not None:
                    recorder.record_action(action, policy_call_id, chunk_index, done)
                    append_observation()
                chunk_index += 1
                if done:
                    termination_reason = "success"
                    break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                termination_reason = "error"
                logging.exception("Episode failed")
                break

        artifacts = EpisodeArtifacts.from_result(self.args, self.task_description, episode_idx, success=done)
        # Invalidate a previous completion marker before rewriting any artifacts.
        if recorder is not None:
            (artifacts.stage_dir / "manifest.json").unlink(missing_ok=True)
        video_path = frame_collector.write_artifacts(
            artifacts, task_description=self.task_description, episode_index=episode_idx,
            dump_interval=self.args.attention_dump_interval, include_terminal=recorder is not None,
        )
        if recorder is not None:
            recorder.write(artifacts.stage_dir, success=bool(done), termination_reason=termination_reason,
                           error=last_error, image_interval=self.args.attention_dump_interval,
                           video_paths={"agentview": str(video_path.resolve()),
                                        "wristview": str(artifacts.wrist_video.resolve())})
        return EpisodeRunResult(
            success=bool(done), steps_taken=t, video_path=video_path,
            image_dir=artifacts.image_dir, attention_path=artifacts.attention_dir, last_error=last_error,
        )


def _parse_task_ids(expr: Optional[str], upper: int) -> List[int]:
    """
    Parse expressions like "5", "10-20", "0,7,10-12" into a sorted, de-duplicated
    list of integer task ids within [0, upper-1]. Ranges are inclusive.
    """
    if expr is None or str(expr).strip() == "":
        return list(range(upper))
    s = str(expr).replace(" ", "")
    out: Set[int] = set()
    parts = [p for p in s.split(",") if p != ""]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except ValueError as exc:
                raise ValueError(f'Invalid range "{part}" in --task-ids.') from exc
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                if 0 <= i < upper:
                    out.add(i)
                else:
                    raise ValueError(f"Task id {i} out of range [0, {upper - 1}] for suite (from range {part}).")
        else:
            try:
                i = int(part)
            except ValueError as exc:
                raise ValueError(f'Invalid id "{part}" in --task-ids.') from exc
            if 0 <= i < upper:
                out.add(i)
            else:
                raise ValueError(f"Task id {i} out of range [0, {upper - 1}] for suite.")
    return sorted(out)


def _prompt_for_model(args: Args, task_description: str) -> str:
    """Return the prompt string actually sent to the model (may be normalized)."""
    prompt = str(task_description).strip()
    if not prompt:
        return prompt

    tokens = [t for t in prompt.split() if t]
    while tokens:
        last = tokens[-1]

        if args.prompt_strip_trailing_id_with_prev and last.isdigit():
            tokens = tokens[:-2] if len(tokens) >= 2 else tokens[:-1]
            continue

        if args.prompt_strip_trailing_word_ending_with_digit and last[-1].isdigit():
            tokens = tokens[:-1]
            continue

        break

    return " ".join(tokens).strip()


class ResultsStore:
    def __init__(self, args: Args):
        self.args = args
        self.path = pathlib.Path(args.results_json_path)

    @staticmethod
    def _default_data() -> Dict[str, Any]:
        return {
            "meta": {},
            "success": [],
            "failure": [],
            "running_counts": {
                "total_episodes": 0,
                "total_successes": 0,
                "success_rate": 0.0,
            },
        }

    @staticmethod
    def _atomic_write_json(obj: Dict[str, Any], path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.parent / f".{path.name}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            if os.path.exists(path):
                os.replace(tmp_path, path)
            else:
                tmp_path.rename(path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise e

    def _read_json(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._default_data()
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return self._default_data()

    def initialize(self, selected) -> pathlib.Path:
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        data = self._read_json()
        meta = data.setdefault("meta", {})
        meta.setdefault("created_at", now_iso)
        meta.update(
            {
                "updated_at": now_iso,
                "task_suite_name": self.args.task_suite_name,
                "selected_task_ids": selected,
                "host": self.args.host,
                "port": self.args.port,
                "resize_size": self.args.resize_size,
                "replan_steps": self.args.replan_steps,
                "attention_dump_interval": self.args.attention_dump_interval,
                "export_stage_data": self.args.export_stage_data,
                "stage_plan_path": self.args.stage_plan_path,
                "num_trials_per_task": self.args.num_trials_per_task,
                "num_rollouts": self.args.num_rollouts,
                "seed": self.args.seed,
                "video_out_path": str(self.args.video_out_path),
                "image_out_path": str(self.args.image_out_path),
                "attention_out_path": str(self.args.attention_out_path),
            }
        )

        data.setdefault("success", [])
        data.setdefault("failure", [])
        data.setdefault(
            "running_counts",
            {"total_episodes": 0, "total_successes": 0, "success_rate": 0.0},
        )
        self._atomic_write_json(data, self.path)
        return self.path

    def record_episode(
        self,
        *,
        task_id: int,
        task_description: str,
        episode_index: int,
        steps_taken: int,
        success: bool,
        video_path: pathlib.Path,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        data = self._read_json()

        bucket = "success" if success else "failure"
        # A rerun may replace a legacy rollout or change success status.
        def matches(existing_record):
            return (existing_record.get("task_id") == task_id
                    and existing_record.get("episode_index") == episode_index
                    and existing_record.get("task_description") == task_description
                    and existing_record.get("extra", {}).get("suite") in (None, (extra or {}).get("suite")))
        if (extra or {}).get("skipped"):
            if any(matches(r) for r in data.get(bucket, [])):
                return
        for old_bucket in ("success", "failure"):
            data[old_bucket] = [r for r in data.get(old_bucket, []) if not matches(r)]

        record: Dict[str, Any] = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "task_id": int(task_id),
            "task_description": str(task_description),
            "episode_index": int(episode_index),
            "steps_taken": int(steps_taken),
            "video": str(video_path),
        }
        if error:
            record["error"] = str(error)
        if extra:
            record["extra"] = extra

        data.setdefault(bucket, []).append(record)

        rc = data.setdefault(
            "running_counts",
            {"total_episodes": 0, "total_successes": 0, "success_rate": 0.0},
        )
        rc["total_episodes"] = len(data["success"]) + len(data["failure"])
        rc["total_successes"] = len(data["success"])
        rc["success_rate"] = float(rc["total_successes"]) / max(1, rc["total_episodes"])
        data.setdefault("meta", {})
        data["meta"]["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self._atomic_write_json(data, self.path)


def _prepare_output_dirs(args: Args) -> None:
    if args.attention_dump_interval <= 0:
        raise ValueError("attention_dump_interval must be positive")
    if args.replan_steps <= 0 or args.num_steps_wait < 0:
        raise ValueError("replan_steps must be positive and num_steps_wait non-negative")
    if args.stage_plan_path:
        with open(args.stage_plan_path, encoding="utf-8") as f:
            plan = json.load(f)
        if not isinstance(plan.get("tasks"), list):
            raise ValueError("stage_plan_path must contain a tasks list")
        if plan.get("task_count") != len(plan["tasks"]):
            logging.warning("Stage plan task_count differs from actual tasks length; saving original plan unchanged")
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.image_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.attention_out_path).mkdir(parents=True, exist_ok=True)


def _select_task_ids_for_suite(
    args: Args,
    suite_name: str,
    task_suite,
    classification_by_suite: Dict[str, TaskClassificationSelection],
) -> List[int]:
    num_tasks_in_suite = task_suite.n_tasks
    selected_task_ids = _parse_task_ids(args.task_ids, num_tasks_in_suite)
    classification_selection = classification_by_suite.get(suite_name)

    if classification_selection is None:
        return selected_task_ids

    invalid_ids = {tid for tid in classification_selection.task_ids_0based if tid < 0 or tid >= num_tasks_in_suite}
    if invalid_ids:
        logging.warning(
            "[%s] classification contains %d out-of-range task ids for current suite size %d. "
            "This usually means the loaded LIBERO suite does not match %s.",
            suite_name,
            len(invalid_ids),
            num_tasks_in_suite,
            _resolve_task_classification_path(args),
        )

    return [tid for tid in selected_task_ids if classification_selection.matches(tid, task_suite.get_task(tid).name)]


def eval_libero(args: Args) -> None:
    if args.num_rollouts is not None and args.num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive when specified")
    if args.category is not None:
        args.results_json_path = _resolve_results_json_path(args)
        logging.info(f"Category-specific results will be saved to: {args.results_json_path}")

    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_names = _get_suite_names(args.task_suite_name)
    _prepare_output_dirs(args)
    classification_by_suite = _load_classification_by_suite(args, suite_names)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    results_store = ResultsStore(args)

    selected_map: Dict[str, List[int]] = {}
    overall_stats = RunStats()
    executed_rollouts = 0

    for suite_name in suite_names:
        key = suite_name.lower()
        if key not in benchmark_dict:
            available = sorted(benchmark_dict.keys())
            raise ValueError(
                f"Unknown task suite: {suite_name}. "
                f"Available suites in your installed `libero` are: {available}. "
                "Note: LIBERO-Plus supports `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`. "
                "Double-check (1) your `PYTHONPATH` points to the intended LIBERO/LIBERO-plus checkout, "
                "and (2) the suite names you pass to the evaluator match `benchmark.get_benchmark_dict()`."
            )

        task_suite = benchmark_dict[key]()
        num_tasks_in_suite = task_suite.n_tasks
        max_steps = _get_max_steps_for_suite(suite_name)
        selected_task_ids = _parse_task_ids(args.task_ids, num_tasks_in_suite)
        filtered_task_ids = _select_task_ids_for_suite(args, suite_name, task_suite, classification_by_suite)

        selected_map[suite_name] = filtered_task_ids

        logging.info(
            f"Evaluating suite: {suite_name} | tasks: {num_tasks_in_suite} | "
            f"category: {args.category or 'ALL'} | selected ids: {filtered_task_ids[:10]}"
            f"{' ...' if len(filtered_task_ids) > 10 else ''}"
        )
        logging.info(f"[{suite_name}] category matched: {len(filtered_task_ids)}/{len(selected_task_ids)}")

        results_store.initialize(selected_map)
        suite_stats = RunStats()

        for task_id in tqdm.tqdm(filtered_task_ids, total=len(filtered_task_ids)):
            if args.num_rollouts is not None and executed_rollouts >= args.num_rollouts:
                break
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            try:
                model_prompt = _prompt_for_model(args, task_description)
                logging.info(f"[DEBUG] Task {task_id} description: '{task_description}'")
                if model_prompt != str(task_description):
                    logging.info(
                        f"[DEBUG] Normalized prompt (sent to model): '{model_prompt}' (from: '{task_description}')"
                    )

                episode_runner = EpisodeRunner(
                    args,
                    client,
                    env,
                    task_id=task_id,
                    task_description=task_description,
                    initial_states=initial_states,
                    model_prompt=model_prompt,
                    max_steps=max_steps,
                    suite_name=suite_name, task_name=task.name,
                    task_bddl_file=str(pathlib.Path(task.problem_folder) / task.bddl_file),
                )

                for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"episodes for task {task_id}"):
                    existing = EpisodeArtifacts.find_existing(args, task_description, episode_idx)
                    if existing is not None:
                        status = "SUCCESS" if existing.success else "FAILURE"
                        logging.info(f"⏭️  Skipping task {task_id} episode {episode_idx}: already completed ({status})")
                        suite_stats.record(success=existing.success)
                        overall_stats.record(success=existing.success)
                        results_store.record_episode(
                            task_id=task_id,
                            task_description=task_description,
                            episode_index=episode_idx,
                            steps_taken=-1,
                            success=existing.success,
                            video_path=existing.video_path,
                            error=None,
                            extra=_episode_extra(args, suite_name, max_steps, model_prompt, skipped=True),
                        )
                        continue

                    if args.num_rollouts is not None and executed_rollouts >= args.num_rollouts:
                        break

                    logging.info(f"\nTask: {task_description} | episode {episode_idx + 1}/{args.num_trials_per_task}")
                    logging.info(f"Starting episode {episode_idx + 1}...")
                    executed_rollouts += 1
                    result = episode_runner.run(episode_idx)
                    suite_stats.record(success=result.success)
                    overall_stats.record(success=result.success)

                    logging.info(f"Success: {result.success}")
                    logging.info(f"# episodes completed so far: {overall_stats.episodes}")
                    logging.info(f"# successes: {overall_stats.successes} ({overall_stats.rate * 100:.1f}%)")

                    results_store.record_episode(
                        task_id=task_id,
                        task_description=task_description,
                        episode_index=episode_idx,
                        steps_taken=result.steps_taken,
                        success=result.success,
                        video_path=result.video_path,
                        error=result.last_error,
                        extra={
                            **_episode_extra(args, suite_name, max_steps, model_prompt),
                            "image_dir": str(result.image_dir),
                            "attention_path": str(result.attention_path),
                            "annotation_dir": str(result.image_dir / "annotation") if args.export_stage_data else None,
                        },
                    )
            finally:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()

        if args.num_rollouts is not None and executed_rollouts >= args.num_rollouts:
            logging.info("Reached global rollout limit: %d", args.num_rollouts)
            break

        logging.info(f"[Suite {suite_name}] success rate: {suite_stats.rate}")
        logging.info(f"[Suite {suite_name}] overall success rate: {overall_stats.rate}")

    logging.info(f"Total success rate: {overall_stats.rate}")
    logging.info(f"Total episodes: {overall_stats.episodes}")


def _get_libero_env(task, resolution, seed):
    """Create the LIBERO environment and return it with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _normalize_cli_args(argv: List[str]) -> List[str]:
    """Accept both current `--flag` and legacy `--args.flag` spellings."""
    normalized = []
    for arg in argv:
        if arg.startswith("--args."):
            normalized.append("--" + arg[len("--args.") :])
        else:
            normalized.append(arg)
    return normalized


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args, args=_normalize_cli_args(sys.argv[1:])))
