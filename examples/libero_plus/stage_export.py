"""Control-step observations for offline stage annotation (no policy changes).

Derived poses / contacts are computed in a separate MjData at the saved qpos.
This avoids both stale post-mj_step kinematics and changing rollout solver state.
"""

import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess

import numpy as np

FORMAT_VERSION = 1


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def contact_flags(contacts, target_ids, groups):
    """Report detected and active constraint contacts separately; neither proves a grasp."""
    result = {}
    for label, ids in groups.items():
        if not ids:
            result[label] = None
            result[label + "_active"] = None
            continue
        matches = [c for c in contacts if
                   (c["geom1_id"] in ids and c["geom2_id"] in target_ids) or
                   (c["geom2_id"] in ids and c["geom1_id"] in target_ids)]
        result[label] = bool(matches)
        result[label + "_active"] = any(c["efc_address"] >= 0 for c in matches)
    for suffix in ("", "_active"):
        left, right = result["left_finger" + suffix], result["right_finger" + suffix]
        result["bilateral" + suffix] = None if left is None or right is None else left and right
    return result


class StageRecorder:
    def __init__(self, env, metadata, stage_plan_path=None):
        import mujoco

        self.mj = mujoco
        self.env = env
        self.core = env.env
        self.model = env.sim.model._model
        self.scratch = mujoco.MjData(self.model)
        self.states, self.records, self.actions = [], [], []
        self.arrays = []
        self.policy_calls = []
        self.stage_plan_text = None
        if stage_plan_path:
            self.stage_plan_text = pathlib.Path(stage_plan_path).read_text(encoding="utf-8")
            json.loads(self.stage_plan_text)
        self.scene_xml = env.sim.model.get_xml()

        def names(kind, n):
            return [mujoco.mj_id2name(self.model, kind, i) for i in range(n)]

        self.geom_names = names(mujoco.mjtObj.mjOBJ_GEOM, self.model.ngeom)
        self.joint_names = names(mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)
        self.body_names = names(mujoco.mjtObj.mjOBJ_BODY, self.model.nbody)
        self.objects = {}
        for kind, mapping in (("object", self.core.objects_dict), ("fixture", self.core.fixtures_dict)):
            for name, obj in mapping.items():
                self.objects[name] = {
                    "kind": kind, "root_body": obj.root_body,
                    "body_id": self._id(mujoco.mjtObj.mjOBJ_BODY, obj.root_body),
                    "contact_geom_ids": [self._id(mujoco.mjtObj.mjOBJ_GEOM, g) for g in obj.contact_geoms],
                    "joint_names": list(obj.joints),
                }
        gripper = self.core.robots[0].gripper
        self.groups = {
            "left_finger": {self._id(mujoco.mjtObj.mjOBJ_GEOM, g) for g in gripper.important_geoms.get("left_fingerpad", [])},
            "right_finger": {self._id(mujoco.mjtObj.mjOBJ_GEOM, g) for g in gripper.important_geoms.get("right_fingerpad", [])},
            "any_gripper": {self._id(mujoco.mjtObj.mjOBJ_GEOM, g) for g in gripper.contact_geoms},
        }
        self.eef_site = self._id(mujoco.mjtObj.mjOBJ_SITE, gripper.important_sites["grip_site"])
        self.gripper_joints = list(gripper.joints)
        versions = {"python": platform.python_version()}
        for package in ("mujoco", "robosuite", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        bddl_path = pathlib.Path(self.core.bddl_file_name)
        self.bddl_text = bddl_path.read_text(encoding="utf-8")
        self.metadata = {
            **metadata, "format_version": FORMAT_VERSION, "versions": versions,
            "bddl_file": str(bddl_path), "bddl_sha256": hashlib.sha256(self.bddl_text.encode()).hexdigest(),
            "parsed_problem": getattr(self.core, "parsed_problem", None),
            "objects": self.objects, "geom_names": self.geom_names,
            "body_names": self.body_names, "joint_names": self.joint_names,
            "joint_qposadr": self.model.jnt_qposadr.copy(),
            "joint_dofadr": self.model.jnt_dofadr.copy(),
            "gripper_geom_groups": {k: sorted(v) for k, v in self.groups.items()},
            "gripper_joint_names": self.gripper_joints,
            "eef_site": gripper.important_sites["grip_site"],
            "control_freq_hz": float(self.core.control_freq),
            "physics_timestep_seconds": float(self.model.opt.timestep),
            "controller_config": getattr(self.core.robots[0], "controller_config", None),
            "controller_parameters": {key: getattr(self.core.robots[0].controller, key, None)
                                      for key in ("input_min", "input_max", "output_min", "output_max",
                                                  "control_delta", "impedance_mode", "kp", "damping_ratio")},
            "controller_class": type(self.core.robots[0].controller).__name__,
            "action_spec": getattr(self.core, "action_spec", None),
            "coordinates": {"position": "world, meters", "rotation": "3x3, local-to-world",
                            "joint_hinge": "radians", "joint_slide": "meters",
                            "contact_force": "recomputed, contact frame, [normal,tangent1,tangent2,torques...]"},
            "state_layout": "env.get_sim_state(); robosuite native binding: time,qpos,qvel; not a full checkpoint",
            "contact_semantics": "snapshot at observation qpos, recomputed in separate MjData; not substep history or stable-grasp labels",
            "stage_plan_sha256": None if self.stage_plan_text is None else hashlib.sha256(self.stage_plan_text.encode()).hexdigest(),
        }

    def _id(self, kind, name):
        idx = self.mj.mj_name2id(self.model, kind, name)
        if idx < 0:
            raise ValueError("Missing simulation entity: " + name)
        return idx

    def capture(self, obs, env_step):
        source = self.env.sim.data._data
        d = self.scratch
        d.time = source.time
        for key in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata",
                    "qfrc_applied", "xfrc_applied", "qacc_warmstart"):
            getattr(d, key)[:] = getattr(source, key)
        if hasattr(d, "eq_active"):
            d.eq_active[:] = source.eq_active
        self.mj.mj_forward(self.model, d)
        contacts = []
        for i in range(d.ncon):
            c = d.contact[i]
            force = np.zeros(6)
            self.mj.mj_contactForce(self.model, d, i, force)
            contacts.append({
                "geom1_id": int(c.geom1), "geom2_id": int(c.geom2),
                "distance": float(c.dist), "efc_address": int(c.efc_address),
                "position_world": c.pos.copy(), "frame": c.frame.copy().reshape(3, 3),
                "force_torque": force,
            })
        eef_pos = d.site_xpos[self.eef_site].copy()
        eef_rot = d.site_xmat[self.eef_site].reshape(3, 3).copy()
        objects = {}
        for name, obj in self.objects.items():
            pos = d.xpos[obj["body_id"]].copy()
            rot = d.xmat[obj["body_id"]].reshape(3, 3).copy()
            objects[name] = {
                "position_world": pos, "rotation_world": rot,
                "position_in_gripper": eef_rot.T @ (pos - eef_pos),
                "rotation_in_gripper": eef_rot.T @ rot,
                "contacts": contact_flags(contacts, set(obj["contact_geom_ids"]), self.groups),
            }
        frame = len(self.records)
        record = {
            "obs_id": frame, "video_frame_id": frame, "env_step": int(env_step),
            "sim_time": float(d.time), "eef_position_world": eef_pos, "eef_rotation_world": eef_rot,
            "gripper_qpos": np.concatenate([d.joint(name).qpos for name in self.gripper_joints]),
            "policy_observation_state": {k: np.asarray(obs[k]).copy() for k in
                                         ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")},
            "objects": objects, "contacts": contacts,
            "policy_call_id": None, "action_chunk_index": None,
        }
        state = self.env.get_sim_state().copy()
        # Convert now: no references to mutable simulator / observation buffers.
        record = jsonable(record)
        self.arrays.append({"qpos": source.qpos.copy(), "qvel": source.qvel.copy(),
                            "ctrl": source.ctrl.copy(), "act": source.act.copy(),
                            "body_position_world": d.xpos.copy(),
                            "body_rotation_world": d.xmat.copy().reshape(-1, 3, 3)})
        self.states.append(state)
        self.records.append(record)

    def record_policy_call(self, response, replan_steps):
        call_id = len(self.policy_calls)
        self.policy_calls.append({"policy_call_id": call_id, "input_obs_id": len(self.records) - 1,
                                  "predicted_actions": jsonable(response["actions"]),
                                  "replan_steps": replan_steps})
        return call_id

    def record_action(self, action, policy_call_id, chunk_index, done):
        source = len(self.records) - 1
        self.actions.append({"action_id": len(self.actions), "source_obs_id": source,
                             "destination_obs_id": source + 1, "action": jsonable(action),
                             "policy_call_id": policy_call_id, "action_chunk_index": chunk_index,
                             "done_after_action": bool(done)})
        self.records[-1].update(policy_call_id=policy_call_id, action_chunk_index=chunk_index)

    def write(self, directory, *, success, termination_reason, error, image_interval, video_paths):
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "manifest.json"
        marker.unlink(missing_ok=True)
        np.savez_compressed(directory / "states.npz", states=np.stack(self.states),
                            obs_id=np.arange(len(self.states)),
                            env_step=np.array([r["env_step"] for r in self.records]),
                            sim_time=np.array([r["sim_time"] for r in self.records]),
                            **{k: np.stack([a[k] for a in self.arrays]) for k in self.arrays[0]})
        for name, rows in (("observations", self.records), ("actions", self.actions), ("policy_calls", self.policy_calls)):
            with (directory / (name + ".jsonl")).open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
        frame_map = [{"obs_id": i, "video_frame_id": i, "env_step": r["env_step"],
                      "sim_time": r["sim_time"], "last_observation": i == len(self.records) - 1,
                      "agentview_png": "../agentview/%06d.png" % i if i % image_interval == 0 or i == len(self.records) - 1 else None,
                      "wristview_png": "../wristview/%06d.png" % i if i % image_interval == 0 or i == len(self.records) - 1 else None}
                     for i, r in enumerate(self.records)]
        write_json(directory / "frame_map.json", frame_map)
        write_json(directory / "metadata.json", {**self.metadata, "success": success,
                   "termination_reason": termination_reason, "error": error,
                   "video_paths": video_paths, "video_fps": 10, "image_interval": image_interval,
                   "executed_actions": len(self.actions), "observation_count": len(self.records)})
        (directory / "scene.xml").write_text(self.scene_xml, encoding="utf-8")
        (directory / "task.bddl").write_text(self.bddl_text, encoding="utf-8")
        if self.stage_plan_text is not None:
            (directory / "stage_plan.json").write_text(self.stage_plan_text, encoding="utf-8")
        # Written last; interrupted / failed exports never look complete.
        files = ["states.npz", "observations.jsonl", "actions.jsonl", "policy_calls.jsonl",
                 "frame_map.json", "metadata.json", "scene.xml", "task.bddl"]
        if self.stage_plan_text is not None:
            files.append("stage_plan.json")
        write_json(marker, {"format_version": FORMAT_VERSION,
                   "complete": error is None and len(self.records) == len(self.actions) + 1,
                   "observation_count": len(self.records), "action_count": len(self.actions),
                   "stage_plan_sha256": self.metadata["stage_plan_sha256"],
                   "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}})


def export_is_complete(directory, stage_plan_path=None):
    try:
        directory = pathlib.Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if not manifest["complete"] or manifest["format_version"] != FORMAT_VERSION:
            return False
        if manifest["observation_count"] != manifest["action_count"] + 1:
            return False
        expected = None if not stage_plan_path else hashlib.sha256(pathlib.Path(stage_plan_path).read_text(encoding="utf-8").encode()).hexdigest()
        if manifest["stage_plan_sha256"] != expected:
            return False
        required = {"states.npz", "observations.jsonl", "actions.jsonl", "policy_calls.jsonl",
                    "frame_map.json", "metadata.json", "scene.xml", "task.bddl"}
        if not required.issubset(manifest["files"]):
            return False
        if not all(hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
                   for name, digest in manifest["files"].items()):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
