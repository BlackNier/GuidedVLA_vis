"""Read-only LIBERO-plus / robosuite state capture, aligned to pre-action images.

Verified source: LIBERO-plus 4976dc30028e805ff8094b55501d532c48fec182,
libero/libero/envs/{env_wrapper,bddl_base_domain,object_states/base_object_states}.py.
No predicate turn_on() calls: those may modify visual state.
"""
import json
import os
from pathlib import Path
import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(plain(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


class StageStateRecorder:
    def __init__(self, wrapped, instruction, task_id, episode_index):
        self.env = wrapped.env
        self.sim = self.env.sim
        self.rows = []
        model = self.sim.model
        states = self.env.object_states_dict
        objects = dict(self.env.objects_dict, **self.env.fixtures_dict)
        # Goal regions may live on a table, not in obj_of_interest.
        goal = self.env.parsed_problem['goal_state']
        def strings(x):
            if isinstance(x, str):
                return [x]
            return [s for v in x for s in strings(v)] if isinstance(x, (list, tuple)) else []
        wanted = set(self.env.obj_of_interest) | (set(strings(goal)) & set(states))
        for name in list(wanted):
            parent = getattr(states.get(name), 'parent_name', None)
            if parent:
                wanted.add(parent)
        self.names = sorted(wanted & set(objects))
        self.sites = sorted(n for n in wanted if getattr(states.get(n), 'object_state_type', '') == 'site')
        gripper = self.env.robots[0].gripper
        geoms = gripper.important_geoms
        self.finger_ids = {}
        for side in ('left', 'right'):
            names = geoms.get(side + '_fingerpad') or geoms.get(side + '_finger')
            if not names:
                raise RuntimeError('Missing gripper collision group: ' + side)
            self.finger_ids[side] = {model.geom_name2id(n) for n in names}
        self.geom_owner = {}
        info = {}
        for name in self.names:
            obj = objects[name]
            for geom in obj.contact_geoms:
                self.geom_owner[model.geom_name2id(geom)] = name
            joints = {}
            for joint in (obj.joints or []):
                jid = model.joint_name2id(joint)
                # Hinge=3, slide=2. Exclude free/ball object pose joints.
                if int(model.jnt_type[jid]) in (2, 3):
                    joints[joint] = {'type': int(model.jnt_type[jid]),
                                     'range': plain(model.jnt_range[jid]),
                                     'qpos_addr': int(model.jnt_qposadr[jid])}
            info[name] = {'category': obj.category_name, 'joints': joints,
                          'articulation': plain(getattr(obj, 'object_properties', {}).get('articulation', {}))}
        site_info = {}
        for name in self.sites:
            sid = model.site_name2id(name)
            site_info[name] = {'parent': states[name].parent_name,
                               'joints': list(self.env.object_sites_dict[name].joints or []),
                               'size': plain(model.site_size[sid])}
        self.meta = {'schema_version': 1, 'instruction': instruction,
                     'benchmark_task_id': task_id, 'episode_index': episode_index,
                     'frame_convention': '0-based pre-action; wait steps excluded; terminal state has no image',
                     'units': {'position': 'metres', 'hinge': 'radians', 'body_quat': 'wxyz', 'eef_quat': 'xyzw'},
                     'objects': info, 'sites': site_info,
                     'goal_state': plain(goal), 'regions': plain(self.env.parsed_problem['regions'])}

    def capture(self, obs, frame_id, env_step, has_image=True):
        d = self.sim.data
        contacts = {name: {'left': False, 'right': False} for name in self.names}
        for i in range(d.ncon):
            c = d.contact[i]
            for finger, other in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                owner = self.geom_owner.get(other)
                if owner:
                    for side, ids in self.finger_ids.items():
                        if finger in ids and c.dist <= 0:
                            contacts[owner][side] = True
        objects = {}
        for name in self.names:
            bid = self.env.obj_body_id[name]
            objects[name] = {'pos': plain(d.body_xpos[bid]), 'quat_wxyz': plain(d.body_xquat[bid]),
                             'contact': contacts[name],
                             'joints': {n: float(d.qpos[v['qpos_addr']]) for n, v in self.meta['objects'][name]['joints'].items()}}
        sites = {n: {'pos': plain(d.get_site_xpos(n)), 'rotation': plain(d.get_site_xmat(n))} for n in self.sites}
        self.rows.append({'frame_id': int(frame_id), 'env_step': int(env_step), 'sim_time': float(d.time),
                          'has_image': has_image, 'eef_pos': plain(obs['robot0_eef_pos']),
                          'eef_quat_xyzw': plain(obs['robot0_eef_quat']),
                          'gripper_qpos': plain(obs['robot0_gripper_qpos']), 'objects': objects, 'sites': sites})

    def write(self, image_dir, image_indices, error=None):
        path = Path(image_dir) / 'stage_state'
        path.mkdir(parents=True, exist_ok=True)
        tmp = path / 'states.jsonl.tmp'
        with tmp.open('w') as f:
            for row in self.rows:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path / 'states.jsonl'))
        # Manifest last: incomplete / errored exports never appear complete.
        atomic_json(path / 'manifest.json', dict(self.meta, image_frames=list(image_indices),
                    row_count=len(self.rows), complete=error is None, error=error))
