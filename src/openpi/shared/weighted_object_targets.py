"""Dataset keys and batch validation for per-object LIBERO supervision."""
import torch

DATA_KEYS = {f'{prefix}_object_{suffix}': f'{prefix}_object_{suffix}'
             for prefix in ('agentview', 'wrist') for suffix in ('masks', 'weights', 'valid')}
TARGET_KEYS = ('per_object_maps', 'per_object_weights', 'per_object_valid')


def pack_targets(data, allowed_views=None):
    allowed = set(allowed_views or ('base_0', 'left_wrist_0', 'right_wrist_0'))
    maps_list, weights_list, valid_list = [], [], []
    shape = None
    for prefix, view in [('agentview', 'base_0'), ('wrist', 'left_wrist_0')]:
        maps = torch.as_tensor(data[f'{prefix}_object_masks'], dtype=torch.float32)
        weights = torch.as_tensor(data[f'{prefix}_object_weights'], device=maps.device, dtype=torch.float32)
        valid = torch.as_tensor(data[f'{prefix}_object_valid'], device=maps.device, dtype=torch.bool)
        if maps.ndim != 3 or maps.shape[-1] != 256 or weights.shape != maps.shape[:2] or valid.shape != weights.shape:
            raise ValueError('Expected per-view maps[B,K,256] and weights/valid[B,K]')
        if shape is not None and shape != maps.shape:
            raise ValueError('Both views must use the same padded object capacity')
        shape = maps.shape
        if not torch.isfinite(maps).all() or not ((maps == 0) | (maps == 1)).all():
            raise ValueError('Per-object masks must be finite binary patch maps')
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Per-object weights must be finite and nonnegative')
        valid = valid & maps.any(-1) & (weights > 0)
        if view not in allowed:
            valid = torch.zeros_like(valid)
        maps_list.append(maps); weights_list.append(weights); valid_list.append(valid)
    maps_list.append(torch.zeros_like(maps_list[0]))
    weights_list.append(torch.zeros_like(weights_list[0]))
    valid_list.append(torch.zeros_like(valid_list[0]))
    maps = torch.stack(maps_list, 1)
    weights = torch.stack(weights_list, 1)
    valid = torch.stack(valid_list, 1)
    return {'per_object_maps': maps, 'per_object_weights': weights, 'per_object_valid': valid,
            # Retained for existing target transport and diagnostic compatibility.
            'object_maps': (maps * valid[..., None]).amax(2), 'object_masks': valid.any(2)}
