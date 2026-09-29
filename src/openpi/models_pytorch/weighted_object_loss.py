"""Separate log-mass penalties for stage objects (PyTorch training only)."""
import torch


def weighted_object_mass_loss(attention_probs, action_start, heads, object_maps,
                              object_weights, object_valid, head_aggregation='mean_heads'):
    """maps[B,V,K,P], weights/valid[B,V,K]; normalize weights per valid view.

    Image keys are [view0 patches, view1 patches, ...], as in legacy object loss.
    Loss averages action steps/heads, then valid views per sample, then valid samples.
    Objects are NOT merged, normalized by area, or softmax-normalized as masks.
    """
    b, v, k, p = object_maps.shape
    if object_weights.shape != (b, v, k) or object_valid.shape != (b, v, k):
        raise ValueError('Per-object weight/valid shape mismatch')
    if attention_probs.shape[0] != b or attention_probs.shape[-1] < v * p:
        raise ValueError('Object maps and attention key layout mismatch')
    if action_start >= attention_probs.shape[-2]:
        raise ValueError('No action query tokens')
    selected = attention_probs[:, list(heads), action_start:, :v*p].float()
    if head_aggregation == 'mean_heads':
        selected = selected.mean(dim=1, keepdim=True)
    elif head_aggregation != 'per_head':
        raise ValueError(f'Unsupported head aggregation: {head_aggregation}')
    selected = selected.reshape(b, selected.shape[1], selected.shape[2], v, p)
    maps = object_maps.float()
    valid = object_valid.bool() & (maps.sum(-1) > 0) & (object_weights > 0)
    weights = torch.where(valid, object_weights.float(), 0.0)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    mass = torch.einsum('bhtvp,bvkp->bhtvk', selected, maps)
    per_view = (-mass.clamp_min(1e-6).log() * weights[:, None, None]).sum(-1).mean(dim=(1, 2))
    valid_views = valid.any(-1)
    per_sample = (per_view * valid_views).sum(-1) / valid_views.sum(-1).clamp_min(1)
    valid_samples = valid_views.any(-1)
    # All-invalid batch has a graph-connected zero; action/skill losses still train.
    return (per_sample * valid_samples).sum() / valid_samples.sum().clamp_min(1)
