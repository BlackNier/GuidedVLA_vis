import os

import torch
import torch.nn.functional as functional


def _attention_debug_enabled() -> bool:
    # Read this at call time so setting the variable after module import still
    # works while debugging a long-lived server process.
    return os.environ.get("OPENPI_ATTENTION_DEBUG", "0") == "1"


if _attention_debug_enabled():
    print(f"attention debug module loaded: {__file__}", flush=True)


class UniqueHeadGather(torch.autograd.Function):
    """Gather along the head dimension (dim=1) for a list of unique indices.

    Standard fancy-index reads like ``tensor[:, index_list]`` compile to
    ``aten::index_select`` in the forward, whose autograd backward emits
    ``aten::index_put_`` with ``accumulate=True``.  CUDAGraphs refuses to
    capture graphs that contain ``index_put_`` with accumulate, so any
    ``torch.compile`` invocation with the CUDAGraphs backend skips graph
    capture for the entire attention kernel.

    Because our head-partition indices are always *unique* (constructed from
    disjoint range comprehensions), the backward only ever writes each output
    slot once — accumulation is unnecessary.  This custom Function replaces
    the accumulating ``index_put_`` in the backward with ``index_copy_``
    (non-accumulating scatter-copy), which CUDAGraphs captures without issue.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
        idx = torch.tensor(indices, dtype=torch.long, device=tensor.device)
        ctx.save_for_backward(idx)
        ctx.n_heads = tensor.shape[1]
        return torch.index_select(tensor, 1, idx)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (idx,) = ctx.saved_tensors
        grad_input = torch.zeros(
            grad_output.shape[0],
            ctx.n_heads,
            *grad_output.shape[2:],
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        grad_input.index_copy_(1, idx, grad_output)
        return grad_input, None


def ugather(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Gather along dim=1 with unique indices, CUDAGraphs-compatible backward."""
    return UniqueHeadGather.apply(tensor, indices)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def cast_attention_inputs(
    query: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.dtype != key_states.dtype or query.dtype != value_states.dtype:
        attn_dtype = torch.promote_types(query.dtype, key_states.dtype)
        attn_dtype = torch.promote_types(attn_dtype, value_states.dtype)
        query = query.to(attn_dtype)
        key_states = key_states.to(attn_dtype)
        value_states = value_states.to(attn_dtype)
    return query, key_states, value_states


def cast_attention_mask(attention_mask: torch.Tensor | None, target_dtype: torch.dtype) -> torch.Tensor | None:
    if attention_mask is None or attention_mask.dtype in (torch.bool, target_dtype):
        return attention_mask
    return attention_mask.to(target_dtype)


def run_sdpa(
    query: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout_p: float,
) -> torch.Tensor:
    query, key_states, value_states = cast_attention_inputs(query, key_states, value_states)
    attention_mask = cast_attention_mask(attention_mask, query.dtype)
    return functional.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        scale=scaling,
    )


def run_depth_sdpa(
    query: torch.Tensor,
    depth_token_k: torch.Tensor,
    depth_token_v: torch.Tensor,
    dropout_p: float,
    scaling: float,
) -> torch.Tensor:
    """Depth heads attend only to depth tokens (no normal KV)."""
    query, depth_token_k, depth_token_v = cast_attention_inputs(query, depth_token_k, depth_token_v)
    return functional.scaled_dot_product_attention(
        query,
        depth_token_k,
        depth_token_v,
        attn_mask=None,
        dropout_p=dropout_p,
        scale=scaling,
    )


def attn_path_guided(
    query: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout_p: float,
    *,
    distill_heads: int = 0,
    training: bool = False,
    depth_token_k: torch.Tensor | None = None,
    depth_token_v: torch.Tensor | None = None,
    depth_head_indices: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Route attention heads to three paths:
      - distill heads (not depth): manual softmax to export attention probs for supervision
      - standard heads (not depth, not distill): fused SDPA
      - depth heads: SDPA against depth-token KV only (no normal KV); never supervised

    Depth heads must not overlap with [0, distill_heads) — they are never supervised.
    All head partitioning uses Python-level list comprehensions so torch.compile sees
    static integer lists — no dynamic-shape tensor indexing.

    Args:
        distill_heads: Number of leading heads that export attention probs (0 = none).
        depth_head_indices: Tuple of head indices that attend depth tokens instead of normal KV.
    """
    debug = _attention_debug_enabled()
    if debug:
        print(
            "attention debug path entered:",
            "query=",
            tuple(query.shape),
            "distill_heads=",
            distill_heads,
            flush=True,
        )

    if (depth_token_k is None) != (depth_token_v is None):
        raise ValueError("depth_token_k and depth_token_v must either both be set or both be None")

    batch_size, num_heads, query_length, _ = query.shape
    if distill_heads < 0 or distill_heads > num_heads:
        raise ValueError(f"distill_heads must be in [0, {num_heads}], got {distill_heads}")

    # ------------------------------------------------------------------ #
    # Compute head partitions as Python lists (compile-time constants).   #
    # With dynamic=False, num_heads is shape-specialised; distill_heads   #
    # and depth_head_indices are already Python ints/tuples.              #
    # ------------------------------------------------------------------ #
    depth_set = frozenset(depth_head_indices) if depth_token_k is not None else frozenset()

    # Leading `distill_heads` heads are supervised standard heads (depth heads excluded by contract)
    distill_standard: list[int] = [i for i in range(min(distill_heads, num_heads)) if i not in depth_set]
    # Remaining heads split into: normal SDPA vs depth SDPA
    standard: list[int] = [i for i in range(distill_heads, num_heads) if i not in depth_set]
    depth_heads: list[int] = sorted(depth_set)

    # Distill heads are contiguous leading heads [0, distill_heads) when no depth
    # head overlaps the distill range. In that case softmax's output already has
    # the final attention_probs shape, so we skip the zeros prealloc + scatter.
    distill_contiguous = distill_heads > 0 and len(distill_standard) == distill_heads

    attn_output = torch.zeros_like(query)
    attention_probs: torch.Tensor | None = None

    # Allocate supervised attention prob buffer (only if any distill heads exist
    # AND we can't populate it via direct assignment in the softmax branch below).
    if distill_heads > 0 and not distill_contiguous:
        attention_probs = torch.zeros(
            batch_size,
            distill_heads,
            query_length,
            key_states.shape[2],
            dtype=torch.float32,
            device=query.device,
        )

    # --- Supervised standard heads: manual softmax to export attention probs ---
    if distill_standard:
        q_d = ugather(query, distill_standard)
        k_d = ugather(key_states, distill_standard)
        v_d = ugather(value_states, distill_standard)
        q_d, k_d, v_d = cast_attention_inputs(q_d, k_d, v_d)

        # The eager attention-capture path must compute logits in float32.
        # q/k can be float16 or bfloat16 during inference; doing the matmul
        # in that dtype can overflow before softmax and produce an all-NaN
        # attention map, even though the final softmax is requested in fp32.
        q_d_float = torch.nan_to_num(q_d.float(), nan=0.0, posinf=0.0, neginf=0.0)
        k_d_float = torch.nan_to_num(k_d.float(), nan=0.0, posinf=0.0, neginf=0.0)
        scores = torch.matmul(q_d_float, k_d_float.transpose(-2, -1)) * float(scaling)
        if debug:
            print(
                "attention debug:",
                "q_d=",
                torch.isfinite(q_d).all().item(),
                "k_d=",
                torch.isfinite(k_d).all().item(),
                "scores=",
                torch.isfinite(scores).all().item(),
                flush=True,
            )
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        if attention_mask is not None:
            mask = attention_mask[..., : scores.shape[-1]]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores.masked_fill(mask < 0, torch.finfo(scores.dtype).min)
        probs = functional.softmax(scores, dim=-1, dtype=torch.float32)
        if debug:
            print("attention debug:", "probs=", torch.isfinite(probs).all().item(), flush=True)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).eps)
        if distill_contiguous:
            attention_probs = probs
        else:
            attention_probs[:, distill_standard] = probs
        probs_dropped = functional.dropout(probs, p=dropout_p, training=training)
        attn_output[:, distill_standard] = torch.matmul(probs_dropped.to(v_d.dtype), v_d)

    # --- Standard heads: fused SDPA ---
    if standard:
        std_out = run_sdpa(
            ugather(query, standard),
            ugather(key_states, standard),
            ugather(value_states, standard),
            attention_mask[:, :, :, : key_states.shape[2]] if attention_mask is not None else None,
            scaling,
            dropout_p,
        )
        attn_output[:, standard] = std_out

    # --- Depth heads: fused SDPA against depth tokens only ---
    if depth_heads:
        query_depth = ugather(query, depth_heads)
        depth_key = ugather(depth_token_k, depth_heads)
        depth_value = ugather(depth_token_v, depth_heads)

        depth_out = run_depth_sdpa(
            query_depth,
            depth_key,
            depth_value,
            dropout_p,
            scaling,
        )
        attn_output[:, depth_heads] = depth_out

    return attn_output.transpose(1, 2).contiguous(), attention_probs


__all__ = ["attn_path_guided", "repeat_kv"]
