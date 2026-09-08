# GuidedVLA: PyTorch implementation of π₀/π₀.₅ extended with object/skill/depth head specialization.
# Paper: "GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization" (RSS 2026)
# Based on openpi (https://github.com/Physical-Intelligence/openpi).
import logging
import math

import torch
from torch import Tensor
from torch import nn
from torch._inductor import list_mode_options
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.control_attention import ControlAwareAttention
from openpi.models_pytorch.control_attention import get_trainable_control_params
from openpi.models_pytorch.control_attention import inject_control_attention
from openpi.models_pytorch.depth.depth_attention import DepthTokenKVProjector
from openpi.models_pytorch.depth.model import DepthEncoder
from openpi.models_pytorch.gemma_pytorch import HeadSupervisionConfig
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.gemma_pytorch import SupervisedHeadStates
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


_BETA_DIST_CACHE: dict = {}


def sample_beta(alpha, beta, bsize, device):
    # Cache the Beta distribution (and its on-device param tensors) per
    # (alpha, beta, device). Constructing `torch.as_tensor(python_float, device='cuda')`
    # every forward step emits as_tensor -> _to_copy -> copy_ -> cudaStreamSynchronize;
    # caching removes that entirely on the hot path.
    key = (float(alpha), float(beta), device)
    dist = _BETA_DIST_CACHE.get(key)
    if dist is None:
        alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
        beta_t = torch.tensor(beta, dtype=torch.float32, device=device)
        dist = torch.distributions.Beta(alpha_t, beta_t)
        _BETA_DIST_CACHE[key] = dist
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.use_skill_loss = getattr(config, "use_skill_loss", False)
        self.skill_num_classes = getattr(config, "skill_num_classes", 8)
        object_head_indices = getattr(config, "object_head_indices", None)
        skill_head_indices = getattr(config, "skill_head_indices", None)
        guided_layer_indices = getattr(config, "guided_layer_indices", None)
        self.object_head_indices = tuple(object_head_indices or [])
        self.skill_head_indices = tuple(skill_head_indices or [])
        self.guided_layer_indices = tuple(guided_layer_indices or [])
        self.object_use_control = bool(getattr(config, "object_use_control", True))
        self.depth_use_control = bool(getattr(config, "depth_use_control", True))
        self.skill_use_control = bool(getattr(config, "skill_use_control", True))

        if not self.object_head_indices and hasattr(config, "num_object_distill_heads"):
            num_object_heads = int(getattr(config, "num_object_distill_heads", 0))
            self.object_head_indices = tuple(range(max(num_object_heads, 0)))
        if not self.skill_head_indices:
            self.skill_head_indices = ()

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )
        # Control whether depth attention modifications apply to the control branch.
        self.paligemma_with_expert._depth_use_control = self.depth_use_control  # noqa: SLF001

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # Work around an Inductor shape-padding bug that can crash compilation of
        # the official pi0_libero inference graph on mixed bf16/fp32 GEMMs.
        # sample_actions is compiled lazily on first call, so pass the config
        # patch through torch.compile options instead of a temporary config context.
        compile_options = dict(list_mode_options("max-autotune"))
        compile_options["shape_padding"] = False
        self.sample_actions = torch.compile(self.sample_actions, dynamic=False, options=compile_options)

        # Switch HuggingFace attention from eager O(n²) to SDPA, which dispatches to
        # FlashAttention2 (H200/sm90) or memory-efficient attention (4060/sm89) automatically.
        # This only affects the HuggingFace direct-call paths (prefix KV cache, no-ControlNet fallback).
        # The custom layer-by-layer loop in gemma_pytorch.py uses F.scaled_dot_product_attention
        # for non-distill heads and manual softmax(QK^T + mask) for distill heads — unaffected.
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "sdpa"  # noqa: SLF001
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "sdpa"  # noqa: SLF001

        # Initialize ControlAttention flag
        self.control_attention_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

        self.use_depth = False
        if config.use_depth:
            self.use_depth = True
            self.depth_module = DepthEncoder(
                depth_model_name=config.depth_model_name, feature_dim=1024, freeze_depth_model=True
            )
            self.depth_token_proj = DepthTokenKVProjector(
                hidden_size=1024,
                num_heads=8,
                head_dim=256,
                num_groups=len(self.guided_layer_indices),
                depth_head_indices=config.depth_head_indices,
            )
        elif self.guided_layer_indices:
            logging.info("guided_layer_indices is set but use_depth=False, so depth guidance is disabled.")

        num_patches = 256
        indices_list = []
        for i in range(3):  # Max Views
            start = i * num_patches
            indices_list.append(torch.arange(start, start + num_patches, dtype=torch.long))

        view_indices = torch.stack(indices_list)
        # Kept as a device anchor for the object-loss fallback; the index content
        # itself is no longer consumed (PR 2 drops the view-patch index_select in
        # favour of full-K weight scatter).
        self.register_buffer("view_patch_indices", view_indices, persistent=False)

        # Optional auxiliary classifier for skill-level supervision built on top
        # of a subset of attention heads. For torch.compile / CUDA graph
        # compatibility, we eagerly initialize this head here instead of doing
        # lazy initialization inside the forward pass.
        if self.use_skill_loss:
            per_head_dim = paligemma_config.head_dim
            self.skill_head: nn.Linear | None = nn.Linear(per_head_dim, self.skill_num_classes)
        else:
            self.skill_head = None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.paligemma_with_expert.set_gradient_checkpointing(enabled=True)
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.paligemma_with_expert.set_gradient_checkpointing(enabled=False)
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.paligemma_with_expert.use_gradient_checkpointing

    def get_guided_layers(self) -> list[int]:
        """Shared layers used by object, skill, and depth guidance."""
        return list(self.guided_layer_indices)

    def get_distill_layers(self) -> list[int]:
        """Backward-compatible alias for the shared guided layers."""
        return self.get_guided_layers()

    def compute_depth_key_values(self, images):
        if not self.use_depth:
            return None

        depth_features = self.depth_module(images[0])
        depth_kv = self.depth_token_proj(depth_features)
        if len(depth_kv) != len(self.guided_layer_indices):
            raise RuntimeError(
                "Depth projector output count does not match guided_layer_indices. "
                f"Expected {len(self.guided_layer_indices)}, got {len(depth_kv)}."
            )
        return depth_kv

    def resolve_supervision_heads(
        self,
        *,
        use_object_loss: bool,
        use_skill_loss: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        enable_object_supervision = use_object_loss and self.config.use_object_loss
        object_head_indices = self.object_head_indices if enable_object_supervision else ()
        skill_head_indices = self.skill_head_indices if use_skill_loss else ()
        return object_head_indices, skill_head_indices

    def build_head_supervision_config(
        self,
        *,
        object_head_indices: tuple[int, ...],
        skill_head_indices: tuple[int, ...],
        use_skill_loss: bool,
    ) -> tuple[HeadSupervisionConfig | None, bool]:
        supervision_layers = self.get_guided_layers()
        use_object_loss = len(object_head_indices) > 0
        use_skill_supervision = use_skill_loss
        if use_skill_supervision and len(skill_head_indices) == 0:
            raise RuntimeError("Skill loss requested but skill_head_indices is empty.")

        # Two separate head counts (decoupled in PR 1):
        #   - num_probs_export_heads: drives the manual-softmax probs export.
        #     Only needs to cover object_head_indices (the skill loss consumes
        #     attention OUTPUT via att_pre, not probs, so skill heads don't need
        #     to enter the expensive [B, H, T_q, K] fp32 export path).
        #   - num_export_heads: clamps the skill-feature extraction window.
        #     Only needs to cover skill_head_indices.
        max_obj_index = max(object_head_indices) if object_head_indices else -1
        num_probs_export_heads = max_obj_index + 1 if max_obj_index >= 0 and use_object_loss else 0

        max_skill_index = max(skill_head_indices) if skill_head_indices else -1
        num_export_heads = max_skill_index + 1 if max_skill_index >= 0 and use_skill_supervision else 0

        include_origin_states = (not self.object_use_control and len(object_head_indices) > 0) or (
            not self.skill_use_control and use_skill_supervision
        )

        if not supervision_layers or not (use_object_loss or use_skill_supervision):
            return None, use_object_loss
        if num_probs_export_heads == 0 and num_export_heads == 0:
            return None, use_object_loss

        head_supervision_config = HeadSupervisionConfig(
            supervised_layers=tuple(supervision_layers),
            num_export_heads=num_export_heads,
            num_probs_export_heads=num_probs_export_heads,
            include_skill_states=use_skill_supervision,
            skill_head_indices=tuple(skill_head_indices) if use_skill_supervision else (),
            include_origin_states=include_origin_states,
        )
        return head_supervision_config, use_object_loss

    def compute_skill_logits(
        self,
        *,
        all_supervised_states: list[tuple[int, SupervisedHeadStates]],
        action_query_start_index: int,
    ) -> Tensor | None:
        if self.skill_head is None:
            raise RuntimeError(
                "Skill loss requested, but the model was initialized without a skill head. "
                "Set config.use_skill_loss=True to enable skill-level supervision."
            )

        if not all_supervised_states:
            return None

        skill_feature_sum: Tensor | None = None
        num_skill_layers = 0
        for _layer_index, states in all_supervised_states:
            if not self.skill_use_control and states.origin_skill_features is not None:
                skill_attention_output = states.origin_skill_features
            else:
                skill_attention_output = states.skill_features

            action_queries = skill_attention_output[:, :, action_query_start_index:, :]
            pooled_features = action_queries.mean(dim=(1, 2))
            skill_feature_sum = pooled_features if skill_feature_sum is None else skill_feature_sum + pooled_features
            num_skill_layers += 1

        if skill_feature_sum is None or num_skill_layers == 0:
            return None

        skill_feature_tensor = skill_feature_sum / num_skill_layers
        if skill_feature_tensor.dtype != self.skill_head.weight.dtype:
            skill_feature_tensor = skill_feature_tensor.to(self.skill_head.weight.dtype)
        logits_input = skill_feature_tensor
        return self.skill_head(logits_input)

    def get_object_attention_probs(
        self,
        states: SupervisedHeadStates,
    ) -> Tensor:
        if self.object_use_control or states.origin_attention_probs is None:
            attention_probs = states.attention_probs
        else:
            attention_probs = states.origin_attention_probs

        if attention_probs is None:
            raise RuntimeError(
                "Object supervision requires cached attention probabilities, but they were not exported. "
                "This indicates an invalid object/depth supervision configuration."
            )
        return attention_probs

    def _object_head_index_tensor(self, object_head_indices: tuple[int, ...], device: torch.device) -> Tensor:
        cache = getattr(self, "_object_head_index_cache", None)
        if cache is None:
            cache = {}
            self._object_head_index_cache = cache
        key = (object_head_indices, device)
        idx = cache.get(key)
        if idx is None:
            idx = torch.tensor(object_head_indices, dtype=torch.long, device=device)
            cache[key] = idx
        return idx

    @staticmethod
    def _build_object_key_weights(
        target_maps: Tensor,
        target_masks: Tensor,
        num_total_keys: int,
    ) -> Tensor:
        """Scatter the per-view object pixel weights into a full-K fp32 tensor.

        The joint attention key axis is laid out as::

            [image patches (V x P) | language | suffix]

        where suffix = [state | action] for pi0, and [action] for pi0.5
        so the image keys occupy the leading ``V*P`` slots. Language, state,
        action, invalid views, and non-object image patches get weight 0.
        """
        batch_size, views, patches = target_maps.shape
        num_image_keys = views * patches
        if num_image_keys > num_total_keys:
            raise ValueError(
                f"target_maps has {num_image_keys} image-key slots but joint attention has "
                f"only {num_total_keys} keys — view_patch layout is inconsistent."
            )
        target_dtype = target_maps.dtype
        per_pixel_weight = target_maps * target_masks.to(target_dtype).unsqueeze(-1)  # [B, V, P]
        weights = torch.zeros(batch_size, num_total_keys, dtype=target_dtype, device=target_maps.device)
        weights[:, :num_image_keys] = per_pixel_weight.flatten(1)
        return weights

    def compute_object_mass_loss(
        self,
        attention_probs: Tensor,
        action_query_start_index: int,
        object_head_indices: tuple[int, ...],
        target_maps: Tensor,
        target_masks: Tensor,
        head_aggregation: str = "mean_heads",
    ) -> Tensor:
        """Object-mass supervision loss.

        In the default ``mean_heads`` mode, the selected object-head attention
        probabilities are averaged first, then object mass is computed as::

            object_mass[b, t] = sum_k mean_h(attn[b, h, t, k]) * w[b, k]
            loss[b, t]        = -log(object_mass.clamp_min(eps))

        where ``w`` is the full-K weight tensor (1.0 at valid object image-key
        positions, 0.0 everywhere else). Loss is averaged over valid
        (batch, action-step) pairs. This matches the paper formula and the
        reported-results training setup.

        In ``per_head`` mode, each object head is independently penalized and
        losses are averaged over valid (batch, action-step, object-head) triples.
        This is kept as a stricter diagnostic option.
        """
        idx = self._object_head_index_tensor(object_head_indices, attention_probs.device)
        head_slice = torch.index_select(attention_probs, 1, idx)
        action_slice = head_slice[:, :, action_query_start_index:, :]  # [B, H_obj, T_a, K]

        num_total_keys = attention_probs.shape[-1]
        object_key_weights = self._build_object_key_weights(target_maps, target_masks, num_total_keys)

        if head_aggregation == "per_head":
            object_mass = (action_slice * object_key_weights[:, None, None, :]).sum(dim=-1)  # [B, H_obj, T_a]
        elif head_aggregation == "mean_heads":
            averaged_action_slice = action_slice.mean(dim=1)  # [B, T_a, K]
            object_mass = (averaged_action_slice * object_key_weights[:, None, :]).sum(dim=-1)  # [B, T_a]
        else:
            raise ValueError(f"Unsupported object loss head aggregation: {head_aggregation!r}")

        loss_per_entry = -torch.log(object_mass.clamp_min(1e-6))

        per_batch_valid = object_key_weights.amax(dim=-1) > 0  # [B] bool
        valid_entry_mask = per_batch_valid.view(-1, *([1] * (loss_per_entry.ndim - 1))).expand_as(loss_per_entry)
        num_valid = valid_entry_mask.sum().clamp_min(1).to(loss_per_entry.dtype)
        return (loss_per_entry * valid_entry_mask.to(loss_per_entry.dtype)).sum() / num_valid

    def compute_object_loss(
        self,
        *,
        all_supervised_states: list[tuple[int, SupervisedHeadStates]],
        action_query_start_index: int,
        object_head_indices: tuple[int, ...],
        object_targets: dict[str, Tensor],
    ) -> Tensor:
        if "object_maps" in object_targets:
            device = object_targets["object_maps"].device
        elif object_targets:
            device = next(iter(object_targets.values())).device
        else:
            device = self.view_patch_indices.device

        if not all_supervised_states:
            return torch.zeros((), device=device, dtype=torch.float32)

        if "object_maps" not in object_targets or "object_masks" not in object_targets:
            return torch.zeros((), device=device, dtype=torch.float32)

        if not object_head_indices:
            return torch.zeros((), device=device, dtype=torch.float32)

        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        num_layers = 0
        head_aggregation = getattr(self.config, "object_loss_head_aggregation", "mean_heads")

        for _, states in all_supervised_states:
            cached_attention_probs = self.get_object_attention_probs(states)
            total_loss = total_loss + self.compute_object_mass_loss(
                cached_attention_probs,
                action_query_start_index,
                object_head_indices,
                object_targets["object_maps"],
                object_targets["object_masks"],
                head_aggregation=head_aggregation,
            )
            num_layers += 1

        if num_layers == 0:
            return torch.zeros((), device=device, dtype=torch.float32)
        return total_loss / num_layers

    def enable_control_attention(
        self,
        *,
        num_control_heads: int | None = None,
        copy_weights: bool | None = None,
        freeze_origin: bool | None = None,
        control_layer_indices: list[int] | None = None,
        use_headwise_gate: bool | None = None,
    ):
        """Enable ControlNet-style attention on the action expert.

        Fusion is always zero_conv (ControlNet design): output = origin + zero_conv(branch).
        Control attention is always injected into the action expert only.

        Args:
            num_control_heads: Number of attention heads for the control branch (default from config)
            copy_weights: Copy origin weights to control branch instead of random init
            freeze_origin: Freeze the original action-expert attention branch
            control_layer_indices: Explicit list of layer indices to inject (None = all)
            use_headwise_gate: Add per-head sigmoid gate to control branch Q projection
        """
        if self.control_attention_enabled:
            logging.warning("ControlAttention is already enabled!")
            return

        # Read from config if not specified
        if num_control_heads is None:
            num_control_heads = self.config.control_attention_num_heads
        if copy_weights is None:
            copy_weights = self.config.control_attention_copy_weights
        if freeze_origin is None:
            freeze_origin = self.config.control_attention_freeze_origin
        if use_headwise_gate is None:
            use_headwise_gate = self.config.control_attention_use_headwise_gate

        # Handle -1 as None (full copy mode)
        if num_control_heads == -1:
            num_control_heads = None

        # Decide which layers receive Control branches
        if control_layer_indices is not None:
            layer_indices = control_layer_indices
        else:
            layer_indices = self.get_distill_layers() if self.config.control_attention_distill_only else None

        if layer_indices is not None and (self.object_use_control or self.skill_use_control or self.depth_use_control):
            required_guided_layers = set(self.guided_layer_indices)
            missing_guided_layers = sorted(required_guided_layers - set(layer_indices))
            if missing_guided_layers:
                raise ValueError(
                    "ControlAttention must be injected into every guided layer when "
                    "object_use_control, skill_use_control, or depth_use_control is enabled. "
                    f"Missing guided layers: {missing_guided_layers}"
                )

        mode_str = "copy" if copy_weights or num_control_heads is None else f"{num_control_heads}heads"
        freeze_str = ", frozen_origin" if freeze_origin else ""
        logging.info(f"Enabling ControlAttention on action expert [{mode_str}, zero_conv{freeze_str}]...")

        replaced_count = inject_control_attention(
            self,
            num_control_heads=num_control_heads,
            copy_weights=copy_weights,
            freeze_origin=freeze_origin,
            layer_indices=layer_indices,
            use_headwise_gate=use_headwise_gate,
        )

        self.control_attention_enabled = True

        logging.info(f"✓ ControlAttention enabled: {replaced_count} expert layers [{mode_str}, zero_conv{freeze_str}]")

    def get_control_attention_params(self):
        """Get trainable ControlAttention parameters for optimizer.

        Returns:
            list: Trainable parameters from Control branches and zero convolutions
        """
        if not self.control_attention_enabled:
            logging.warning("ControlAttention is not enabled, returning empty list")
            return []

        return get_trainable_control_params(self)

    @staticmethod
    def _get_attention_input_dtype(attn_module) -> torch.dtype:
        if isinstance(attn_module, ControlAwareAttention):
            return attn_module.origin.q_proj.weight.dtype
        return attn_module.q_proj.weight.dtype

    def _align_prefix_embeddings_dtype(self, *embeddings):
        """Align prefix embeddings to the PaliGemma backbone dtype."""
        target_dtype = self._get_attention_input_dtype(
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn
        )
        return tuple(
            emb.to(dtype=target_dtype) if emb is not None and emb.dtype != target_dtype else emb for emb in embeddings
        )

    def _align_expert_embeddings_dtype(self, *embeddings):
        """Align suffix embeddings to the action expert dtype."""
        target_dtype = self._get_attention_input_dtype(
            self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn
        )
        return tuple(
            emb.to(dtype=target_dtype) if emb is not None and emb.dtype != target_dtype else emb for emb in embeddings
        )

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.is_gradient_checkpointing_enabled() and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks, *, dtype: torch.dtype):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        attention_mask_4d = torch.zeros(att_2d_masks_4d.shape, dtype=dtype, device=att_2d_masks.device)
        return attention_mask_4d.masked_fill_(~att_2d_masks_4d, -2.3819763e38)

    def build_joint_attention_inputs(
        self,
        *,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        attention_mask_4d = self._prepare_attention_masks_4d(att_2d_masks, dtype=dtype)
        if attention_mask_4d.device != device:
            attention_mask_4d = attention_mask_4d.to(dtype=dtype, device=device)
        return attention_mask_4d, position_ids

    def run_joint_backbone(
        self,
        *,
        prefix_embs: Tensor,
        suffix_embs: Tensor,
        attention_mask_4d: Tensor,
        position_ids: Tensor,
        adarms_cond: Tensor | None,
        depth_kv,
        head_supervision_config: HeadSupervisionConfig | None,
    ) -> tuple[Tensor, list[tuple[int, SupervisedHeadStates]]]:
        def forward_backbone(
            prefix_embs_: Tensor,
            suffix_embs_: Tensor,
            attention_mask_4d_: Tensor,
            position_ids_: Tensor,
            adarms_cond_: Tensor | None,
            head_supervision_config_: HeadSupervisionConfig | None,
        ):
            (_, suffix_out), _, all_supervised_states = self.paligemma_with_expert(
                attention_mask=attention_mask_4d_,
                position_ids=position_ids_,
                past_key_values=None,
                inputs_embeds=[prefix_embs_, suffix_embs_],
                use_cache=False,
                adarms_cond=[None, adarms_cond_],
                head_supervision_config=head_supervision_config_,
                guided_layer_indices=self.guided_layer_indices,
                depth_kv=depth_kv,
            )
            return suffix_out, all_supervised_states

        return forward_backbone(
            prefix_embs,
            suffix_embs,
            attention_mask_4d,
            position_ids,
            adarms_cond,
            head_supervision_config,
        )

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        return time_beta * 0.999 + 0.001

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_mask_parts = []  # list of bool tensors, one per token group

        # Process images — batch all views into a single SigLIP forward pass
        def image_embed_func(imgs):
            return self.paligemma_with_expert.embed_image(imgs)

        # Defensive contiguous() so embed_image's dynamic=False stride guards never fail.
        all_images = torch.cat(images, dim=0).contiguous(memory_format=torch.contiguous_format)
        all_img_emb = self._apply_checkpoint(image_embed_func, all_images)
        img_embs = all_img_emb.split([img.shape[0] for img in images], dim=0)

        for img_emb, img_mask in zip(img_embs, img_masks, strict=True):
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Build attention mask as a tensor — avoids torch.tensor(python_list) graph break
            att_mask_parts.append(torch.zeros(num_img_embs, dtype=torch.bool, device=img_emb.device))

        # Process language tokens (cheap op — no gradient checkpointing)
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_mask_parts.append(torch.zeros(num_lang_embs, dtype=torch.bool, device=lang_emb.device))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        # Concatenate token-group masks along sequence dim, then broadcast over batch
        att_masks_1d = torch.cat(att_mask_parts, dim=0)  # [total_prefix_len]
        bsize = pad_masks.shape[0]
        att_masks = att_masks_1d[None, :].expand(bsize, -1)  # [B, total_prefix_len]

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_mask_parts = []  # list of bool tensors — avoids torch.tensor(python_list) graph break

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32 and state.dtype != torch.float32:
                state = state.to(torch.float32)

            # Embed state (cheap Linear — no gradient checkpointing)
            state_emb = self.state_proj(state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # State attends to prefix; prefix does not attend to state/actions (mask=1 blocks)
            att_mask_parts.append(torch.ones(1, dtype=torch.bool, device=device))

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP (cheap ops — no grad checkpointing)
        action_emb = self.action_in_proj(noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)  # swish == silu
            action_time_emb = self.action_time_mlp_out(x)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)  # swish == silu
            x = self.time_mlp_out(x)
            time_emb = F.silu(x)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # First action token attends bidirectionally; remaining action tokens are causal (mask=0)
        device = timestep.device
        action_horizon = self.config.action_horizon
        att_mask_parts.append(torch.ones(1, dtype=torch.bool, device=device))
        if action_horizon > 1:
            att_mask_parts.append(torch.zeros(action_horizon - 1, dtype=torch.bool, device=device))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_1d = torch.cat(att_mask_parts, dim=0)  # [suffix_len]
        att_masks = att_masks_1d[None, :].expand(bsize, -1)  # [B, suffix_len]

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        observation,
        actions,
        *,
        object_targets: dict[str, Tensor] | None = None,
        use_object_loss: bool = False,
        use_skill_loss: bool = False,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        skill_soft_tensor = getattr(observation, "skill_soft", None)
        if use_skill_loss and skill_soft_tensor is None:
            raise RuntimeError(
                "Skill loss requested but observation.skill_soft is missing. "
                "Provide observation.skill_id so the data pipeline can construct soft labels."
            )
        if use_object_loss and object_targets is None:
            raise RuntimeError("Object loss requested but object_targets is missing.")

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        depth_kv = self.compute_depth_key_values(images)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        # Keep prefix on the PaliGemma dtype and suffix on the action expert dtype.
        (prefix_embs,) = self._align_prefix_embeddings_dtype(prefix_embs)
        (suffix_embs,) = self._align_expert_embeddings_dtype(suffix_embs)
        attention_mask_4d, position_ids = self.build_joint_attention_inputs(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            # Joint attention always mixes in the float32 action expert, so use a
            # float32 additive mask up front and avoid per-layer SDPA casts.
            dtype=torch.float32,
            device=prefix_embs.device,
        )

        object_head_indices, skill_head_indices = self.resolve_supervision_heads(
            use_object_loss=use_object_loss,
            use_skill_loss=use_skill_loss,
        )

        expert_query_start_idx = prefix_embs.shape[1]
        action_query_start_idx = expert_query_start_idx + 1 if not self.pi05 else expert_query_start_idx

        head_supervision_config, shouldcompute_object_loss = self.build_head_supervision_config(
            object_head_indices=object_head_indices,
            skill_head_indices=skill_head_indices,
            use_skill_loss=use_skill_loss,
        )

        suffix_out, all_supervised_states = self.run_joint_backbone(
            prefix_embs=prefix_embs,
            suffix_embs=suffix_embs,
            attention_mask_4d=attention_mask_4d,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
            depth_kv=depth_kv,
            head_supervision_config=head_supervision_config,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        if suffix_out.dtype != torch.float32:
            suffix_out = suffix_out.to(dtype=torch.float32)

        skill_logits_for_loss: Tensor | None = None
        skill_soft_for_loss: Tensor | None = None
        if use_skill_loss:
            if skill_soft_tensor.device != suffix_out.device or skill_soft_tensor.dtype != torch.float32:
                skill_soft_for_loss = skill_soft_tensor.to(device=suffix_out.device, dtype=torch.float32)
            else:
                skill_soft_for_loss = skill_soft_tensor
            skill_logits_for_loss = self.compute_skill_logits(
                all_supervised_states=all_supervised_states,
                action_query_start_index=action_query_start_idx,
            )

        v_t = self.action_out_proj(suffix_out)

        main_loss = F.mse_loss(u_t, v_t, reduction="mean")
        object_loss = torch.zeros((), device=main_loss.device, dtype=torch.float32)
        skill_loss = torch.zeros((), device=main_loss.device, dtype=torch.float32)

        if shouldcompute_object_loss and object_targets is not None:
            object_loss = self.compute_object_loss(
                all_supervised_states=all_supervised_states,
                action_query_start_index=action_query_start_idx,
                object_head_indices=object_head_indices,
                object_targets=object_targets,
            )

        if use_skill_loss and skill_logits_for_loss is not None and skill_soft_for_loss is not None:
            log_prob = F.log_softmax(skill_logits_for_loss, dim=-1, dtype=torch.float32)
            kl_loss = F.kl_div(log_prob, skill_soft_for_loss, reduction="batchmean", log_target=False)
            skill_loss = kl_loss

        return main_loss, object_loss, skill_loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        depth_kv = self.compute_depth_key_values(images)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Prefix cache runs through the PaliGemma backbone.
        (prefix_embs,) = self._align_prefix_embeddings_dtype(prefix_embs)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks, dtype=prefix_embs.dtype)

        _, past_key_values, _ = self.paligemma_with_expert(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Use a plain Python float for dt so torch.compile sees it as a constant.
        dt = -1.0 / num_steps

        x_t = noise
        # Replace the while-loop (data-dependent branch → graph break) with a
        # compile-friendly for-loop over a static Python range.  Time values are
        # computed as Python floats so they become compile-time constants.
        for step in range(num_steps):
            t = 1.0 + step * dt  # Python float: 1.0, (n-1)/n, ..., 1/n
            expanded_time = torch.full((bsize,), t, dtype=torch.float32, device=device)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                depth_kv=depth_kv,
            )
            x_t = x_t + dt * v_t
        return x_t

    def _build_attention_capture_config(self) -> HeadSupervisionConfig:
        """Build the inference-only configuration used to export attention maps.

        Attention probabilities are normally materialized only for auxiliary
        supervision during training.  For visualization we explicitly request
        all action-expert query heads in the four guided layers.  This method is
        used only by the eager attention-capture path, so the normal compiled
        inference path remains unchanged.
        """
        num_heads = int(self.paligemma_with_expert.gemma_expert.model.config.num_attention_heads)
        return HeadSupervisionConfig(
            supervised_layers=tuple(self.guided_layer_indices),
            num_export_heads=0,
            num_probs_export_heads=num_heads,
            include_skill_states=False,
            include_origin_states=self.control_attention_enabled,
        )

    @staticmethod
    def _compact_attention_maps(
        attention_probs: Tensor,
        *,
        action_query_start_index: int,
        view_patch_counts: list[int],
        head_indices: list[int] | tuple[int, ...] | None = None,
    ) -> Tensor:
        """Return image-only attention averaged over action queries.

        Output shape is ``[heads, views, patches]`` for a single-item batch.
        Keeping heads and views makes the saved file useful for both the
        paper-style object-head visualization and all-head baselines, while
        avoiding the much larger full ``[heads, queries, keys]`` tensor.
        """
        if attention_probs.ndim != 4 or attention_probs.shape[0] != 1:
            raise ValueError(
                "Attention capture currently expects batch size 1, got "
                f"{tuple(attention_probs.shape)}."
            )
        if action_query_start_index >= attention_probs.shape[2]:
            raise ValueError(
                "Attention query slice is empty: "
                f"start={action_query_start_index}, query_length={attention_probs.shape[2]}. "
                "The query axis contains suffix/action queries only; do not use the prefix length as an offset."
            )
        if head_indices is not None:
            attention_probs = attention_probs[:, list(head_indices)]

        num_image_keys = sum(view_patch_counts)
        image_probs = attention_probs[:, :, action_query_start_index:, :num_image_keys]
        image_probs = image_probs.mean(dim=2)[0]  # [heads, image_keys]

        views = []
        offset = 0
        for patch_count in view_patch_counts:
            views.append(image_probs[:, offset : offset + patch_count])
            offset += patch_count
        return torch.stack(views, dim=1).detach().to(device="cpu", dtype=torch.float16)

    def _extract_attention_snapshot(
        self,
        all_supervised_states: list[tuple[int, SupervisedHeadStates]],
        *,
        action_query_start_index: int,
        view_patch_counts: list[int],
        timestep: float,
    ) -> dict:
        """Convert per-layer attention states into a compact serializable snapshot."""
        origin_maps: dict[str, Tensor] = {}
        object_maps: dict[str, Tensor] = {}

        for layer_index, states in all_supervised_states:
            origin_probs = states.origin_attention_probs
            if origin_probs is None:
                origin_probs = states.attention_probs
            if origin_probs is not None:
                origin_maps[str(layer_index)] = self._compact_attention_maps(
                    origin_probs,
                    action_query_start_index=action_query_start_index,
                    view_patch_counts=view_patch_counts,
                )

            # pi0_libero_object uses the control/object branch.  The full
            # object+depth+skill config uses origin attention for object loss;
            # mirror that configuration here so the saved "object" map is the
            # same branch that receives object supervision.
            if self.config.use_object_loss and states.attention_probs is not None:
                object_probs = states.attention_probs if self.object_use_control else origin_probs
                if object_probs is not None:
                    object_maps[str(layer_index)] = self._compact_attention_maps(
                        object_probs,
                        action_query_start_index=action_query_start_index,
                        view_patch_counts=view_patch_counts,
                        head_indices=self.object_head_indices,
                    )

        snapshot = {
            "timestep": float(timestep),
            "layer_indices": [int(layer_index) for layer_index, _ in all_supervised_states],
            "view_patch_counts": [int(count) for count in view_patch_counts],
            "origin": origin_maps,
        }
        if object_maps:
            snapshot["object"] = object_maps
        return snapshot

    @torch.no_grad()
    def sample_actions_with_attention(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        attention_interval=1,
    ) -> tuple[Tensor, dict]:
        """Run eager inference and return actions plus compact attention snapshots.

        ``attention_interval`` is measured in denoising steps.  For example,
        with ``num_steps=10`` and ``attention_interval=2``, maps are saved at
        denoising loop steps 0, 2, 4, 6, and 8.  This method intentionally
        bypasses the compiled action-only function because attention capture
        materializes per-layer probability tensors.
        """
        attention_interval = int(attention_interval)
        if attention_interval <= 0:
            raise ValueError(f"attention_interval must be positive, got {attention_interval}")

        bsize = observation.state.shape[0]
        if bsize != 1:
            raise ValueError("Attention export currently supports batch size 1 only.")
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        # ``images`` are still BCHW tensors here, so ``image.shape[1]`` is
        # the RGB channel count (3), not the number of SigLIP image tokens.
        # Attention keys are laid out as 256 patches per view.  Use the same
        # registered layout that is used by the object-head supervision code.
        num_patches_per_view = int(self.view_patch_indices.shape[1])
        view_patch_counts = [num_patches_per_view for _ in images]
        depth_kv = self.compute_depth_key_values(images)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        (prefix_embs,) = self._align_prefix_embeddings_dtype(prefix_embs)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks, dtype=prefix_embs.dtype)

        _, past_key_values, _ = self.paligemma_with_expert(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        snapshots = []
        capture_config = self._build_attention_capture_config()
        for step in range(num_steps):
            t = 1.0 + step * dt
            expanded_time = torch.full((bsize,), t, dtype=torch.float32, device=device)
            step_result = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                depth_kv=depth_kv,
                attention_capture_config=capture_config if step % attention_interval == 0 else None,
                view_patch_counts=view_patch_counts,
            )
            if isinstance(step_result, tuple):
                v_t, snapshot = step_result
            else:
                v_t, snapshot = step_result, None
            if snapshot is not None:
                snapshots.append(snapshot)
            x_t = x_t + dt * v_t

        return x_t, {
            "format_version": 1,
            "model_type": "pi05" if self.pi05 else "pi0",
            "layer_indices": list(self.guided_layer_indices),
            "view_names": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
            "num_patches_per_view": 256,
            "snapshots": snapshots,
        }

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        depth_kv=None,
        attention_capture_config: HeadSupervisionConfig | None = None,
        view_patch_counts: list[int] | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # Expert-only denoising inference directly calls gemma_expert.model.
        (suffix_embs,) = self._align_expert_embeddings_dtype(suffix_embs)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        # Denoising uses expert-side queries, which run in float32 by design.
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks, dtype=suffix_embs.dtype)

        # Call paligemma_with_expert - always returns 3 values: outputs_embeds, past_key_values, all_supervised_states
        outputs_embeds, _, all_supervised_states = self.paligemma_with_expert(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            guided_layer_indices=self.guided_layer_indices,
            depth_kv=depth_kv,
            head_supervision_config=attention_capture_config,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        action_output = self.action_out_proj(suffix_out)
        if attention_capture_config is None:
            return action_output

        if view_patch_counts is None:
            raise ValueError("view_patch_counts is required when attention capture is enabled")
        # ``all_supervised_states[*].attention_probs`` has query axis
        # [state, action...] for pi0 or [action...] for pi05.  It does not
        # include the prefix image/language queries, so prefix_pad_masks.shape[1]
        # must not be used as an offset here.
        action_query_start_idx = 1 if not self.pi05 else 0
        snapshot = self._extract_attention_snapshot(
            all_supervised_states,
            action_query_start_index=action_query_start_idx,
            view_patch_counts=view_patch_counts,
            timestep=float(timestep[0].item()),
        )
        return action_output, snapshot

    @torch.no_grad()
    def compute_skill_logits_for_infer(
        self,
        observation,
        device,
        *,
        actions: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        deterministic: bool = True,
    ) -> Tensor:
        """Compute skill logits for inference.

        This reuses the same attention outputs as the training-time skill loss
        but does not require object targets or ground-truth skill ids.

        Args:
            observation: `Observation` object (already on correct device).
            device: Torch device used for computation.

        Returns:
            Tensor of shape [B, skill_num_classes] with unnormalized logits.
        """
        if self.skill_head is None:
            raise RuntimeError(
                "Skill logits requested for inference, but the model was initialized without a skill head. "
                "Set config.use_skill_loss=True to enable skill logits."
            )

        # Preprocess observation (no train-time augmentations).
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        bsize = state.shape[0]

        # Prepare actions for suffix tokens. If not provided, fall back to zeros
        # (maintains the same token layout as training).
        if actions is None:
            actions = torch.zeros(
                bsize,
                self.config.action_horizon,
                self.config.action_dim,
                device=device,
                dtype=torch.float32,
            )
        else:
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)
            actions = actions.to(device=device, dtype=torch.float32)
            if actions.shape[0] != bsize:
                raise ValueError(f"actions batch size ({actions.shape[0]}) does not match observation ({bsize}).")
            if actions.shape[1:] != (self.config.action_horizon, self.config.action_dim):
                raise ValueError(
                    "actions shape mismatch: expected "
                    f"[B, {self.config.action_horizon}, {self.config.action_dim}], got {tuple(actions.shape)}."
                )

        # Deterministic inference avoids random noise/time which can make skill
        # predictions unstable at eval time.
        if time is None:
            if deterministic:
                infer_time = float(self.config.skill_infer_time)
                time = torch.full((bsize,), infer_time, device=device, dtype=torch.float32)
            else:
                time = self.sample_time(actions.shape[0], device)
        else:
            time = torch.as_tensor(time, device=device, dtype=torch.float32)
            if time.dim() == 0:
                time = time.expand(bsize)
            if time.dim() != 1 or time.shape[0] != bsize:
                raise ValueError(f"time shape mismatch: expected [B], got {tuple(time.shape)}.")

        if noise is None:
            noise = torch.zeros_like(actions) if deterministic else self.sample_noise(actions.shape, device)
        else:
            noise = noise.to(device=device, dtype=torch.float32)
            if noise.shape != actions.shape:
                raise ValueError(f"noise shape mismatch: expected {tuple(actions.shape)}, got {tuple(noise.shape)}.")

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions

        # Build prefix (image + language) and suffix (state + action/time) tokens.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        depth_kv = self.compute_depth_key_values(images)
        (prefix_embs,) = self._align_prefix_embeddings_dtype(prefix_embs)
        (suffix_embs,) = self._align_expert_embeddings_dtype(suffix_embs)
        attention_mask_4d, position_ids = self.build_joint_attention_inputs(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            # Skill-logit inference shares the same joint expert attention path.
            dtype=torch.float32,
            device=prefix_embs.device,
        )

        expert_query_start_idx = prefix_embs.shape[1]
        action_query_start_idx = expert_query_start_idx + 1 if not self.pi05 else expert_query_start_idx

        head_supervision_config, _ = self.build_head_supervision_config(
            object_head_indices=(),
            skill_head_indices=self.skill_head_indices,
            use_skill_loss=True,
        )

        _, all_supervised_states = self.run_joint_backbone(
            prefix_embs=prefix_embs,
            suffix_embs=suffix_embs,
            attention_mask_4d=attention_mask_4d,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
            depth_kv=depth_kv,
            head_supervision_config=head_supervision_config,
        )

        if not all_supervised_states:
            return torch.zeros(
                bsize,
                self.skill_num_classes,
                device=device,
                dtype=self.skill_head.weight.dtype,
            )

        skill_logits = self.compute_skill_logits(
            all_supervised_states=all_supervised_states,
            action_query_start_index=action_query_start_idx,
        )
        if skill_logits is None:
            return torch.zeros(
                bsize,
                self.skill_num_classes,
                device=device,
                dtype=self.skill_head.weight.dtype,
            )

        return skill_logits
