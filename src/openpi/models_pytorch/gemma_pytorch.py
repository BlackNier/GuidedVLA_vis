# GuidedVLA: extended Gemma/PaliGemma forward pass supporting dual-path ControlNet attention,
# supervised head state export (object / skill), depth cross-attention, and skill head feature extraction.
# Paper: "GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization" (RSS 2026)
# Based on openpi (https://github.com/Physical-Intelligence/openpi).
import dataclasses
import logging
from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from openpi.models_pytorch.attention.attn_paths import attn_path_guided
from openpi.models_pytorch.attention.attn_paths import repeat_kv
from openpi.models_pytorch.attention.attn_paths import ugather
from openpi.models_pytorch.control_attention import ControlAwareAttention
from openpi.models_pytorch.depth.depth_attention import DepthHeadConfig
from openpi.models_pytorch.depth.depth_attention import prepare_guided_attention_config


@dataclasses.dataclass(frozen=True)
class HeadSupervisionConfig:
    """Configuration for auxiliary per-head supervision (H_object and H_skill).

    Passed from PI0Pytorch into the backbone forward to control which layers and
    how many heads export attention states for loss computation.

    Two independent head counts:
      - ``num_export_heads`` clamps the skill-feature extraction window (cheap:
        projected attention output, not probs).
      - ``num_probs_export_heads`` gates the manual-softmax probs export path
        (expensive: full ``[B, H, T_q, K]`` fp32 tensor saved for backward).
        Setting it to 0 means no probs are materialized for this layer.
    """

    supervised_layers: tuple[int, ...]  # transformer layer indices that export attention states
    num_export_heads: int  # clamp for skill-feature extraction (max(skill_head_indices) + 1)
    num_probs_export_heads: int = 0  # leading heads populated in attention_probs (max(object_head_indices) + 1)
    include_skill_states: bool = False  # export H_skill features for L_skill
    skill_head_indices: tuple[int, ...] = ()  # H_skill head indices
    include_origin_states: bool = False  # also export origin-branch states (when supervision uses main branch)


@dataclasses.dataclass
class SupervisedHeadStates:
    """Per-layer attention states exported from supervised heads (H_object / H_skill).

    Collected per-layer and passed back to PI0Pytorch for L_object and L_skill computation.
    """

    attention_probs: torch.Tensor | None = None  # attention weights — used for L_object
    skill_features: torch.Tensor | None = None  # H_skill features — used for L_skill
    origin_attention_probs: torch.Tensor | None = None  # origin-branch attention weights
    origin_skill_features: torch.Tensor | None = None  # origin-branch H_skill features


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        # Initialize gradient checkpointing settings
        self.use_gradient_checkpointing = False
        logging.info("PaliGemmaWithExpertModel: gradient checkpointing disabled by default")
        # Control whether depth attention modifications apply to control branch.
        self._depth_use_control = True

        self.to_bfloat16_for_selected_params(precision)

    def set_gradient_checkpointing(self, *, enabled: bool):
        """Update gradient checkpointing setting and log the change."""
        old_value = self.use_gradient_checkpointing

        self.use_gradient_checkpointing = enabled

        if old_value != self.use_gradient_checkpointing:
            logging.info(
                f"PaliGemmaWithExpertModel: gradient checkpointing {'enabled' if self.use_gradient_checkpointing else 'disabled'}"
            )

    def apply_depth_expert_initialization(self, guided_layer_indices: list[int], depth_head_indices: list[int]):
        print(f"Applying Zero-Initialization to Output Projection (O_proj) for heads {depth_head_indices}...")

        for layer_idx in guided_layer_indices:
            layer = self.gemma_expert.model.layers[layer_idx]

            o_proj = layer.self_attn.o_proj
            head_dim = layer.self_attn.head_dim

            with torch.no_grad():
                for head_idx in depth_head_indices:
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim
                    o_proj.weight[:, start_idx:end_idx].zero_()

            print(f"  -> Layer {layer_idx}: Heads {depth_head_indices} silenced.")

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    @staticmethod
    def _select_skill_heads(
        skill_attn: torch.Tensor,
        skill_head_indices: list[int] | tuple[int, ...] | None,
    ) -> torch.Tensor:
        """Select skill heads by explicit indices."""
        if skill_head_indices is not None:
            if len(skill_head_indices) == 0:
                return skill_attn[:, :0, :, :]
            return ugather(skill_attn, list(skill_head_indices))
        return skill_attn

    @staticmethod
    def _expand_branch_query_heads(
        q_branch: torch.Tensor,
        origin_num_heads: int,
        q_expand_linear: nn.Module | None,
    ) -> torch.Tensor:
        """Expand control branch query heads to match origin heads."""
        branch_num_heads = q_branch.shape[1]
        if q_expand_linear is not None:
            batch_size, _, seq_len, head_dim = q_branch.shape
            q_branch_flat = q_branch.transpose(1, 2).reshape(batch_size * seq_len, -1)
            q_branch_exp_flat = q_expand_linear(q_branch_flat)
            return q_branch_exp_flat.reshape(batch_size, seq_len, origin_num_heads, head_dim).transpose(1, 2)
        if branch_num_heads < origin_num_heads:
            n_rep = origin_num_heads // branch_num_heads
            return q_branch.repeat_interleave(n_rep, dim=1)
        return q_branch

    @staticmethod
    def _resolve_supervision_heads(num_heads: int | None, max_heads: int) -> int:
        """Normalize the number of supervised heads; negative values mean no supervision."""
        if num_heads is None:
            return max_heads
        if num_heads < 0:
            return 0
        return min(num_heads, max_heads)

    @staticmethod
    def _extract_guided_attention_args(
        guided_attention_config: DepthHeadConfig | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple[int, ...]]:
        if guided_attention_config is None:
            return None, None, ()
        return (
            guided_attention_config.depth_token_k,
            guided_attention_config.depth_token_v,
            guided_attention_config.depth_head_indices,
        )

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def get_rotary_cos_sin(self, seq_len: int, head_dim: int, position_ids: torch.LongTensor, device, dtype):
        dummy_tensor = torch.zeros(
            position_ids.shape[0],
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        return self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)

    def dispatch_attention(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dt_k,
        dt_v,
        depth_head_indices,
        num_probs_heads,
    ):
        """Unified attention path for guided and non-guided layers.

        ``num_probs_heads`` is the number of leading heads that will populate
        ``attention_probs`` via the manual-softmax path. -1 (or 0) disables the
        manual path entirely and routes all heads through fused SDPA.
        """
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        dropout_p = 0.0
        distill_heads = self._resolve_supervision_heads(num_probs_heads, query.shape[1])
        return attn_path_guided(
            query,
            key_states,
            value_states,
            attention_mask,
            scaling,
            dropout_p,
            distill_heads=distill_heads,
            training=module.training,
            depth_token_k=dt_k,
            depth_token_v=dt_v,
            depth_head_indices=depth_head_indices,
        )

    # ------------------------------------------------------------------
    # Unified layer computation — replaces the four former
    # _compute_layer_standard / _compute_layer_dual_path_train /
    # _compute_layer_dual_path_infer / _compute_layer_expert_only_infer.
    # ------------------------------------------------------------------

    def compute_layer_complete(
        self,
        pali_layer,
        expert_layer,
        inputs_embeds,
        attention_mask,
        position_ids,
        layer_past_key_values,
        adarms_cond,
        num_supervision_heads: int,
        num_probs_heads: int,
        dt_k,
        dt_v,
        depth_head_indices,
        expert_uses_control_attn,
        rotary_cos_sin,
        head_supervision_config: HeadSupervisionConfig | None,
    ):
        """
        Unified layer computation for all execution modes.

        Replaces the former _compute_layer_standard / _compute_layer_dual_path_train /
        _compute_layer_dual_path_infer / _compute_layer_expert_only_infer.

        Mode selection (all based on None-checks and bool flags, compile-safe):
          has_pali_input=True,  expert_uses_control_attn=False → standard training
          has_pali_input=True,  expert_uses_control_attn=True  → ControlNet training
          has_pali_input=False, expert_uses_control_attn=True  → ControlNet inference
          has_pali_input=False, expert_uses_control_attn=False → expert-only inference

        Precision notes:
          - PaliGemma layers may be bfloat16 (weights + activations under autocast).
          - The action expert is always float32.
          - Explicit dtype-cast guards are preserved for all cross-precision projections.
        """
        include_skill_states = head_supervision_config.include_skill_states if head_supervision_config else False
        skill_head_indices = head_supervision_config.skill_head_indices if head_supervision_config else ()
        include_origin_states = head_supervision_config.include_origin_states if head_supervision_config else False

        has_pali_input = inputs_embeds[0] is not None
        is_inference = layer_past_key_values is not None

        # ------------------------------------------------------------------ #
        # Phase 1 — Pali Q/K/V (training) or cached K/V (inference)          #
        # ------------------------------------------------------------------ #
        # ControlAwareAttention is ONLY injected into the action expert.
        # pali_layer.self_attn is always a standard attention module.
        pali_attn = pali_layer.self_attn

        if has_pali_input:
            pali_input = inputs_embeds[0]
            pali_normed, pali_gate = pali_layer.input_layernorm(pali_input, cond=adarms_cond[0])
            input_shape_pali = pali_normed.shape[:-1]
            head_dim = pali_attn.head_dim
            q_pali = pali_attn.q_proj(pali_normed).view(*input_shape_pali, -1, head_dim).transpose(1, 2)
            k_pali = pali_attn.k_proj(pali_normed).view(*input_shape_pali, -1, head_dim).transpose(1, 2)
            v_pali = pali_attn.v_proj(pali_normed).view(*input_shape_pali, -1, head_dim).transpose(1, 2)
            pali_seq_len = q_pali.shape[2]
        else:
            pali_input = None
            pali_gate = None
            pali_seq_len = 0
            head_dim = pali_attn.head_dim
            if is_inference:
                k_pali, v_pali = layer_past_key_values
            else:
                raise ValueError("inputs_embeds[0] is None but no past_key_values provided")

        # ------------------------------------------------------------------ #
        # Phase 2 — Expert Q/K/V                                              #
        # ------------------------------------------------------------------ #
        expert_input = inputs_embeds[1]
        expert_normed, expert_gate = expert_layer.input_layernorm(expert_input, cond=adarms_cond[1])

        if expert_uses_control_attn:
            control_attn = expert_layer.self_attn
            if expert_normed.dtype != control_attn.origin.q_proj.weight.dtype:
                expert_normed = expert_normed.to(control_attn.origin.q_proj.weight.dtype)
            (q_origin, k_origin, v_origin), (q_branch, k_branch, v_branch), q_branch_gate = (
                control_attn.compute_dual_path_qkv(expert_normed)
            )
            origin_num_heads = q_origin.shape[1]
            expert_seq_len = q_origin.shape[2]
            q_branch_exp = self._expand_branch_query_heads(
                q_branch, origin_num_heads, control_attn.get_q_expand_linear()
            )
        else:
            attn = expert_layer.self_attn
            if expert_normed.dtype != attn.q_proj.weight.dtype:
                expert_normed = expert_normed.to(attn.q_proj.weight.dtype)
            input_shape = expert_normed.shape[:-1]
            q_origin = attn.q_proj(expert_normed).view(*input_shape, -1, head_dim).transpose(1, 2)
            k_origin = attn.k_proj(expert_normed).view(*input_shape, -1, head_dim).transpose(1, 2)
            v_origin = attn.v_proj(expert_normed).view(*input_shape, -1, head_dim).transpose(1, 2)
            origin_num_heads = q_origin.shape[1]
            expert_seq_len = q_origin.shape[2]
            q_branch_exp = None
            q_branch_gate = None

        cos, sin = rotary_cos_sin

        # ------------------------------------------------------------------ #
        # Phase 3 — Rotary embeddings + build joint Q/K/V                     #
        # ------------------------------------------------------------------ #
        if has_pali_input and expert_uses_control_attn:
            # ControlNet training: split rotary per segment so branch gets its own apply.
            # Origin result is mathematically identical to joint rotary, but we need
            # the branch to have its own apply_rotary call with the expert-segment cos/sin.
            cos_pali = cos[:, :pali_seq_len, :]
            sin_pali = sin[:, :pali_seq_len, :]
            cos_exp = cos[:, pali_seq_len:, :]
            sin_exp = sin[:, pali_seq_len:, :]
            q_pali_rot, k_pali_rot = modeling_gemma.apply_rotary_pos_emb(
                q_pali, k_pali, cos_pali, sin_pali, unsqueeze_dim=1
            )
            q_origin_rot, k_origin_rot = modeling_gemma.apply_rotary_pos_emb(
                q_origin, k_origin, cos_exp, sin_exp, unsqueeze_dim=1
            )
            q_branch_rot, k_branch_rot = modeling_gemma.apply_rotary_pos_emb(
                q_branch_exp, k_branch, cos_exp, sin_exp, unsqueeze_dim=1
            )
            q_origin_joint = torch.cat([q_pali_rot, q_origin_rot], dim=2)
            k_origin_joint = torch.cat([k_pali_rot, k_origin_rot], dim=2)
            v_origin_joint = torch.cat([v_pali, v_origin], dim=2)
            k_branch_joint = torch.cat([k_pali_rot, k_branch_rot], dim=2)
            v_branch_joint = torch.cat([v_pali, v_branch], dim=2)
            # Branch fast-only: when neither probs export NOR skill features are
            # needed, only compute expert-segment Q for the branch (expert output
            # is unaffected; saves pali-length attention compute). Both supervision
            # channels are consulted to preserve legacy behaviour after PR 1
            # decoupled the two head-count plumbing values.
            fast_path = num_supervision_heads <= 0 and num_probs_heads <= 0
            if fast_path:
                q_branch_joint = q_branch_rot
                branch_attention_mask = attention_mask[:, :, pali_seq_len:, :] if attention_mask is not None else None
            else:
                q_branch_joint = torch.cat([q_pali_rot, q_branch_rot], dim=2)
                branch_attention_mask = attention_mask

        elif has_pali_input:
            # Standard training (no ControlNet): concat then apply rotary jointly.
            q_joint = torch.cat([q_pali, q_origin], dim=2)
            k_joint = torch.cat([k_pali, k_origin], dim=2)
            q_joint, k_joint = modeling_gemma.apply_rotary_pos_emb(q_joint, k_joint, cos, sin, unsqueeze_dim=1)
            q_origin_joint = q_joint
            k_origin_joint = k_joint
            v_origin_joint = torch.cat([v_pali, v_origin], dim=2)
            q_branch_joint = None
            k_branch_joint = None
            v_branch_joint = None
            fast_path = num_supervision_heads <= 0 and num_probs_heads <= 0
            branch_attention_mask = None

        else:
            # Inference: expert-only Q, pali K/V already rotary-encoded in KV cache.
            q_origin_rot, k_origin_rot = modeling_gemma.apply_rotary_pos_emb(
                q_origin, k_origin, cos, sin, unsqueeze_dim=1
            )
            q_origin_joint = q_origin_rot
            k_origin_joint = torch.cat([k_pali, k_origin_rot], dim=2)
            v_origin_joint = torch.cat([v_pali, v_origin], dim=2)
            fast_path = num_supervision_heads <= 0 and num_probs_heads <= 0
            if expert_uses_control_attn:
                q_branch_rot, k_branch_rot = modeling_gemma.apply_rotary_pos_emb(
                    q_branch_exp, k_branch, cos, sin, unsqueeze_dim=1
                )
                q_branch_joint = q_branch_rot
                k_branch_joint = torch.cat([k_pali, k_branch_rot], dim=2)
                v_branch_joint = torch.cat([v_pali, v_branch], dim=2)
            else:
                q_branch_joint = None
                k_branch_joint = None
                v_branch_joint = None
            branch_attention_mask = attention_mask

        batch_size = q_origin_joint.shape[0]
        scaling = pali_attn.scaling

        # ------------------------------------------------------------------ #
        # Phase 4 — Attention dispatch                                         #
        # ------------------------------------------------------------------ #
        depth_use_control = self._depth_use_control
        if expert_uses_control_attn and depth_use_control:
            dt_k_origin, dt_v_origin, depth_head_indices_origin = None, None, ()
            dt_k_branch, dt_v_branch, depth_head_indices_branch = dt_k, dt_v, depth_head_indices
        else:
            dt_k_origin, dt_v_origin, depth_head_indices_origin = dt_k, dt_v, depth_head_indices
            dt_k_branch, dt_v_branch, depth_head_indices_branch = None, None, ()

        # Origin: collect distill (probs) states only when requested for origin.
        # num_probs_heads drives the manual-softmax probs export; skill-feature
        # extraction is independent and runs off att_pre later in Phase 5.
        if expert_uses_control_attn:
            origin_distill_heads = num_probs_heads if include_origin_states else -1
            # In ControlNet inference fast path, skip all distill collection
            if fast_path and is_inference:
                origin_distill_heads = -1
        else:
            origin_distill_heads = -1 if fast_path else num_probs_heads

        att_output_origin, attn_probs_origin = self.dispatch_attention(
            pali_attn,
            q_origin_joint,
            k_origin_joint,
            v_origin_joint,
            attention_mask,
            scaling,
            dt_k_origin,
            dt_v_origin,
            depth_head_indices_origin,
            origin_distill_heads,
        )
        att_output_origin_pre = att_output_origin

        # Branch attention (ControlNet only)
        att_output_branch_pre = None
        attn_probs_branch = None
        if expert_uses_control_attn:
            if has_pali_input:
                # Training: fast_path means branch Q is expert-only → -1 heads
                branch_distill_heads = -1 if fast_path else num_probs_heads
            else:
                # Inference: skip distill when fast_path
                branch_distill_heads = -1 if fast_path else num_probs_heads
            att_output_branch, attn_probs_branch = self.dispatch_attention(
                pali_attn,
                q_branch_joint,
                k_branch_joint,
                v_branch_joint,
                branch_attention_mask,
                scaling,
                dt_k_branch,
                dt_v_branch,
                depth_head_indices_branch,
                branch_distill_heads,
            )
            att_output_branch_pre = att_output_branch

        # ------------------------------------------------------------------ #
        # Phase 5 — Collect distill states                                     #
        # ------------------------------------------------------------------ #
        # Phase 5 — Collect supervised head states (H_object / H_skill)       #
        # Run when EITHER probs (object) OR skill features are wanted for this layer.
        supervised_states: SupervisedHeadStates | None = None
        if num_supervision_heads != -1 or num_probs_heads > 0:
            if expert_uses_control_attn:
                att_pre = att_output_branch_pre
                probs_for_sup = attn_probs_branch
            else:
                att_pre = att_output_origin_pre
                probs_for_sup = attn_probs_origin

            # Skill features are clamped by num_supervision_heads (now decoupled
            # from the probs export count). When skill is disabled, this is -1.
            num_heads_clamp = None if num_supervision_heads < 0 else num_supervision_heads

            skill_feats: torch.Tensor | None = None
            if include_skill_states and num_supervision_heads > 0:
                skill_attn = att_pre.transpose(1, 2)
                if num_heads_clamp is not None:
                    skill_attn = skill_attn[:, :num_heads_clamp]
                skill_feats = self._select_skill_heads(skill_attn, skill_head_indices)

            origin_probs: torch.Tensor | None = None
            origin_skill_feats: torch.Tensor | None = None
            if include_origin_states and expert_uses_control_attn:
                if attn_probs_origin is not None:
                    origin_probs = attn_probs_origin
                if include_skill_states and num_supervision_heads > 0:
                    skill_orig = att_output_origin_pre.transpose(1, 2)
                    if num_heads_clamp is not None:
                        skill_orig = skill_orig[:, :num_heads_clamp]
                    origin_skill_feats = self._select_skill_heads(skill_orig, skill_head_indices)

            supervised_states = SupervisedHeadStates(
                attention_probs=probs_for_sup,
                skill_features=skill_feats,
                origin_attention_probs=origin_probs,
                origin_skill_features=origin_skill_feats,
            )

        # ------------------------------------------------------------------ #
        # Phase 6 — Output projection + MLP + residuals                       #
        # ------------------------------------------------------------------ #
        outputs_embeds = []

        if has_pali_input and not expert_uses_control_attn:
            # Standard training: split joint attention output by sequence position.
            att_output_origin = att_output_origin_pre.reshape(batch_size, -1, origin_num_heads * head_dim)
            start_pos = 0
            for i, (hidden_states, layer) in enumerate(zip(inputs_embeds, [pali_layer, expert_layer], strict=True)):
                gate_i = pali_gate if i == 0 else expert_gate
                end_pos = start_pos + hidden_states.shape[1]
                seg = att_output_origin[:, start_pos:end_pos]
                attn_layer = layer.self_attn
                if seg.dtype != attn_layer.o_proj.weight.dtype:
                    seg = seg.to(attn_layer.o_proj.weight.dtype)
                out_emb = attn_layer.o_proj(seg)
                out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gate_i)  # noqa: SLF001
                after_residual = out_emb
                out_emb, post_gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)
                out_emb = layer.mlp(out_emb)
                out_emb = modeling_gemma._gated_residual(after_residual, out_emb, post_gate)  # noqa: SLF001
                outputs_embeds.append(out_emb)
                start_pos = end_pos
        else:
            # ControlNet training or expert-only/ControlNet inference.
            att_output_origin = att_output_origin_pre.reshape(batch_size, -1, origin_num_heads * head_dim)

            if has_pali_input:
                # ControlNet training: pali segment needs its own o_proj + MLP.
                pali_seg = att_output_origin[:, :pali_seq_len]
                if pali_seg.dtype != pali_attn.o_proj.weight.dtype:
                    pali_seg = pali_seg.to(pali_attn.o_proj.weight.dtype)
                pali_out = pali_attn.o_proj(pali_seg)
                pali_out = modeling_gemma._gated_residual(pali_input, pali_out, pali_gate)  # noqa: SLF001
                pali_after_residual = pali_out
                pali_out, pali_post_gate = pali_layer.post_attention_layernorm(pali_out, cond=adarms_cond[0])
                if pali_layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    pali_out = pali_out.to(dtype=torch.bfloat16)
                pali_out = pali_layer.mlp(pali_out)
                pali_out = modeling_gemma._gated_residual(pali_after_residual, pali_out, pali_post_gate)  # noqa: SLF001
                outputs_embeds.append(pali_out)
            else:
                outputs_embeds.append(None)

            # Expert output.
            if expert_uses_control_attn:
                control_attn = expert_layer.self_attn
                # In ControlNet training: origin output has [pali, expert] segments.
                # In ControlNet inference: origin output is expert-only.
                exp_seg_origin = att_output_origin if not has_pali_input else att_output_origin[:, pali_seq_len:]
                exp_seg_origin = exp_seg_origin.reshape(batch_size, -1, origin_num_heads * head_dim)

                # Branch output: in training fast_path, Q was expert-only so output is already
                # expert-only; otherwise slice out the expert segment.
                att_out_branch = att_output_branch_pre.reshape(batch_size, -1, origin_num_heads * head_dim)
                branch_is_expert_only = not has_pali_input or fast_path
                exp_seg_branch = att_out_branch if branch_is_expert_only else att_out_branch[:, pali_seq_len:]

                # Apply head-wise sigmoid gate on the branch expert segment.
                if q_branch_gate is not None:
                    exp_seg_branch = exp_seg_branch.reshape(batch_size, expert_seq_len, origin_num_heads, head_dim)
                    gate_sigmoid = torch.sigmoid(q_branch_gate.transpose(1, 2))  # [B, exp_seq, H, 1]
                    exp_seg_branch = (exp_seg_branch * gate_sigmoid).reshape(
                        batch_size, expert_seq_len, origin_num_heads * head_dim
                    )

                origin_o_proj = control_attn.origin.o_proj
                branch_o_proj = control_attn.object_branch.o_proj
                if exp_seg_origin.dtype != origin_o_proj.weight.dtype:
                    exp_seg_origin = exp_seg_origin.to(origin_o_proj.weight.dtype)
                if exp_seg_branch.dtype != branch_o_proj.weight.dtype:
                    exp_seg_branch = exp_seg_branch.to(branch_o_proj.weight.dtype)
                expert_out = control_attn.fuse_outputs(origin_o_proj(exp_seg_origin), branch_o_proj(exp_seg_branch))
            else:
                # Expert-only inference without ControlNet.
                attn = expert_layer.self_attn
                num_heads_out = q_origin_joint.shape[1]
                att_out = att_output_origin_pre.reshape(batch_size, -1, num_heads_out * head_dim)
                if att_out.dtype != attn.o_proj.weight.dtype:
                    att_out = att_out.to(attn.o_proj.weight.dtype)
                expert_out = attn.o_proj(att_out)

            expert_out = modeling_gemma._gated_residual(expert_input, expert_out, expert_gate)  # noqa: SLF001
            expert_after_residual = expert_out
            expert_out, expert_post_gate = expert_layer.post_attention_layernorm(expert_out, cond=adarms_cond[1])
            if expert_layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                expert_out = expert_out.to(dtype=torch.bfloat16)
            expert_out = expert_layer.mlp(expert_out)
            expert_out = modeling_gemma._gated_residual(  # noqa: SLF001
                expert_after_residual, expert_out, expert_post_gate
            )
            outputs_embeds.append(expert_out)

        return outputs_embeds, supervised_states

    def layer_forward(
        self,
        hidden_states_arg,
        att_mask_arg,
        pos_ids_arg,
        layer_past_kv_arg,
        adarms_cond_arg,
        pali_layer_module,
        expert_layer_module,
        num_heads_arg,
        num_probs_heads_arg,
        guided_attention_config,
        head_supervision_config,
        expert_flag_arg,
        rotary_cs_arg,
    ):
        """Thin wrapper around compute_layer_complete for gradient checkpointing compatibility.

        ``num_heads_arg`` is the skill-feature clamp (legacy ``num_supervision_heads``).
        ``num_probs_heads_arg`` is the manual-softmax probs export count.
        """
        dt_k, dt_v, depth_head_indices = self._extract_guided_attention_args(guided_attention_config)
        return self.compute_layer_complete(
            pali_layer_module,
            expert_layer_module,
            hidden_states_arg,
            att_mask_arg,
            pos_ids_arg,
            layer_past_kv_arg,
            adarms_cond_arg,
            num_heads_arg,
            num_probs_heads_arg,
            dt_k,
            dt_v,
            depth_head_indices,
            expert_flag_arg,
            rotary_cs_arg,
            head_supervision_config,
        )

    def forward(
        self,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        head_supervision_config: HeadSupervisionConfig | None = None,
        guided_layer_indices: list[int] | None = None,
        depth_kv: list[DepthHeadConfig] | None = None,
    ):
        """
        Args:
            head_supervision_config: If provided, specifies which layers and how many heads
                to export attention states for H_object (L_object) and H_skill (L_skill) supervision.
        """
        collect_supervised_states = head_supervision_config is not None
        # Keep a stable output structure for torch.compile by always returning a list.
        all_supervised_states: list[tuple[int, SupervisedHeadStates]] = []

        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
            return [prefix_output, suffix_output], prefix_past_key_values, all_supervised_states
        if inputs_embeds[0] is None:
            # Use per-layer path when ControlNet is active or when depth attention is needed.
            expert_layer = self.gemma_expert.model.layers[0]
            expert_uses_control_attn = isinstance(expert_layer.self_attn, ControlAwareAttention)
            if expert_uses_control_attn or depth_kv is not None or head_supervision_config is not None:
                # Continue to layer-by-layer processing below with [None, suffix] inputs.
                # Attention visualization also needs this path for raw pi0,
                # whose action expert has no ControlAwareAttention wrapper.
                inputs_embeds = [None, inputs_embeds[1]]
            else:
                # No ControlNet or depth attention: can directly call gemma_expert.model.
                suffix_output = self.gemma_expert.model(
                    inputs_embeds=inputs_embeds[1],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[1],
                )
                suffix_output = suffix_output.last_hidden_state
                prefix_output = None
                prefix_past_key_values = None
                return [prefix_output, suffix_output], prefix_past_key_values, all_supervised_states

        models = [self.paligemma.language_model, self.gemma_expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        guided_attention_configs: list[DepthHeadConfig | None] = [
            prepare_guided_attention_config(_li, guided_layer_indices or [], depth_kv) if depth_kv is not None else None
            for _li in range(num_layers)
        ]

        supervision_disabled = -1

        if head_supervision_config is not None:
            _collect_layers_set = set(head_supervision_config.supervised_layers)
            _skill_val = head_supervision_config.num_export_heads
            _probs_val = head_supervision_config.num_probs_export_heads
            _target_skill = _skill_val if _skill_val > 0 else supervision_disabled
            _target_probs = _probs_val if _probs_val > 0 else supervision_disabled
        else:
            _collect_layers_set = set()
            _target_skill = supervision_disabled
            _target_probs = supervision_disabled

        # Per-layer skill-feature clamp (legacy ``num_supervision_heads`` plumbing).
        per_layer_num_heads = [
            (_target_skill if _li in _collect_layers_set else supervision_disabled) for _li in range(num_layers)
        ]
        # Per-layer manual-softmax probs export count (PR 1 decoupling).
        per_layer_num_probs_heads = [
            (_target_probs if _li in _collect_layers_set else supervision_disabled) for _li in range(num_layers)
        ]

        expert_uses_control_attn_list: list[bool] = [
            isinstance(models[1].layers[_li].self_attn, ControlAwareAttention) for _li in range(num_layers)
        ]

        # Precompute rotary cos/sin once per forward to reduce per-layer overhead.
        total_seq_len = sum(emb.shape[1] for emb in inputs_embeds if emb is not None)
        sample_emb = next(emb for emb in inputs_embeds if emb is not None)
        head_dim = self.paligemma.config.text_config.head_dim
        rotary_cos_sin = self.get_rotary_cos_sin(
            total_seq_len, head_dim, position_ids, device=sample_emb.device, dtype=sample_emb.dtype
        )

        compile_active = hasattr(torch, "compiler") and torch.compiler.is_compiling()
        use_checkpoint = self.training and self.use_gradient_checkpointing and not compile_active

        # Process all layers with gradient checkpointing if enabled
        for layer_idx, (pali_layer, expert_layer) in enumerate(
            zip(self.paligemma.language_model.layers, self.gemma_expert.model.layers, strict=True)
        ):
            curr_num_heads = per_layer_num_heads[layer_idx]
            curr_num_probs_heads = per_layer_num_probs_heads[layer_idx]
            current_guided_attention_config = guided_attention_configs[layer_idx]
            curr_expert_flag = expert_uses_control_attn_list[layer_idx]
            layer_past_kv = past_key_values[layer_idx] if past_key_values is not None else None

            layer_args = (
                inputs_embeds,
                attention_mask,
                position_ids,
                layer_past_kv,
                adarms_cond,
                pali_layer,
                expert_layer,
                curr_num_heads,
                curr_num_probs_heads,
                current_guided_attention_config,
                head_supervision_config,
                curr_expert_flag,
                rotary_cos_sin,
            )

            if use_checkpoint:
                inputs_embeds, supervised_states = torch.utils.checkpoint.checkpoint(
                    self.layer_forward,
                    *layer_args,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                inputs_embeds, supervised_states = self.layer_forward(*layer_args)

            if collect_supervised_states and supervised_states is not None:
                all_supervised_states.append((layer_idx, supervised_states))

        # final norm
        # Define final norm computation function for gradient checkpointing
        def compute_final_norms(inputs_embeds, adarms_cond):
            return [
                models[i].norm(hidden_states, cond=adarms_cond[i])[0] if hidden_states is not None else None
                for i, hidden_states in enumerate(inputs_embeds)
            ]

        # Apply gradient checkpointing to final norm if enabled
        if use_checkpoint:
            outputs_embeds = torch.utils.checkpoint.checkpoint(
                compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
            )
        else:
            outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

        prefix_output = outputs_embeds[0]
        suffix_output = outputs_embeds[1]
        prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values, all_supervised_states
