from collections.abc import Sequence
from contextlib import nullcontext as _nullcontext
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._sample_kwargs.get("attention_capture", False) and not is_pytorch:
            raise ValueError("attention_capture is currently supported only for PyTorch policies.")

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            capture_attention = bool((self._sample_kwargs or {}).get("attention_capture", False))
            if capture_attention:
                if not hasattr(model, "sample_actions_with_attention"):
                    raise ValueError("attention_capture requires a PyTorch model with sample_actions_with_attention().")
                # Attention export intentionally uses the eager path.  The
                # regular action-only path remains torch.compile-optimized.
                self._sample_actions = model.sample_actions_with_attention
            else:
                self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device.
            # Use np.ascontiguousarray to avoid unnecessary copies when input is already numpy.
            inputs = jax.tree.map(
                lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(self._pytorch_device)[None, ...], inputs
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions.
        sample_kwargs = dict(self._sample_kwargs)
        # This flag selects the eager capture method in __init__; it is not a
        # model argument itself.
        sample_kwargs.pop("attention_capture", None)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension.
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim).
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        _inference_ctx = torch.inference_mode() if self._is_pytorch_model else _nullcontext()
        # When torch.compile uses CUDAGraphs (mode="reduce-overhead"), tensor output buffers
        # from a previous run can be overwritten by the next run. Calling
        # cudagraph_mark_step_begin() before each invocation tells CUDAGraphs that a new
        # independent step is starting, preventing stale-buffer errors.
        if (
            self._is_pytorch_model
            and hasattr(torch, "compiler")
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        ):
            torch.compiler.cudagraph_mark_step_begin()
        with _inference_ctx:
            model_output = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        attention_output = None
        if isinstance(model_output, tuple) and len(model_output) == 2 and isinstance(model_output[1], dict):
            actions_output, attention_output = model_output
        else:
            actions_output = model_output

        outputs = {
            "state": inputs["state"],
            "actions": actions_output,
        }

        model_time = time.monotonic() - start_time

        # Convert outputs to numpy.  Attention is intentionally kept outside
        # the dataset output transform because LiberoOutputs returns actions
        # only.  The compact attention payload is serialized by the rollout
        # client into attention.pt.
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            if attention_output is not None:
                attention_output = _to_numpy_tree(attention_output)
                if os.environ.get("OPENPI_ATTENTION_DEBUG", "0") == "1":
                    snapshots = attention_output.get("snapshots", [])
                    if snapshots:
                        origin = snapshots[0].get("origin", {})
                        for layer in ("9", "10", "11", "12"):
                            if layer in origin:
                                value = np.asarray(origin[layer])
                                print(
                                    "policy attention numpy:",
                                    layer,
                                    "finite=",
                                    np.isfinite(value).all(),
                                    "shape=",
                                    value.shape,
                                    "dtype=",
                                    value.dtype,
                                    flush=True,
                                )
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)

        if attention_output is not None:
            outputs["attention"] = attention_output

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _to_numpy_tree(value: Any) -> Any:
    """Convert nested torch tensors in an attention payload to NumPy values."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_numpy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy_tree(item) for item in value)
    return value


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
