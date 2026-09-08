import dataclasses
import enum
import logging
import socket

import torch
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None

    port: int = 8000
    record: bool = False

    # Optional eager attention export for LIBERO rollouts.  The client saves
    # one attention/<frame>.pt file for each exported image frame.
    attention_dump: bool = False
    attention_dump_interval: int = 1

    max_cuda_mem_fraction: float = 0.9
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if args.attention_dump and args.attention_dump_interval <= 0:
        raise ValueError("attention_dump_interval must be positive")

    sample_kwargs = None
    if args.attention_dump:
        sample_kwargs = {
            "attention_capture": True,
            "attention_interval": args.attention_dump_interval,
        }

    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
            )
        case Default():
            checkpoint = DEFAULT_CHECKPOINT[args.env]
            return _policy_config.create_trained_policy(
                _config.get_config(checkpoint.config),
                checkpoint.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
            )


class PolicyServerApp:
    def __init__(self, args: Args):
        self.args = args
        self.base_policy = create_policy(args)
        self.policy = self.base_policy
        if args.record:
            self.policy = _policy.PolicyRecorder(self.base_policy, "policy_records")

    def serve_forever(self) -> None:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

        server = websocket_policy_server.WebsocketPolicyServer(
            policy=self.policy,
            host="0.0.0.0",
            port=self.args.port,
            metadata=self.base_policy.metadata,
        )
        server.serve_forever()


def _maybe_configure_cuda(args: Args) -> None:
    if not torch.cuda.is_available():
        return

    try:
        frac = max(0.0, min(1.0, args.max_cuda_mem_fraction))
        for dev_idx in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
        logging.info("Set CUDA memory fraction to %.2f for %d device(s)", frac, torch.cuda.device_count())
    except Exception as e:  # pragma: no cover - defensive
        logging.warning("Failed to set CUDA memory fraction: %s", e)


def main(args: Args) -> None:
    _maybe_configure_cuda(args)
    PolicyServerApp(args).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
