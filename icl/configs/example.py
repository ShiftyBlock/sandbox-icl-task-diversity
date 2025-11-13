"""Example configuration for ICL training."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal
import json


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for task generation."""
    name: str = "noisy_linear_regression"
    n_tasks: int = 1_048_576
    n_dims: int = 8
    n_points: int = 16
    batch_size: int = 256
    data_seed: int = 101
    task_seed: int = 102
    noise_seed: int = 103
    data_scale: float = 1.0
    task_scale: float = 1.0
    noise_scale: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model architecture."""
    name: str = "transformer"
    n_points: int = 16
    n_layer: int = 8
    n_embd: int = 128
    n_head: int = 2
    seed: int = 100


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training hyperparameters."""
    optimizer: Literal["adam", "adamw"] = "adam"
    lr: float = 1e-3
    schedule: Literal["triangle", "warmup_cosine_decay"] = "triangle"
    warmup_steps: int = 250_000
    total_steps: int = 500_000
    weight_decay: float = 0.0


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for evaluation."""
    n_samples: int = 1_048_576
    batch_size: int = 4_096
    data_seed: int = 104
    task_seed: int = 105
    noise_seed: int = 106
    every: int = 1000


@dataclass(frozen=True)
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    project: str = ""
    entity: str = ""
    mode: Literal["online", "offline", "disabled"] = "online"


@dataclass(frozen=True)
class Config:
    """Main configuration object."""
    dtype: str = "float32"
    work_dir: Path = field(default_factory=lambda: Path.cwd() / "experiments")
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict:
        """Convert config to dictionary, handling Path objects."""
        d = asdict(self)
        d['work_dir'] = str(self.work_dir)
        return d

    def to_json(self, **kwargs) -> str:
        """Serialize config to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """Create config from dictionary."""
        work_dir = Path(data.get('work_dir', Path.cwd() / "experiments"))

        return cls(
            dtype=data.get('dtype', 'float32'),
            work_dir=work_dir,
            task=TaskConfig(**data.get('task', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            eval=EvalConfig(**data.get('eval', {})),
            wandb=WandbConfig(**data.get('wandb', {}))
        )


def get_config() -> Config:
    """Get default configuration."""
    return Config()
