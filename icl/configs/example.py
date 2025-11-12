from dataclasses import dataclass, asdict
import json


@dataclass
class TaskConfig:
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


@dataclass
class ModelConfig:
    name: str = "transformer"
    n_points: int = 16
    n_layer: int = 8
    n_embd: int = 128
    n_head: int = 2
    seed: int = 100


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    schedule: str = "triangle"
    warmup_steps: int = 250_000
    total_steps: int = 500_000
    weight_decay: float = 0.0


@dataclass
class EvalConfig:
    n_samples: int = 1_048_576
    batch_size: int = 4_096
    data_seed: int = 104
    task_seed: int = 105
    noise_seed: int = 106
    every: int = 1000


@dataclass
class WandbConfig:
    project: str = ""  # Specify wandb project
    entity: str = ""
    mode: str = "online"


@dataclass
class Config:
    dtype: str = "float32"
    work_dir: str = ""  # Specify working directory
    task: TaskConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    eval: EvalConfig = None
    wandb: WandbConfig = None

    def __post_init__(self):
        if self.task is None:
            self.task = TaskConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.eval is None:
            self.eval = EvalConfig()
        if self.wandb is None:
            self.wandb = WandbConfig()

    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, **kwargs):
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        task = TaskConfig(**config_dict.get('task', {}))
        model = ModelConfig(**config_dict.get('model', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        eval_cfg = EvalConfig(**config_dict.get('eval', {}))
        wandb = WandbConfig(**config_dict.get('wandb', {}))

        return cls(
            dtype=config_dict.get('dtype', 'float32'),
            work_dir=config_dict.get('work_dir', ''),
            task=task,
            model=model,
            training=training,
            eval=eval_cfg,
            wandb=wandb
        )


def get_config() -> Config:
    """Get default configuration."""
    return Config(
        dtype="float32",
        work_dir="",  # Specify working directory
        task=TaskConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        eval=EvalConfig(),
        wandb=WandbConfig()
    )
