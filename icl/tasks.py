"""Task definitions for in-context learning."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Protocol

import torch

from icl.models import BaseModel, create_model

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


Sampler = Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def get_task_name(task: "Task") -> str:
    return "Latent" if task.name.endswith("(0)") else "Pretrain"


########################################################################################################################
# Noisy Linear Regression                                                                                              #
########################################################################################################################


@dataclasses.dataclass
class NoisyLinearRegression:
    n_tasks: int
    n_dims: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    dtype: Any
    device: str = 'cpu'

    def __post_init__(self):
        self.data_generator = torch.Generator(device=self.device).manual_seed(self.data_seed)
        self.task_generator = torch.Generator(device=self.device).manual_seed(self.task_seed)
        self.noise_generator = torch.Generator(device=self.device).manual_seed(self.noise_seed)
        self.task_pool = self.generate_task_pool() if self.n_tasks > 0 else None

    @property
    def name(self) -> str:
        return f"NoisyLinReg({self.n_tasks})"

    @classmethod
    def from_task_pool(cls, task_pool: torch.Tensor, **kwargs) -> "NoisyLinearRegression":
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        return task

    def generate_task_pool(self) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(self.task_seed)
        shape = self.n_tasks, self.n_dims, 1
        tasks = torch.randn(shape, dtype=self.dtype, device=self.device, generator=gen) * self.task_scale
        return tasks

    def sample_data(self, step: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(self.data_seed + step)
        shape = self.batch_size, self.n_points, self.n_dims
        data = torch.randn(shape, dtype=self.dtype, device=self.device, generator=gen) * self.data_scale
        return data

    def sample_tasks(self, step: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(self.task_seed + step)
        if self.n_tasks > 0:
            idxs = torch.randint(0, self.n_tasks, (self.batch_size,), generator=gen, device=self.device)
            tasks = self.task_pool[idxs]
        else:
            shape = self.batch_size, self.n_dims, 1
            tasks = torch.randn(shape, dtype=self.dtype, device=self.device, generator=gen) * self.task_scale
        return tasks

    def evaluate(self, data: torch.Tensor, tasks: torch.Tensor, step: int) -> torch.Tensor:
        targets = (data @ tasks)[:, :, 0]
        gen = torch.Generator(device=self.device).manual_seed(self.noise_seed + step)
        noise = torch.randn(targets.shape, dtype=self.dtype, device=self.device, generator=gen) * self.noise_scale
        return targets + noise

    def sample_batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, tasks = self.sample_data(step), self.sample_tasks(step)
        targets = self.evaluate(data, tasks, step)
        return data, tasks, targets

    @staticmethod
    def evaluate_oracle(data: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        targets = (data @ tasks)[:, :, 0]
        return targets

    def get_default_eval_tasks(
        self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, **kwargs
    ) -> list["NoisyLinearRegression"]:
        del kwargs
        assert task_seed != self.task_seed
        assert data_seed != self.data_seed
        assert noise_seed != self.noise_seed
        config = dataclasses.asdict(self)
        config["batch_size"] = batch_size
        config["task_seed"] = task_seed
        config["data_seed"] = data_seed
        config["noise_seed"] = noise_seed
        config["n_tasks"] = 0
        eval_tasks = [self.__class__(**config)]
        if self.n_tasks > 0:
            config["n_tasks"] = self.n_tasks
            eval_tasks.append(NoisyLinearRegression.from_task_pool(**config, task_pool=self.task_pool.clone()))
        return eval_tasks

    def get_default_eval_models(self) -> list[BaseModel]:
        models = [create_model(name="ridge", lam=self.noise_scale**2 / self.task_scale**2, dtype=self.dtype)]
        if self.n_tasks > 0:
            assert self.task_scale == 1.0  # TODO
            models.append(
                create_model(
                    name="discrete_mmse", scale=self.noise_scale, task_pool=self.task_pool.clone(), dtype=self.dtype
                )
            )
        return models


########################################################################################################################
# Task Factory                                                                                                         #
########################################################################################################################

Task = NoisyLinearRegression


def create_task(name: str, **kwargs) -> Task:
    """Factory function for creating tasks.

    Args:
        name: Task name ('noisy_linear_regression')
        **kwargs: Task-specific arguments

    Returns:
        Instantiated task

    Raises:
        ValueError: If task name is unknown
    """
    tasks = {"noisy_linear_regression": NoisyLinearRegression}

    if name not in tasks:
        raise ValueError(f"Unknown task: {name}. Choose from: {list(tasks.keys())}")

    return tasks[name](**kwargs)
