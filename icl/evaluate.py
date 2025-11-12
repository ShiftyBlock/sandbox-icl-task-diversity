from typing import Callable

import torch
import torch.nn as nn

from icl.models import Model, get_model_name
from icl.tasks import Sampler, Task

# Preds = {
#     task_name: {model_name: Tensor[n_samples, n_points], ...},
#     ...
# }
Preds = dict[str, dict[str, torch.Tensor]]


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.square(a - b).mean(0)


def get_oracle_step(task: Task) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def step(xs: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        preds = task.evaluate_oracle(xs, ws)
        return preds

    return step


def get_baseline_step(model: Model) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def step(data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = model(data, targets)
        return preds

    return step


def get_bsln_preds(train_task: Task, batch_samplers: dict[str, Sampler], n_samples: int, batch_size: int, device: str) -> Preds:
    # Initialize preds and compile oracle and baseline models
    preds = {}
    oracle_step = get_oracle_step(train_task)
    bsln_models = {
        get_model_name(model): model.to(device)
        for model in train_task.get_default_eval_models()
    }

    # Loop through eval tasks
    for task_name, sample_batch in batch_samplers.items():
        # Initialize task preds
        preds[task_name] = {"True": []}
        for model_name in bsln_models:
            preds[task_name][model_name] = []

        # Accumulate preds...
        for i in range(1, n_samples // batch_size + 1):
            xs, ws, ys = sample_batch(i)
            xs, ws, ys = xs.to(device), ws.to(device), ys.to(device)
            _, n_points = ys.shape

            with torch.no_grad():
                preds[task_name]["True"].append(oracle_step(xs, ws))  # ...for oracle
                for model_name, model in bsln_models.items():  # ...for baseline models
                    preds[task_name][model_name].append(model(xs, ys))

        # Concatenate preds
        preds[task_name]["True"] = torch.cat(preds[task_name]["True"])
        for model_name in bsln_models:
            preds[task_name][model_name] = torch.cat(preds[task_name][model_name])

    return preds


def get_model_preds(
    model: nn.Module,
    batch_samplers: dict[str, Sampler],
    n_samples: int,
    batch_size: int,
    device: str,
) -> Preds:
    preds = {}
    model.eval()

    for task_name, sample_batch in batch_samplers.items():
        preds[task_name] = {"Transformer": []}
        for i in range(1, n_samples // batch_size + 1):
            xs, _, ys = sample_batch(i)
            xs, ys = xs.to(device), ys.to(device)

            with torch.no_grad():
                preds[task_name]["Transformer"].append(model(xs, ys, training=False))

        preds[task_name]["Transformer"] = torch.cat(preds[task_name]["Transformer"])

    return preds
