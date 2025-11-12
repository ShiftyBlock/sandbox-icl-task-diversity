import torch
import torch.optim as optim
from typing import Callable


def warmup_cosine_decay_schedule(warmup_steps: int, total_steps: int, lr: float) -> Callable[[int], float]:
    """Learning rate schedule with warmup and cosine decay."""
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
    return schedule


def triangle_schedule(warmup_steps: int, total_steps: int, lr: float) -> Callable[[int], float]:
    """Triangle learning rate schedule."""
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return lr * step / warmup_steps
        else:
            return lr * (total_steps - step) / (total_steps - warmup_steps)
    return schedule


def get_optimizer_and_lr_schedule(
    optimizer: str, schedule: str, model: torch.nn.Module, **kwargs
) -> tuple[optim.Optimizer, Callable[[int], float]]:
    # Learning Rate Schedule
    if schedule == "warmup_cosine_decay":
        lr_schedule = warmup_cosine_decay_schedule(kwargs["warmup_steps"], kwargs["total_steps"], kwargs["lr"])
    elif schedule == "triangle":
        lr_schedule = triangle_schedule(kwargs["warmup_steps"], kwargs["total_steps"], kwargs["lr"])
    else:
        raise NotImplementedError(f"Schedule {schedule} not implemented")

    # Weight decay mask based on nanoGPT (https://github.com/karpathy/nanoGPT)
    # Only apply weight decay to weights in the transformer blocks (_h), not to biases or embeddings
    if optimizer == "adamw":
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Apply weight decay only to kernel/weight parameters in the _h module
            if '_h' in name and 'weight' in name and 'ln' not in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': kwargs.get("weight_decay", 0.0)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        optimizer_obj = optim.AdamW(param_groups, lr=kwargs["lr"])
    elif optimizer == "adam":
        optimizer_obj = optim.Adam(model.parameters(), lr=kwargs["lr"])
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")

    return optimizer_obj, lr_schedule
