import json
import os
import logging

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import icl.utils as u
import wandb
from icl.evaluate import Preds, get_bsln_preds, get_model_preds, mse
from icl.models import Transformer, get_model
from icl.optim import get_optimizer_and_lr_schedule
from icl.tasks import Sampler, Task, get_task, get_task_name


def initialize(model: Transformer, config: ConfigDict, device: str) -> torch.nn.Module:
    model = model.to(device)
    return model


def get_batch_sampler(task: Task) -> Sampler:
    def sample_batch(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = task.sample_batch(step)
        return batch
    return sample_batch


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, data: torch.Tensor, targets: torch.Tensor) -> float:
    model.train()
    optimizer.zero_grad()

    preds = model(data, targets, training=True)
    loss = torch.square(preds - targets).mean()

    loss.backward()
    optimizer.step()

    return loss.item()


def eval_step(model: nn.Module, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        preds = model(data, targets, training=False)
    return preds


def _init_log(bsln_preds: Preds, n_dims: int) -> dict:
    log = {"train/step": [], "train/lr": []}
    for _task_name, _task_preds in bsln_preds.items():
        log[f"eval/{_task_name}"] = {}
        for _bsln_name, _bsln_preds in _task_preds.items():
            log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"] = []
            if _bsln_name != "True":
                _errs = mse(_bsln_preds, _task_preds["True"]) / n_dims
                log[f"eval/{_task_name}"][f"{_bsln_name} | True"] = _errs.tolist()
    return log


def train(config: ConfigDict) -> None:
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Setup train experiment
    exp_name = f"train_{u.get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)
    logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")

    # Experiment completed?
    if os.path.exists(os.path.join(exp_dir, "log.json")):
        logging.info(f"{exp_name} already completed")
        return None

    # Config
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())

    # Model, optimizer and lr schedule
    dtype = getattr(torch, config.dtype)
    model = get_model(**config.model, dtype=dtype)
    logging.info(u.tabulate_model(model, config.task.n_dims, config.task.n_points, config.task.batch_size))
    model = initialize(model, config, device)
    optimizer, lr_schedule = get_optimizer_and_lr_schedule(**config.training, model=model)
    logging.info("Initialized Model, Optimizer and LR Schedule")

    # Data samplers
    train_task = get_task(**config.task, dtype=dtype, device=device)
    sample_train_batch = get_batch_sampler(train_task)
    batch_samplers = {
        get_task_name(task): get_batch_sampler(task)
        for task in train_task.get_default_eval_tasks(**config.eval)
    }
    logging.info("Initialized Data Samplers")

    # Evaluate baselines
    logging.info("Evaluate Baselines...")
    bsln_preds = get_bsln_preds(train_task, batch_samplers, config.eval.n_samples, config.eval.batch_size, device)

    # Loggers
    log = _init_log(bsln_preds, config.task.n_dims)
    wandb.init(config=config.to_dict(), name=exp_name, **config.wandb)

    # Training loop
    logging.info("Start Train Loop")
    for i in range(1, config.training.total_steps + 1):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule(i)

        # Train step
        data, _, targets = sample_train_batch(i)
        data, targets = data.to(device), targets.to(device)
        loss = train_step(model, optimizer, data, targets)

        # Evaluate
        if i % config.eval.every == 0 or i == config.training.total_steps:
            # Log step and lr
            logging.info(f"Step: {i}, Loss: {loss:.6f}")
            log["train/step"].append(i)
            log["train/lr"].append(lr_schedule(i))
            wandb.log({"train/lr": lr_schedule(i), "train/loss": loss}, step=i)

            # Evaluate model
            eval_preds = get_model_preds(model, batch_samplers, config.eval.n_samples, config.eval.batch_size, device)

            # Log model evaluation
            for _task_name, _task_preds in bsln_preds.items():
                for _bsln_name, _bsln_preds in _task_preds.items():
                    _errs = mse(eval_preds[_task_name]["Transformer"], _bsln_preds) / config.task.n_dims
                    log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"].append(_errs.tolist())
                    wandb.log({f"eval/{_task_name}/{_bsln_name}": _errs.mean().item()}, step=i)

    # Checkpoint
    checkpoint_path = os.path.join(exp_dir, f"checkpoint_{i}.pt")
    torch.save({
        'step': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    # Save logs
    with open(os.path.join(exp_dir, "log.json"), "w") as f:
        f.write(json.dumps(log))

    logging.info("Training completed!")
    return None
