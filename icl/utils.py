import hashlib

import torch
from ml_collections import ConfigDict

from icl.models import Transformer


def filter_config(config: ConfigDict) -> ConfigDict:
    with config.unlocked():
        for k, v in list(config.items()):
            if v is None:
                del config[k]
            elif isinstance(v, ConfigDict):
                config[k] = filter_config(v)
    return config


def get_hash(config: ConfigDict) -> str:
    return hashlib.md5(config.to_json(sort_keys=True).encode("utf-8")).hexdigest()


def to_seq(data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, n_dims = data.shape
    dtype = data.dtype
    data = torch.cat([torch.zeros((batch_size, seq_len, 1), dtype=dtype, device=data.device), data], dim=2)
    targets = torch.cat([targets[:, :, None], torch.zeros((batch_size, seq_len, n_dims), dtype=dtype, device=data.device)], dim=2)
    seq = torch.stack([data, targets], dim=2).reshape(batch_size, 2 * seq_len, n_dims + 1)
    return seq


def seq_to_targets(seq: torch.Tensor) -> torch.Tensor:
    return seq[:, ::2, 0]


def tabulate_model(model: Transformer, n_dims: int, n_points: int, batch_size: int) -> str:
    dummy_data = torch.ones((batch_size, n_points, n_dims), dtype=model.dtype)
    dummy_targets = torch.ones((batch_size, n_points), dtype=model.dtype)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return f"""
Model Summary:
--------------
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}

Model configuration:
- n_points: {n_points}
- n_dims: {n_dims}
- batch_size: {batch_size}
- n_layer: {model.n_layer}
- n_embd: {model.n_embd}
- n_head: {model.n_head}
"""
