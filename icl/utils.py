"""Utility functions for ICL training."""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from icl.models import Transformer
    from icl.configs.example import Config


def compute_config_hash(config: Config) -> str:
    """Generate deterministic hash from configuration."""
    return hashlib.md5(
        config.to_json(sort_keys=True).encode("utf-8")
    ).hexdigest()


def to_sequence(data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Convert (data, targets) pairs into interleaved sequence.

    Args:
        data: Input features of shape (batch, n_points, n_dims)
        targets: Target values of shape (batch, n_points)

    Returns:
        Interleaved sequence of shape (batch, 2*n_points, n_dims+1)
    """
    batch_size, seq_len, n_dims = data.shape
    device = data.device
    dtype = data.dtype

    # Pad data with zeros for targets
    padded_data = torch.cat([
        torch.zeros((batch_size, seq_len, 1), dtype=dtype, device=device),
        data
    ], dim=2)

    # Pad targets with zeros for data
    padded_targets = torch.cat([
        targets.unsqueeze(-1),
        torch.zeros((batch_size, seq_len, n_dims), dtype=dtype, device=device)
    ], dim=2)

    # Interleave: [target, data, target, data, ...]
    sequence = torch.stack([padded_data, padded_targets], dim=2)
    return sequence.reshape(batch_size, 2 * seq_len, n_dims + 1)


def sequence_to_targets(sequence: torch.Tensor) -> torch.Tensor:
    """Extract target predictions from sequence.

    Args:
        sequence: Interleaved sequence of shape (batch, 2*n_points, n_dims+1)

    Returns:
        Target predictions of shape (batch, n_points)
    """
    return sequence[:, ::2, 0]


def get_model_summary(model: Transformer, n_dims: int, n_points: int, batch_size: int) -> str:
    """Generate human-readable model summary.

    Args:
        model: Transformer model to summarize
        n_dims: Number of input dimensions
        n_points: Number of points in context
        batch_size: Batch size for input

    Returns:
        Formatted string with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return f"""
╔══════════════════════════════════════════════════════════╗
║                    MODEL SUMMARY                         ║
╠══════════════════════════════════════════════════════════╣
║ Total Parameters:      {total_params:>15,}              ║
║ Trainable Parameters:  {trainable_params:>15,}              ║
║ Frozen Parameters:     {frozen_params:>15,}              ║
╠══════════════════════════════════════════════════════════╣
║ Architecture:                                            ║
║   Layers:              {model.n_layer:>15}              ║
║   Embedding Dim:       {model.n_embd:>15}              ║
║   Attention Heads:     {model.n_head:>15}              ║
║   Context Length:      {n_points:>15}              ║
╠══════════════════════════════════════════════════════════╣
║ Input Configuration:                                     ║
║   Feature Dims:        {n_dims:>15}              ║
║   Context Points:      {n_points:>15}              ║
║   Batch Size:          {batch_size:>15}              ║
╚══════════════════════════════════════════════════════════╝
"""
