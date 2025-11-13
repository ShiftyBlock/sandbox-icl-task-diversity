"""GPT-2 transformer implementation in PyTorch.

Based on nanoGPT: https://github.com/karpathy/nanoGPT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GPT2Config:
    """Configuration for GPT-2 style transformer."""
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    dtype: Any = torch.float32


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with flash attention support."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by num heads"

        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V projections for all heads in parallel
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=config.dtype)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=config.dtype)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
            persistent=False
        )

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal self-attention with scaled dot-product
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        if training:
            attn = self.attn_dropout(attn)

        # Aggregate values
        y = attn @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.out_proj(y)
        if training:
            y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias, dtype=config.dtype)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=config.dtype)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate='tanh')
        x = self.fc2(x)
        return self.dropout(x) if training else x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias, dtype=config.dtype)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias, dtype=config.dtype)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x


class GPT2Model(nn.Module):
    """GPT-2 transformer model."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Positional embeddings
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd, dtype=config.dtype)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias, dtype=config.dtype)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply scaled init to residual projections
        for name, param in self.named_parameters():
            if name.endswith('out_proj.weight') or name.endswith('fc2.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 paper."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_embds: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Forward pass through transformer.

        Args:
            input_embds: Input embeddings of shape (batch, seq_len, n_embd)
            training: Whether in training mode

        Returns:
            Output embeddings of shape (batch, seq_len, n_embd)
        """
        B, T, C = input_embds.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Add positional embeddings
        positions = torch.arange(T, dtype=torch.long, device=input_embds.device).unsqueeze(0)
        pos_embds = self.pos_emb(positions)
        x = input_embds + pos_embds

        if training:
            x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Final layer norm
        return self.ln_f(x)
