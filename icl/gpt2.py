"""Implementation based on nanoGPT (https://github.com/karpathy/nanoGPT)
"""
from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class GPT2Config:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    dtype: Any = torch.float32


class GPT2SelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=config.dtype)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=config.dtype)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        B, T, C = x.size()  # batch_size, block_size, n_embd

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att) if training else att
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y)) if training else self.c_proj(y)
        return y


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias, dtype=config.dtype)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=config.dtype)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')
        x = self.c_proj(x)
        x = self.dropout(x) if training else x
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias, dtype=config.dtype)
        self.attn = GPT2SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias, dtype=config.dtype)
        self.mlp = GPT2MLP(config)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), training=training)
        x = x + self.mlp(self.ln_2(x), training=training)
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.wpe = nn.Embedding(config.block_size, config.n_embd, dtype=config.dtype)
        self.drop = nn.Dropout(config.dropout)
        self.hs = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias, dtype=config.dtype)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_embds: torch.Tensor, training: bool = False) -> torch.Tensor:
        B, T, C = input_embds.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_embds.device).unsqueeze(0)  # (1, T)
        pos_embds = self.wpe(pos)  # (1, T, n_embd)
        x = input_embds + pos_embds  # (B, T, n_embd)
        x = self.drop(x) if training else x
        for h in self.hs:
            x = h(x, training=training)
        x = self.ln_f(x)
        return x
