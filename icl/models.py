"""Model implementations for in-context learning."""
from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from icl.gpt2 import GPT2Config, GPT2Model
from icl.utils import to_sequence, sequence_to_targets


class BaseModel(Protocol):
    """Protocol for ICL models."""

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Make predictions given data and targets."""
        ...


class Transformer(nn.Module):
    """Transformer model for in-context learning."""

    def __init__(
        self,
        n_points: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
        seed: int,
        dtype: Any,
    ):
        super().__init__()
        self.n_points = n_points
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.seed = seed
        self.dtype = dtype

        # Build architecture
        config = GPT2Config(
            block_size=2 * n_points,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dtype=dtype,
        )

        self.input_proj = nn.Linear(n_embd + 1, n_embd, bias=False, dtype=dtype)
        self.transformer = GPT2Model(config)
        self.output_proj = nn.Linear(n_embd, 1, bias=False, dtype=dtype)

        # Initialize
        torch.manual_seed(seed)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize projection layer weights."""
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)

    def forward(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            data: Input features (batch, n_points, n_dims)
            targets: Target values (batch, n_points)
            training: Whether in training mode

        Returns:
            Predictions (batch, n_points)
        """
        # Convert to sequence
        sequence = to_sequence(data, targets)

        # Process through transformer
        embeddings = self.input_proj(sequence)
        hidden_states = self.transformer(embeddings, training=training)
        predictions = self.output_proj(hidden_states)

        # Extract target predictions
        return sequence_to_targets(predictions)


class Ridge(nn.Module):
    """Ridge regression baseline."""

    def __init__(self, lam: float, dtype: Any):
        super().__init__()
        self.lam = lam
        self.dtype = dtype

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Perform ridge regression in-context.

        Args:
            data: Input features (batch, n_points, n_dims)
            targets: Target values (batch, n_points)

        Returns:
            Predictions (batch, n_points)
        """
        batch_size, n_points, _ = data.shape
        device = data.device

        # First prediction is zero (no context)
        predictions = [torch.zeros(batch_size, dtype=self.dtype, device=device)]

        # Predict each point using all previous points
        targets_expanded = targets.unsqueeze(-1)
        for i in range(1, n_points):
            pred = self._ridge_predict(
                data[:, :i],
                targets_expanded[:, :i],
                data[:, i:i+1],
            )
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

    def _ridge_predict(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Solve ridge regression and predict.

        Args:
            X: Training features (batch, i, n_dims)
            Y: Training targets (batch, i, 1)
            X_test: Test features (batch, 1, n_dims)

        Returns:
            Predictions (batch,)
        """
        n_dims = X.shape[-1]
        device = X.device

        # Solve: (X^T X + λI) w = X^T Y
        XTX = torch.bmm(X.transpose(1, 2), X)
        ridge_term = self.lam * torch.eye(n_dims, dtype=self.dtype, device=device)
        A = XTX + ridge_term.unsqueeze(0)

        XTY = torch.bmm(X.transpose(1, 2), Y)

        # Solve in float32 for stability
        weights = torch.linalg.solve(A.float(), XTY.float()).to(self.dtype)

        # Predict
        predictions = torch.bmm(X_test, weights)
        return predictions[:, 0, 0]


class DiscreteMMSE(nn.Module):
    """Discrete MMSE estimator for task inference."""

    def __init__(self, scale: float, task_pool: torch.Tensor, dtype: Any):
        super().__init__()
        self.scale = scale
        self.dtype = dtype
        self.register_buffer('task_pool', task_pool)  # (n_tasks, n_dims, 1)

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Perform MMSE estimation in-context.

        Args:
            data: Input features (batch, n_points, n_dims)
            targets: Target values (batch, n_points)

        Returns:
            Predictions (batch, n_points)
        """
        _, n_points, _ = data.shape
        targets_expanded = targets.unsqueeze(-1)
        W = self.task_pool.squeeze(-1).T  # (n_dims, n_tasks)

        # First prediction uses mean task
        predictions = [data[:, 0] @ W.mean(dim=1)]

        # Subsequent predictions use MMSE
        for i in range(1, n_points):
            pred = self._mmse_predict(
                data[:, :i],
                targets_expanded[:, :i],
                data[:, i:i+1],
                W,
            )
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

    def _mmse_predict(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        X_test: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MMSE prediction.

        Args:
            X: Training features (batch, i, n_dims)
            Y: Training targets (batch, i, 1)
            X_test: Test features (batch, 1, n_dims)
            W: Task matrix (n_dims, n_tasks)

        Returns:
            Predictions (batch,)
        """
        # Compute likelihood for each task: p(Y | X, w)
        predictions_per_task = torch.matmul(X, W)  # (batch, i, n_tasks)
        errors = Y - predictions_per_task  # (batch, i, n_tasks)

        # Log likelihood under Gaussian noise
        dist = Normal(0, self.scale)
        log_likelihoods = dist.log_prob(errors).sum(dim=1).to(self.dtype)  # (batch, n_tasks)

        # Posterior distribution over tasks
        task_posterior = F.softmax(log_likelihoods, dim=1)  # (batch, n_tasks)

        # MMSE task estimate
        w_mmse = torch.matmul(task_posterior, W.T).unsqueeze(-1)  # (batch, n_dims, 1)

        # Predict
        predictions = torch.bmm(X_test, w_mmse)
        return predictions[:, 0, 0]


def create_model(name: str, **kwargs) -> BaseModel:
    """Factory function for creating models.

    Args:
        name: Model name ('transformer', 'ridge', or 'discrete_mmse')
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model

    Raises:
        ValueError: If model name is unknown
    """
    models = {
        "transformer": Transformer,
        "ridge": Ridge,
        "discrete_mmse": DiscreteMMSE,
    }

    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(models.keys())}")

    return models[name](**kwargs)


def get_model_name(model: BaseModel) -> str:
    """Get human-readable model name."""
    if isinstance(model, Ridge):
        return "Ridge"
    elif isinstance(model, DiscreteMMSE):
        return "dMMSE"
    elif isinstance(model, Transformer):
        return "Transformer"
    else:
        return model.__class__.__name__
