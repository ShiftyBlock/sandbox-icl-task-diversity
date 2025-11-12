from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import icl.utils as u
from icl.gpt2 import GPT2Config, GPT2Model


########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


def get_model_name(model):
    if isinstance(model, Ridge):
        return "Ridge"
    elif isinstance(model, DiscreteMMSE):
        return "dMMSE"
    elif isinstance(model, Transformer):
        return "Transformer"
    else:
        raise ValueError(f"model type={type(model)} not supported")


########################################################################################################################
# Transformer                                                                                                          #
########################################################################################################################


class Transformer(nn.Module):
    def __init__(self, n_points: int, n_layer: int, n_embd: int, n_head: int, seed: int, dtype: Any):
        super().__init__()
        self.n_points = n_points
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.seed = seed
        self.dtype = dtype

        config = GPT2Config(
            block_size=2 * n_points,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dtype=dtype,
        )
        self._in = nn.Linear(n_embd + 1, n_embd, bias=False, dtype=dtype)
        self._h = GPT2Model(config)
        self._out = nn.Linear(n_embd, 1, bias=False, dtype=dtype)

        # Initialize weights
        torch.manual_seed(seed)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self._in.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self._out.weight, mean=0.0, std=0.02)

    def forward(self, data: torch.Tensor, targets: torch.Tensor, training: bool = False) -> torch.Tensor:
        input_seq = u.to_seq(data, targets)
        embds = self._in(input_seq)
        outputs = self._h(input_embds=embds, training=training)
        preds = self._out(outputs)
        preds = u.seq_to_targets(preds)
        return preds


########################################################################################################################
# Ridge                                                                                                                #
########################################################################################################################


class Ridge(nn.Module):
    def __init__(self, lam: float, dtype: Any):
        super().__init__()
        self.lam = lam
        self.dtype = dtype

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        batch_size, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # batch_size x n_points x 1
        preds = [torch.zeros(batch_size, dtype=self.dtype, device=data.device)]
        preds.extend(
            [self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], self.lam) for _i in range(1, n_points)]
        )
        preds = torch.stack(preds, dim=1)
        return preds

    def predict(self, X: torch.Tensor, Y: torch.Tensor, test_x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            lam: (float)
        Return:
            batch_size (float)
        """
        _, _, n_dims = X.shape
        XT = X.transpose(1, 2)  # batch_size x n_dims x i
        XT_Y = XT @ Y  # batch_size x n_dims x 1
        ridge_matrix = torch.matmul(XT, X) + lam * torch.eye(n_dims, dtype=self.dtype, device=X.device).unsqueeze(0)  # batch_size x n_dims x n_dims
        # batch_size x n_dims x 1
        ws = torch.linalg.solve(ridge_matrix.float(), XT_Y.float()).to(self.dtype)
        pred = test_x @ ws  # @ should be ok (batched row times column)
        return pred[:, 0, 0]


########################################################################################################################
# MMSE                                                                                                                #
########################################################################################################################


class DiscreteMMSE(nn.Module):
    def __init__(self, scale: float, task_pool: torch.Tensor, dtype: Any):
        super().__init__()
        self.scale = scale
        self.dtype = dtype
        self.register_buffer('task_pool', task_pool)  # n_tasks x n_dims x 1

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        _, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # batch_size x n_points x 1
        W = self.task_pool.squeeze().T  # n_dims x n_tasks
        preds = [data[:, 0] @ W.mean(dim=1)]  # batch_size
        preds.extend(
            [
                self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], W, self.scale)
                for _i in range(1, n_points)
            ]
        )
        preds = torch.stack(preds, dim=1)  # batch_size x n_points
        return preds

    def predict(self, X: torch.Tensor, Y: torch.Tensor, test_x: torch.Tensor, W: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            W: n_dims x n_tasks (float)
            scale: (float)
        Return:
            batch_size (float)
        """
        # X @ W is batch_size x i x n_tasks, Y is batch_size x i x 1, so broadcasts to alpha being batch_size x n_tasks
        dist = Normal(0, scale)
        alpha = dist.log_prob(Y - torch.matmul(X, W)).to(self.dtype).sum(dim=1)
        # softmax is batch_size x n_tasks, W.T is n_tasks x n_dims, so w_mmse is batch_size x n_dims x 1
        w_mmse = torch.matmul(F.softmax(alpha, dim=1), W.T).unsqueeze(-1)
        # test_x is batch_size x 1 x n_dims, so pred is batch_size x 1 x 1
        pred = test_x @ w_mmse
        return pred[:, 0, 0]


########################################################################################################################
# Get Model                                                                                                            #
########################################################################################################################

Model = Transformer | Ridge | DiscreteMMSE


def get_model(name: str, **kwargs) -> Model:
    models = {"transformer": Transformer, "ridge": Ridge, "discrete_mmse": DiscreteMMSE}
    return models[name](**kwargs)
