"""Activation + norm helpers for the model."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACT = {
    "mish": F.mish,
    "relu": F.relu,
    "none": lambda x: x,
    "identity": lambda x: x,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "selu": F.selu,
    "swish": F.silu,
    "silu": F.silu,
    "softmax": lambda x: F.softmax(x, dim=-1),
}

# Map activation name -> the lc0 net.proto ActivationFunction enum value.
# (DEFAULT=0, MISH=1, RELU=2, NONE=3, TANH=4, SIGMOID=5, SELU=6, SWISH=7,
#  RELU_2=8, SOFTMAX=9)
ACTIVATION_ENUM = {
    "default": 0,
    "mish": 1,
    "relu": 2,
    "none": 3,
    "tanh": 4,
    "sigmoid": 5,
    "selu": 6,
    "swish": 7,
    "silu": 7,
    "softmax": 9,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    key = (name or "none").lower()
    if key not in _ACT:
        raise ValueError(f"Unknown activation '{name}'")
    return _ACT[key]


DTYPES = {
    "float32": torch.float32,
    "f32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "f16": torch.float16,
}


def get_dtype(name: str) -> torch.dtype:
    return DTYPES[(name or "float32").lower()]


# LayerNorm epsilon used by the JAX model (`nnx.LayerNorm(epsilon=1e-3)`).
LN_EPS = 1e-3


def make_norm(num_features: int, use_rmsnorm: bool) -> nn.Module:
    """Create a normalization layer for the main body.

    LayerNorm matches the JAX reference (eps 1e-3, affine gamma+beta). RMSNorm is
    the user-requested alternative for the main body (no mean-subtraction, gamma
    only). Smolgen always uses LayerNorm via `make_layernorm` directly.
    """
    if use_rmsnorm:
        return nn.RMSNorm(num_features, eps=LN_EPS)
    return nn.LayerNorm(num_features, eps=LN_EPS)


def make_layernorm(num_features: int) -> nn.Module:
    return nn.LayerNorm(num_features, eps=LN_EPS)
