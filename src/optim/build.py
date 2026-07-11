"""Build optimizers from config, including masked weight decay and Muon grouping."""
from __future__ import annotations

import fnmatch
from typing import List

import torch
import torch.nn as nn

from ..config import OptimizerConfig
from .lr_schedule import LrSchedule
from .muon import Muon


def _matches(rules: List[dict], otherwise: bool, name: str) -> bool:
    for rule in rules:
        if fnmatch.fnmatch(name, rule["match"]):
            return bool(rule.get("include", True))
    return otherwise


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    kind = cfg.kind.lower()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if kind == "nadamw":
        c = cfg.nadamw
        decay, no_decay = [], []
        for n, p in named:
            if _matches(c.decay_rules, c.decay_otherwise_include, n):
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": c.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.NAdam(
            groups, lr=1e-3, betas=(c.beta_1, c.beta_2), eps=c.epsilon,
            decoupled_weight_decay=True)

    if kind == "muon":
        c = cfg.muon
        muon_params, adamw_params = [], []
        for n, p in named:
            # Default: 2D matrices use Muon, everything else uses AdamW. A name
            # matched by exclude_rules (include=True means "exclude from Muon")
            # is forced to AdamW.
            excluded = _matches(c.exclude_rules, False, n)
            if p.ndim == 2 and not excluded:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        groups = [
            {"params": muon_params, "use_muon": True, "momentum": c.momentum,
             "nesterov": c.nesterov, "ns_steps": c.ns_steps,
             "weight_decay": c.weight_decay, "lr_mul": c.lr_mul},
            {"params": adamw_params, "use_muon": False,
             "betas": (c.adamw_beta1, c.adamw_beta2), "eps": c.adamw_eps,
             "weight_decay": c.adamw_weight_decay, "lr_mul": c.adamw_lr_mul},
        ]
        return Muon(groups)

    raise ValueError(f"Unknown optimizer kind '{kind}'")


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set the LR on every param group, honoring an optional per-group lr_mul."""
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_mul", 1.0)


__all__ = ["build_optimizer", "apply_lr", "LrSchedule"]
