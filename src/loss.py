"""Loss functions, ported from ../src/lczero_training/model/loss_function.py.

The batch is a 4-tuple:
  inputs        (B, 112, 8, 8)
  probabilities (B, 1858)       negative entries flag illegal moves
  values        (B, 6, 3)       rows=value types [q, d, m]
  future        (B, 2, 1858)    next-ply move DISTRIBUTIONS for the future heads:
                                row 0 = opponent reply, row 1 = our next move;
                                negatives flag illegal; all -1 == unavailable.
All per-sample losses return a tensor of shape (B,); the trainer means over B.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import LossConfig
from .model import LczeroModel, ModelPrediction
from .model.policy_map import POLICY_MAP

# Constant (1858 x 64) selection matrices mapping a move distribution to its
# from-square and to-square marginals (promotion moves, which live outside the
# 64x64 grid, contribute to neither). This is the get_up_down decomposition.
_FROM_MAT = torch.zeros(1858, 64)
_TO_MAT = torch.zeros(1858, 64)
for _m, _g in enumerate(POLICY_MAP):
    if _g < 4096:
        _FROM_MAT[_m, _g // 64] = 1.0
        _TO_MAT[_m, _g % 64] = 1.0


def _q_from_wdl(wdl_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(wdl_logits, dim=-1)
    qw = torch.tensor([1.0, 0.0, -1.0], device=wdl_logits.device, dtype=wdl_logits.dtype)
    return probs @ qw


def _wdl_from_qd(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    w = (1.0 + q - d) / 2.0
    l = (1.0 - q - d) / 2.0
    return torch.stack([w, d, l], dim=-1)


class PolicyLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.head = cfg.head_name
        self.metric = cfg.metric_name or cfg.head_name
        self.weight = cfg.weight
        self.type = cfg.type.lower()
        self.illegal = cfg.illegal_moves.lower()
        self.temperature = cfg.temperature if cfg.temperature > 0 else 1.0

    def _ce_neg_ent(self, pred: ModelPrediction, probs):
        """Per-sample (cross_entropy, -H(target)).

        KLD(target || pred) = CE - H(target) = ce + neg_ent, since
        neg_ent = sum(p * log p) = -H(target).
        """
        logits = pred.policy[self.head]
        targets = probs.to(logits.dtype)
        if self.illegal == "mask":
            logits = torch.where(targets >= 0, logits, torch.full_like(logits, float("-inf")))
        targets = F.relu(targets)
        if self.temperature != 1.0:
            targets = targets.pow(1.0 / self.temperature)
            s = targets.sum(dim=-1, keepdim=True)
            targets = targets / torch.where(s > 0, s, torch.ones_like(s))
        # normalize targets to a distribution
        tsum = targets.sum(dim=-1, keepdim=True)
        targets_norm = targets / torch.where(tsum > 0, tsum, torch.ones_like(tsum))
        logp = F.log_softmax(logits, dim=-1)
        # safe cross entropy: ignore -inf*0
        ce = -(targets_norm * torch.nan_to_num(logp, neginf=0.0)).sum(dim=-1)
        neg_ent = (targets_norm * torch.log(targets_norm.clamp_min(1e-12))).sum(dim=-1)
        return ce, neg_ent

    def __call__(self, pred: ModelPrediction, values, probs) -> torch.Tensor:
        ce, neg_ent = self._ce_neg_ent(pred, probs)
        if self.type == "kl":
            return ce + neg_ent
        return ce

    def kld(self, pred: ModelPrediction, values, probs) -> torch.Tensor:
        """Per-sample KL divergence D(target || pred) = CE - H(target)."""
        ce, neg_ent = self._ce_neg_ent(pred, probs)
        return ce + neg_ent


class ValueLoss:
    def __init__(self, cfg):
        self.head = cfg.head_name
        self.metric = cfg.metric_name or cfg.head_name
        self.weight = cfg.weight
        self.value_type = cfg.value_type

    def _ce_neg_ent(self, pred: ModelPrediction, values):
        """Per-sample (cross_entropy, -H(target)) for the WDL distribution."""
        wdl_logits = pred.value[self.head][0]
        q = values[:, self.value_type, 0]
        d = values[:, self.value_type, 1]
        target = _wdl_from_qd(q, d).detach().to(wdl_logits.dtype)
        logp = F.log_softmax(wdl_logits, dim=-1)
        ce = -(target * logp).sum(dim=-1)
        neg_ent = (target * torch.log(target.clamp_min(1e-12))).sum(dim=-1)
        return ce, neg_ent

    def __call__(self, pred: ModelPrediction, values, probs) -> torch.Tensor:
        ce, _ = self._ce_neg_ent(pred, values)
        return ce

    def kld(self, pred: ModelPrediction, values, probs) -> torch.Tensor:
        """Per-sample KL divergence D(target || pred) = CE - H(target) for WDL."""
        ce, neg_ent = self._ce_neg_ent(pred, values)
        return ce + neg_ent


class MovesLeftLoss:
    def __init__(self, cfg):
        self.head = cfg.head_name
        self.metric = cfg.metric_name or cfg.head_name
        self.weight = cfg.weight
        self.value_type = cfg.value_type

    def __call__(self, pred: ModelPrediction, values, probs) -> torch.Tensor:
        ml_pred = pred.movesleft[self.head].squeeze(-1)
        target = values[:, self.value_type, 2]
        scale = 20.0
        delta = 10.0 / scale
        return F.huber_loss(ml_pred / scale, target / scale, delta=delta, reduction="none")


class ValueErrorLoss:
    def __init__(self, cfg):
        self.head = cfg.head_name
        self.metric = cfg.metric_name or cfg.head_name
        self.weight = cfg.weight
        self.value_type = cfg.value_type
        self.propagate = cfg.propagate_value_gradients

    def __call__(self, pred: ModelPrediction, values, probs) -> torch.Tensor:
        wdl_logits = pred.value[self.head][0]
        error_pred = pred.value[self.head][1].squeeze(-1)
        q_pred = _q_from_wdl(wdl_logits)
        target_q = values[:, self.value_type, 0]
        sq_err = (q_pred - target_q) ** 2
        if not self.propagate:
            sq_err = sq_err.detach()
        return (error_pred - sq_err) ** 2


class ValueCategoricalLoss:
    def __init__(self, cfg):
        self.head = cfg.head_name
        self.metric = cfg.metric_name or cfg.head_name
        self.weight = cfg.weight
        self.value_type = cfg.value_type

    def __call__(self, pred: ModelPrediction, values, probs) -> torch.Tensor:
        logits = pred.value[self.head][2]
        num_buckets = logits.shape[-1]
        target_q = values[:, self.value_type, 0]
        idx = torch.floor((target_q + 1.0) / 2.0 * num_buckets).long().clamp(0, num_buckets - 1)
        logp = F.log_softmax(logits, dim=-1)
        return -logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


class FutureLoss:
    """Auxiliary future-move loss: masked softmax cross-entropy of a future ply's
    full move policy, using the exact same illegal-move masking as the main policy
    head (logits -> -inf where the target flags the move illegal, then softmax CE
    over the legal set). The head already emits a 1858-move policy in the target
    ply's perspective (opponent heads mirror their square tokens inside the model),
    so no target flip happens here. Unavailable (game-end) plies, where every move
    is illegal, are dropped from the mean.
    """

    def __init__(self, cfg):
        self.head = cfg.head_name
        self.metric = cfg.metric_name or cfg.head_name
        self.weight = cfg.weight
        # Which row of the (B, future_plies, 1858) target tensor to supervise.
        self.row = {"opp": 0, "next": 1, "opp2": 2, "next2": 3}[cfg.target.lower()]

    def __call__(self, pred: ModelPrediction, future) -> torch.Tensor:
        logits = pred.future[self.head]                 # (B, 1858)
        target = future[:, self.row, :].to(logits.dtype)  # negatives => illegal
        # Mask illegal moves exactly like the main policy head.
        logits = torch.where(target >= 0, logits, torch.full_like(logits, float("-inf")))
        legal = F.relu(target)
        s = legal.sum(-1, keepdim=True)
        avail = (s.squeeze(-1) > 0)                     # is this future ply present?
        tnorm = legal / s.clamp_min(1e-8)               # normalized legal target
        logp = F.log_softmax(logits, dim=-1)
        # nan/-inf guard: all-illegal (absent) rows give -inf logits -> nan logp,
        # but tnorm is 0 there and avail masks them out of the mean.
        ce = -(tnorm * torch.nan_to_num(logp, nan=0.0, neginf=0.0)).sum(-1)
        per_sample = ce * avail.float()
        return per_sample.sum() / avail.float().sum().clamp_min(1.0)


class LczeroLoss:
    """Aggregates all configured losses.

    Returns (mean_total_loss, dict[name -> mean_unweighted_loss]).
    """

    def __init__(self, cfg: LossConfig, board_aux=None):
        self.policy = [PolicyLoss(c) for c in cfg.policy]
        self.value = [ValueLoss(c) for c in cfg.value]
        self.movesleft = [MovesLeftLoss(c) for c in cfg.movesleft]
        self.value_error = [ValueErrorLoss(c) for c in cfg.value_error]
        self.value_categorical = [ValueCategoricalLoss(c) for c in cfg.value_categorical]
        self.future = [FutureLoss(c) for c in cfg.future]
        self.board_aux = board_aux  # BoardAuxConfig or None (V4-V7)

    def __call__(self, model: LczeroModel, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        inputs, probs, values, future, board = batch
        pred = model(inputs)
        unweighted: Dict[str, torch.Tensor] = {}
        weighted: List[torch.Tensor] = []

        def add(group, prefix):
            for fn in group:
                per_sample = fn(pred, values, probs)
                loss = per_sample.mean()
                unweighted[f"{prefix}/{fn.metric}"] = loss.detach()
                weighted.append(loss * fn.weight)

        # Policy heads: log the configured loss AND the KLD diagnostic
        # (KLD = CE - H(target)); both surfaced for train and val.
        for fn in self.policy:
            ce, neg_ent = fn._ce_neg_ent(pred, probs)
            kld = ce + neg_ent
            loss = kld.mean() if fn.type == "kl" else ce.mean()
            unweighted[f"policy/{fn.metric}"] = loss.detach()
            unweighted[f"policy/{fn.metric}_kld"] = kld.mean().detach()
            weighted.append(loss * fn.weight)

        # Value (WDL) heads: log the CE loss AND the KLD diagnostic
        # (KLD = CE - H(target WDL)); both surfaced for train and val.
        for fn in self.value:
            ce, neg_ent = fn._ce_neg_ent(pred, values)
            loss = ce.mean()
            unweighted[f"value/{fn.metric}"] = loss.detach()
            unweighted[f"value/{fn.metric}_kld"] = (ce + neg_ent).mean().detach()
            weighted.append(loss * fn.weight)

        add(self.movesleft, "movesleft")
        add(self.value_error, "value_error")
        add(self.value_categorical, "value_categorical")

        # Future heads return an already-reduced (masked-mean) scalar.
        for fn in self.future:
            loss = fn(pred, future)
            unweighted[f"future/{fn.metric}"] = loss.detach()
            weighted.append(loss * fn.weight)

        # Dense per-square board-aux head (V4-V7): masked cross-entropy.
        if self.board_aux is not None and pred.board is not None:
            ba = self.board_aux
            logits = pred.board                              # (B, 64, C)
            if ba.kind == "dynamics":
                B = logits.shape[0]
                lg = logits.reshape(B, 64, ba.horizon, 13).permute(0, 2, 1, 3).reshape(-1, 13)
                tg = board.reshape(-1)                       # board: (B, horizon, 64)
            else:
                lg = logits.reshape(-1, logits.shape[-1])
                tg = board.reshape(-1)                       # board: (B, 64)
            if (tg != -100).any():
                loss = F.cross_entropy(lg, tg, ignore_index=-100)
            else:
                loss = logits.sum() * 0.0                    # all-masked batch: no signal
            unweighted[f"board/{ba.kind}"] = loss.detach()
            weighted.append(loss * ba.weight)

        total = torch.stack(weighted).sum()
        return total, unweighted
