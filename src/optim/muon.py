"""Muon optimizer (MomentUm Orthogonalized by Newton-schulz).

Muon orthogonalizes the momentum of 2D weight matrices via a Newton-Schulz
iteration before applying the update, which empirically trains transformers
faster/more stably than Adam for the matrix parameters. Non-matrix parameters
(biases, norm gains, gates, embeddings, output heads) are updated with a standard
decoupled-weight-decay AdamW step.

Reference: Keller Jordan et al., "Muon" (modded-nanogpt). This is a single-process
implementation (no distributed gather); it works fine under DDP because each rank
holds the full parameter and computes the same orthogonalization on identical
(all-reduced) gradients.
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute an orthogonalization of G (≈ U V^T from its SVD) via Newton-Schulz."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Hybrid Muon + AdamW. Per-param-group flag `use_muon` selects the rule.

    Group keys consumed:
      use_muon (bool)
      lr (set each step by the trainer)
      momentum, nesterov, ns_steps, weight_decay         (Muon groups)
      betas, eps, weight_decay                            (AdamW groups)
    """

    def __init__(self, param_groups):
        # param_groups is a list of dicts already tagged with use_muon and hypers.
        defaults = dict(lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5,
                        weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8, use_muon=False)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            if group["use_muon"]:
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    update = g.add(buf, alpha=momentum) if nesterov else buf
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # Match the update RMS to ~0.2 across all shapes, per
                    # "Muon is Scalable for LLM Training" (arXiv:2502.16982).
                    # The orthogonalized update has RMS 1/sqrt(max(m,n)); scaling
                    # by 0.2*sqrt(max(m,n)) yields a constant per-element RMS of 0.2.
                    scale = 0.2 * (max(update.size(0), update.size(1)) ** 0.5)
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr * scale)
            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bc1 = 1 - beta1 ** t
                    bc2 = 1 - beta2 ** t
                    denom = (exp_avg_sq.sqrt() / (bc2 ** 0.5)).add_(eps)
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-lr / bc1)
        return loss
