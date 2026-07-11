"""Training loop for the PyTorch lczero-training reimplementation.

Single-GPU:   python -m src.train --config configs/foo.yaml
Multi-GPU:    torchrun --standalone --nproc_per_node=8 -m src.train --config ...

Features: AMP (bf16/fp16 autocast), gradient accumulation, global-norm grad
clipping, the piecewise LR schedule, masked-weight-decay NAdamW / Muon,
optional SWA, checkpointing (resumable), TensorBoard logging, and periodic export
to the Leela .pb.gz format.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from .config import RootConfig, load_config
from .data import make_dataloader
from .export import export_model
from .fp8 import convert_model_to_fp8
from .loss import LczeroLoss
from .model import LczeroModel
from .model.utils import get_dtype
from .optim.build import apply_lr, build_optimizer
from .optim.lr_schedule import LrSchedule


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def _is_main() -> bool:
    rank, _, _ = _dist_info()
    return rank == 0


def setup_distributed() -> bool:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, step: int, model, optimizer,
                    num_averages: float) -> None:
    """Full, resumable checkpoint (model + optimizer + step). SWA weights are NOT
    stored here; they live in separate weights-only `swa-<step>.pt` files."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    state = {
        "step": step,
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "num_averages": num_averages,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_swa_weights(ckpt_dir: str, step: int, swa_model) -> None:
    """Weights-ONLY checkpoint for the SWA (averaged) model: just its state_dict,
    no optimizer/step/other training state. These are what you evaluate / export
    the averaged model from (see `--eval-swa`)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"swa-{step}.pt")
    tmp = path + ".tmp"
    torch.save(swa_model.state_dict(), tmp)
    os.replace(tmp, path)


def latest_checkpoint(ckpt_dir: str, prefix: str = "ckpt-") -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    cks = [f for f in os.listdir(ckpt_dir) if f.startswith(prefix) and f.endswith(".pt")]
    if not cks:
        return None
    cks.sort(key=lambda f: int(f[len(prefix):-3]))
    return os.path.join(ckpt_dir, cks[-1])


def prune_checkpoints(ckpt_dir: str, keep: int, prefix: str = "ckpt-") -> None:
    cks = [f for f in os.listdir(ckpt_dir) if f.startswith(prefix) and f.endswith(".pt")]
    cks.sort(key=lambda f: int(f[len(prefix):-3]))
    for f in cks[:-keep] if keep > 0 else []:
        try:
            os.remove(os.path.join(ckpt_dir, f))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg: RootConfig) -> None:
    distributed = setup_distributed()
    rank, world_size, local_rank = _dist_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tcfg = cfg.training
    compute_dtype = get_dtype(cfg.model.defaults.compute_dtype)
    use_amp = compute_dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(compute_dtype == torch.float16))

    torch.manual_seed(1234 + rank)

    model = LczeroModel(cfg.model).to(device)
    if _is_main():
        nparams = sum(p.numel() for p in model.parameters())
        print(f"[rank0] model params: {nparams/1e6:.2f}M  compute_dtype={compute_dtype} "
              f"amp={use_amp} world_size={world_size}", flush=True)

    # Optional FP8 (Blackwell/Hopper/Ada): swap body Linear GEMMs for Float8Linear.
    # Done before the optimizer / checkpoint load so parameter FQNs match (the
    # swapped modules keep the same .weight/.bias names) and after .to(device) so
    # the capability check sees the real GPU.
    if tcfg.fp8.enabled:
        if compute_dtype == torch.float32 and _is_main():
            print("[fp8] WARNING: compute_dtype=float32; FP8 is intended to run "
                  "under bf16 autocast. Set model.defaults.compute_dtype=bfloat16.",
                  flush=True)
        convert_model_to_fp8(model, tcfg.fp8, device)

    optimizer = build_optimizer(model, tcfg.optimizer)
    lr_sched = LrSchedule(tcfg.lr_schedule)
    loss_fn = LczeroLoss(tcfg.losses, board_aux=cfg.model.board_aux)
    _ba = cfg.model.board_aux
    board_aux_t = (_ba.kind, _ba.horizon) if _ba is not None else None

    # SWA shadow weights (kept on the main rank only, in float32).
    swa_model = None
    num_averages = 0.0
    if tcfg.swa.enabled and _is_main():
        swa_model = copy.deepcopy(model)
        for p in swa_model.parameters():
            p.requires_grad_(False)

    start_step = 0
    ckpt_dir = tcfg.checkpoint_path
    resume = latest_checkpoint(ckpt_dir)
    if resume is not None:
        if _is_main():
            print(f"[rank0] resuming from {resume}", flush=True)
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        # load_state_dict restores the checkpoint's param-group hypers (including
        # a stale weight_decay=0.0); re-apply the freshly-built config values so
        # config changes take effect across a resume.
        intended_wd = [g["weight_decay"] for g in optimizer.param_groups]
        optimizer.load_state_dict(ckpt["optimizer"])
        for g, wd in zip(optimizer.param_groups, intended_wd):
            g["weight_decay"] = wd
        start_step = ckpt["step"]
        num_averages = ckpt.get("num_averages", 0.0)
        # SWA weights live in their own weights-only files; restore the latest.
        if swa_model is not None:
            swa_ckpt = latest_checkpoint(ckpt_dir, prefix="swa-")
            if swa_ckpt is not None:
                swa_model.load_state_dict(torch.load(swa_ckpt, map_location=device))
                if _is_main():
                    print(f"[rank0] resuming SWA weights from {swa_ckpt}", flush=True)

    # Keep a handle to the unwrapped LczeroModel for checkpointing, export, SWA and
    # grad clipping. torch.compile / DDP wrap this same module and share its
    # Parameters, so `raw_model` always has clean (prefix-free) state_dict keys and
    # the live gradients regardless of the wrappers below.
    raw_model = model
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    if tcfg.fp8.enabled and tcfg.fp8.compile:
        # Compile last so it wraps DDP (lets inductor's DDPOptimizer split the graph
        # at gradient-bucket boundaries) and fuses the FP8 scaling/casts.
        if _is_main():
            print("[fp8] torch.compile(model) enabled", flush=True)
        model = torch.compile(model)
    elif tcfg.compile:
        if _is_main():
            print(f"[compile] torch.compile(mode={tcfg.compile!r})", flush=True)
        model = torch.compile(model, mode=(None if tcfg.compile == "default" else tcfg.compile))

    # Per-rank dataloader: distinct seed per rank so ranks see different data.
    per_rank_bs = max(1, tcfg.batch_size // world_size)
    loader = make_dataloader(cfg.data, batch_size=per_rank_bs, train=True,
                             seed=1 + rank, infinite=True, diff_focus=True,
                             board_aux=board_aux_t)
    data_iter = iter(loader)

    # Two sub-run writers so every metric chart overlays its train and eval lines
    # (identical tag names under separate "train"/"eval" runs).
    import os as _os
    if _is_main():
        writer_train = SummaryWriter(_os.path.join(cfg.tensorboard_path, "train"))
        writer_eval = SummaryWriter(_os.path.join(cfg.tensorboard_path, "eval"))
    else:
        writer_train = writer_eval = None

    # Evaluation streams (rank 0 only): one over the held-out test split (val) and
    # one over the train split, so every metric is reported for BOTH train and val.
    eval_iter = trainev_iter = None
    if _is_main() and tcfg.eval_steps > 0:
        eval_iter = iter(make_dataloader(cfg.data, batch_size=per_rank_bs,
                                         train=False, seed=12345, infinite=True,
                                         board_aux=board_aux_t))
        trainev_iter = iter(make_dataloader(cfg.data, batch_size=per_rank_bs,
                                            train=True, seed=98765, infinite=True,
                                            board_aux=board_aux_t))

    grad_accum = max(1, tcfg.grad_accum)
    t0 = time.time()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, tcfg.total_steps):
        lr = lr_sched(step)
        apply_lr(optimizer, lr)

        total_loss_val = 0.0
        last_unweighted = {}
        for micro in range(grad_accum):
            batch = tuple(t.to(device, non_blocking=True) for t in next(data_iter))

            sync_ctx = model.no_sync() if (distributed and micro < grad_accum - 1) \
                else contextlib.nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=compute_dtype,
                                    enabled=use_amp):
                    total, unweighted = loss_fn(model, batch)
                scaler.scale(total / grad_accum).backward()
            total_loss_val += float(total.detach())
            last_unweighted = unweighted

        grad_norm = None
        if tcfg.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), tcfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # SWA update
        if swa_model is not None and tcfg.swa.enabled:
            if (step + 1) % tcfg.swa.period_steps == 0:
                _swa_update(swa_model, raw_model, num_averages, tcfg.swa.num_averages)
                num_averages = min(tcfg.swa.num_averages, num_averages + 1.0)

        running += total_loss_val / grad_accum

        if _is_main() and (step + 1) % tcfg.log_steps == 0:
            n = tcfg.log_steps
            dt = time.time() - t0
            pos_per_s = n * tcfg.batch_size / dt
            avg = running / n
            running = 0.0
            t0 = time.time()
            msg = (f"step {step+1}/{tcfg.total_steps} loss {avg:.4f} lr {lr:.2e} "
                   f"{pos_per_s:.0f} pos/s")
            if grad_norm is not None:
                msg += f" gnorm {float(grad_norm):.2f}"
            print(msg, flush=True)
            if writer_train is not None:
                # Train-side diagnostics (one line, on the train run).
                writer_train.add_scalar("lr", lr, step + 1)
                writer_train.add_scalar("pos_per_s", pos_per_s, step + 1)
                if grad_norm is not None:
                    writer_train.add_scalar("grad_norm", float(grad_norm), step + 1)

        if eval_iter is not None and (step + 1) % tcfg.eval_steps == 0:
            # Report EVERY metric on BOTH splits, under identical tag names in the
            # train/eval sub-runs, so each metric chart shows train + eval together.
            for w, it in ((writer_train, trainev_iter), (writer_eval, eval_iter)):
                m = evaluate(raw_model, loss_fn, it, tcfg.eval_batches,
                             device, compute_dtype, use_amp)
                tag = "train" if w is writer_train else "eval"
                print(f"[{tag} @ {step+1}] " +
                      " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
                if w is not None:
                    for k, v in m.items():
                        w.add_scalar(k, v, step + 1)
            if swa_model is not None and tcfg.swa.enabled:
                for w, it in ((writer_train, trainev_iter), (writer_eval, eval_iter)):
                    m = evaluate(swa_model, loss_fn, it, tcfg.eval_batches,
                                 device, compute_dtype, use_amp)
                    if w is not None:
                        for k, v in m.items():
                            w.add_scalar(f"swa/{k}", v, step + 1)

        if _is_main() and (step + 1) % tcfg.checkpoint_steps == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt-{step+1}.pt")
            save_checkpoint(ckpt_path, step + 1, raw_model, optimizer, num_averages)
            prune_checkpoints(ckpt_dir, tcfg.max_checkpoints_to_keep)
            if swa_model is not None:
                save_swa_weights(ckpt_dir, step + 1, swa_model)
                prune_checkpoints(ckpt_dir, tcfg.max_checkpoints_to_keep, prefix="swa-")
            # also export a network for inspection / play testing
            export_target = raw_model
            if cfg.export.export_swa and swa_model is not None:
                export_target = swa_model
            out = cfg.export.destination.format(step=step + 1)
            export_model(export_target, out, min_version=cfg.export.min_version,
                         training_steps=step + 1)
            print(f"[rank0] checkpoint + export at step {step+1} -> {out}", flush=True)

    if _is_main():
        # Final checkpoint + export.
        ckpt_path = os.path.join(ckpt_dir, f"ckpt-{tcfg.total_steps}.pt")
        save_checkpoint(ckpt_path, tcfg.total_steps, raw_model, optimizer, num_averages)
        if swa_model is not None:
            save_swa_weights(ckpt_dir, tcfg.total_steps, swa_model)
            prune_checkpoints(ckpt_dir, tcfg.max_checkpoints_to_keep, prefix="swa-")
        export_target = swa_model if (cfg.export.export_swa and swa_model is not None) else raw_model
        out = cfg.export.destination.format(step=tcfg.total_steps)
        export_model(export_target, out, min_version=cfg.export.min_version,
                     training_steps=tcfg.total_steps)
        print(f"[rank0] training done. final net -> {out}", flush=True)
        if writer_train is not None:
            writer_train.close()
            writer_eval.close()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(eval_model, loss_fn, eval_iter, n_batches, device, compute_dtype, use_amp):
    """Average the unweighted losses over n_batches of the held-out test split."""
    was_training = eval_model.training
    eval_model.eval()
    sums, total_sum, count = {}, 0.0, 0
    for _ in range(n_batches):
        batch = tuple(t.to(device, non_blocking=True) for t in next(eval_iter))
        with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=use_amp):
            total, unw = loss_fn(eval_model, batch)
        total_sum += float(total)
        for k, v in unw.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        count += 1
    if was_training:
        eval_model.train()
    out = {k: v / count for k, v in sums.items()}
    out["loss"] = total_sum / count
    return out


@torch.no_grad()
def _swa_update(swa_model, model, num_averages: float, max_avg: float) -> None:
    alpha = num_averages / (num_averages + 1.0)
    beta = 1.0 / (num_averages + 1.0)
    for ps, pm in zip(swa_model.parameters(), model.parameters()):
        ps.mul_(alpha).add_(pm.detach(), alpha=beta)
    for bs, bm in zip(swa_model.buffers(), model.buffers()):
        bs.copy_(bm)


@torch.no_grad()
def eval_swa_checkpoint(cfg: RootConfig, weights_path: str) -> dict:
    """Evaluate a weights-only checkpoint (an `swa-<step>.pt`) on the held-out test
    split, single-process. Also accepts a full `ckpt-<step>.pt` (uses its "model"
    weights) so the same command can spot-check the live model too."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    compute_dtype = get_dtype(cfg.model.defaults.compute_dtype)
    use_amp = compute_dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"
    model = LczeroModel(cfg.model).to(device)
    if cfg.training.fp8.enabled:
        convert_model_to_fp8(model, cfg.training.fp8, device)
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:   # a full checkpoint -> take its weights
        sd = sd["model"]
    model.load_state_dict(sd)
    loss_fn = LczeroLoss(cfg.training.losses)
    eval_iter = iter(make_dataloader(cfg.data, batch_size=cfg.training.batch_size,
                                     train=False, seed=0, infinite=True))
    m = evaluate(model, loss_fn, eval_iter, cfg.training.eval_batches,
                 device, compute_dtype, use_amp)
    print(f"[eval-swa] {weights_path}", flush=True)
    for k, v in m.items():
        print(f"  {k} = {v:.4f}", flush=True)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-swa", default=None,
                    help="Evaluate a weights-only SWA checkpoint (or a full ckpt) on "
                         "the test split and exit; no training.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.eval_swa:
        eval_swa_checkpoint(cfg, args.eval_swa)
        return
    train(cfg)


if __name__ == "__main__":
    main()
