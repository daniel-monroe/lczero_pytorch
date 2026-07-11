# AGENTS.md — PyTorch reimplementation of lczero-training

This directory (`torch_training/`) is a from-scratch **PyTorch** reimplementation of
the Leela Chess Zero training pipeline that lives in `../src/lczero_training`
(JAX/Flax) and `../tf` (legacy TensorFlow). It is fully self-contained and has its
own virtualenv, **detached from the TensorFlow/JAX requirements** (the user is
phasing out TensorFlow).

## Goals (from the user)
1. Re-implement the whole lczero-training pipeline in PyTorch.
2. Must **scale to very large models** and be **very fast** (8× A100-40GB available).
3. Train a model to reasonable results, export it, load it into the **lc0 engine**
   (`/home/admin/lc0/build/release/lc0`) and get reasonable results with no errors.
4. Support **removing the QKV biases** in attention (config flag).
5. Support replacing **LayerNorm in the main body (not smolgen) with RMSNorm** (config flag).
6. Add **Muon** as an optimizer option.
7. Separate Python environment, detached from TF requirements.
8. Keep this AGENTS.md updated with detailed progress.

## Environment
- venv: `torch_training/.venv` (Python 3.10). Activate: `. .venv/bin/activate`.
- Key deps: `torch==2.6.0+cu124` (driver is CUDA 12.4), `numpy`, `protobuf==5.29.*`,
  `grpcio-tools` (codegen), `tensorboard`, `pyyaml`. See `requirements.txt`.
- GPUs: 8× NVIDIA A100-PCIE-40GB.

## Data
- `/mnt/bt5-rescored-data/` — 2770 directories named `*.tar` (they are **directories**,
  not tar files), each containing gzipped chunks `training.*.gz`.
- Each `.gz` decompresses to a sequence of **V7 training records, 8396 bytes each**.
- V7 struct (little-endian): `4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff`
  - `version(4s) input_format(i) probs[1858](7432s) planes[104 u64](832s)`
  - `castling_us_ooo, us_oo, them_ooo, them_oo, stm/ep, rule50, invariance_info (7×B)`
  - `dep_result(b)`
  - 15 floats: root_q, best_q, root_d, best_d, root_m, best_m, plies_left,
    result_q, result_d, played_q, played_d, played_m, orig_q, orig_d, orig_m
  - `visits(I) played_idx(H) best_idx(H)`
  - `pol_kld(f) q_st(f) d_st(f) opp_played_idx(H) next_played_idx(H) extra[8](8f)`

## Tensor contract (matches JAX `tensor_generator.cc`)
Per sample the loader yields three float32 arrays (model consumes a batch of these):
- `inputs`  : `(112, 8, 8)` board planes
  - planes 0..103: bit-unpack of the 104 u64 (`bit = (plane >> (sq^7)) & 1`)
  - 104: castling_us_ooo, 105: castling_us_oo, 106: castling_them_ooo, 107: castling_them_oo
  - 108: stm/enpassant, 109: rule50/99.0, 110: all 0, 111: all 1
- `probs`   : `(1858,)` policy target (negative entries flag illegal moves)
- `values`  : `(6, 3)` rows = value types [result, best, played, orig, root, st],
  cols = `[q, d, m]`. WDL derived downstream: `w=(1+q-d)/2, d=d, l=(1-q-d)/2`.

## Architecture (ported from `../src/lczero_training/model`)
BT-style attention network with DeepNorm post-norm residuals:
- Embedding: preprocess(64×12→64×dense)→concat→Linear→act→LayerNorm→MaGating→FFN(+α)→LayerNorm
- Encoder ×N: `x = x + MHA(x)·α; ln1; ffn; ln2(out1 + ffn·α)` where
  `α=(2N)^-0.25`, deepnorm init scale `β=(8N)^-0.25`.
- MHA with optional **Smolgen** (shared weight_gen_dense across blocks) adding attention bias.
- Heads: value (wdl[3] + optional error + optional categorical), policy (attention+promotion,
  gather via fixed 1858 policy_map), movesleft.

### Embedding (single, simplest mode only)
Per the user, only the simplest embedding is supported now: lc0
`INPUT_EMBEDDING_NONE` (the daniel-monroe fork's "old" style). It is just a square
embedding `Linear(112 -> embedding_size)` + activation, with optional MaGating —
**no positional encoding, no preprocess, no LayerNorm, no FFN**. The `pe_dense` and
`pe_map` modes (and `model/pos_encoding.py`) were removed. `EmbeddingConfig` now has
only `embedding_size`.
IMPORTANT: a `NONE` (non-dense) attention net does NOT load in the old lc0 v0.31.2
(no NONE path; eigen conflates non-dense with kPosEncoding). It REQUIRES the
freshly-built lc0 master CUDA backend (see below). Verified: builds, exports, loads,
searches, returns a bestmove on the lc0 CUDA backend, no errors.

### Auxiliary heads/losses (daniel-monroe fork parity)
- **soft policy head**: just another bilinear `policy_head` (e.g. name `soft`) + a
  `losses.policy` entry with `temperature: T` (tempered visit target). No new code.
  Exported to lc0's `policy_heads.soft` field (harmless; lc0 plays the `vanilla` head).
- **opponent / next "future" heads** (`model.future_head`): tiny per-square heads
  (`FutureHead` = embedding → Linear→2) predicting a future move's **from-square**
  and **to-square** marginals. Targets come from V7 `opp_played_idx` / `next_played_idx`
  (verified populated in /mnt/bt5-rescored-data), carried in `values` **row 6**
  `[opp_idx, next_idx, nan]`; decomposed via the `get_up_down` equivalent in
  `FutureLoss` (one-hot from/to, sum-to-1; promotions→uniform). `opponent: true`
  flips the 64-square target (board flip). Sentinels (idx>=1858, game end) are masked.
  These are **training-only** — `export.py` ignores `future_heads` (lc0 never sees them).
  Config: `model.future_head: [{name, embedding_size}]` + `losses.future:
  [{head_name, target: opp|next, opponent: bool, weight}]`. Example: `configs/aux_heads.yaml`.
  Verified: future losses collapse 8.5->0.06 on overfit; exported net loads+plays in lc0.

### Config flags requested by user
- `use_ma_gating` (bool): MaGating (mult+add input gating). Set false to drop it;
  lc0 treats ip_mult_gate/ip_add_gate as optional, so it stays loadable.
  → Simplest lc0-compatible net = `input_embedding: pe_map` + `use_ma_gating: false`
    (verified: lc0 loads + searches + bestmove, no errors).
- `remove_qkv_bias` (bool): drop bias on MHA q/k/v Linear layers.
- `rmsnorm_main_body` (bool): use RMSNorm instead of LayerNorm everywhere in the
  main body (embedding norms + encoder ln1/ln2) but **NOT** inside smolgen
  (smolgen keeps LayerNorm). Note: lc0 cannot consume RMSNorm directly, so on
  export we fold/convert — see Export notes.

## FP8 training (Blackwell / Hopper / Ada) — `src/fp8.py`
Optional FP8 compute via **torchao** (`pip install torchao`; optional dep, not
needed for bf16 runs). `training.fp8.enabled: true` swaps the body `nn.Linear` GEMMs
in place for an FP8 linear that casts inputs+weights to e4m3 for the matmul but
keeps a **high-precision master `weight`**. The swapped module subclasses `nn.Linear`
and preserves `.weight`/`.bias` names, so it is transparent to the optimizer (same
FQNs → masked-WD globs still match), checkpointing (same state_dict keys), and lc0
export (`export.py` reads the high-precision weight). An FP8-trained net is
**identical in format** to a bf16 one — FP8 is purely a training speedup.

Two recipe families (`training.fp8.recipe`):
- **Scaled FP8** — `tensorwise` (per-tensor scale; default) / `rowwise` (per-row;
  needs recent torchao). Via `torchao.float8.convert_to_float8_training` →
  `Float8Linear`. Runs on Ada (8.9) / Hopper (9.0) / Blackwell (10.0+).
- **`mxfp8`** (microscaling) — e8m0 scale shared per block of `mx_block_size` (32)
  elements; the format **Blackwell** tensor cores run natively. Via
  `torchao.prototype.mx_formats` (`quantize_` with `MXLinearConfig`, falling back to
  `swap_linear_with_mx_linear`) → `MXLinear`. `mx_gemm_kernel`: `auto`(=cublas on
  Blackwell)/`cublas`/`cutlass`/`emulated`. The hardware GEMM needs cc ≥ 10.0;
  `emulated` is a bit-accurate reference that runs on any CUDA GPU (testing only).

Other config: `compile` (bool; `torch.compile` fuses the FP8 scaling/casts —
recommended, often required to beat bf16), `skip_fqns` (substrings kept in high
precision; default skips embedding/all heads/smolgen/promotion, leaving the **36**
encoder q/k/v/output_dense + ffn GEMMs in FP8). The filter also auto-skips any
Linear whose in/out dims aren't multiples of the alignment (16 for scaled FP8, the
MX block size for mxfp8).

Plumbing: keep `compute_dtype: bfloat16` (FP8 runs under bf16 autocast; GradScaler
stays off). `convert_model_to_fp8` runs after `.to(device)` and **before** the
optimizer / checkpoint load. `torch.compile` wraps DDP last (enables DDPOptimizer
bucketing); `raw_model` keeps a clean handle for checkpoint/export/clip/SWA.

HW guard: scaled FP8 needs cc ≥ 8.9; mxfp8 (non-emulated) needs cc ≥ 10.0. On this
8×A100 (sm_80) box FP8 **cannot run** — enabling it raises a clear error by design;
validate end-to-end on Blackwell. Example config: `configs/small_fp8.yaml` (set to
`mxfp8`/`auto`). Verified on A100: config parses, filter selects the 36 body GEMMs
(align 16 and 32), and the capability / emulated-relaxation / torchao-missing /
bad-recipe / bad-kernel paths all raise cleanly; bf16 path unaffected.

## Export to lc0 (`.pb.gz`)
Protobuf: `pblczero.Net` (generated `src/proto/net_pb2.py` from lc0 v0.31.2
`net.proto`). Weights are LINEAR16-quantized: per tensor store `min_val,max_val` and
`uint16` = round((w-min)/(max-min)*65535), where the stored buffer is the
JAX-logical array **transposed** then flattened. PyTorch `nn.Linear.weight` is
already `(out,in)` = transpose of Flax `(in,out)`, so a torch weight maps to the
Leela buffer by a plain row-major flatten (verify with `len//2 == numel`).
Special case: embedding kernel **row 109 divided by 99.0** on export.
Header NetworkFormat: input=INPUT_CLASSICAL_112_PLANE, network=
NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT, policy=POLICY_ATTENTION, value=VALUE_WDL,
moves_left=MOVES_LEFT_V1, input_embedding=INPUT_EMBEDDING_PE_DENSE,
default_activation=MISH, smolgen_activation=SWISH, ffn_activation=DEFAULT.

## How to run (all from `/home/admin/daniel/lczero-training/torch_training/`)
First: `cd /home/admin/daniel/lczero-training/torch_training && . .venv/bin/activate`
- Single GPU: `CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/run1.yaml`
- Multi-GPU (REQUIRED env on this box, see NCCL note):
  `NCCL_P2P_DISABLE=1 PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=8 -m src.train --config configs/run8.yaml`
- Export a checkpoint by hand (from Python): build `LczeroModel(cfg.model)`, load
  `ckpt["model"]` into it, then `src.export.export_model(model, path)`.
- Validate in lc0 (GPU, latest master — use this for non-dense embeddings):
  `LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
   /home/admin/daniel/lc0-src/build/cuda/lc0 --backend=cuda --weights=<net.pb.gz>`
  then UCI: `uci` / `isready` / `position startpos` / `go nodes 200`.
- Older CPU-only fallback (PE_DENSE nets only): `/home/admin/lc0/build/release/lc0
  --backend=eigen --weights=<net.pb.gz>`.

## lc0 builds available
- `/home/admin/daniel/lc0-src` — lc0 **v0.33.0-dev** (master, commit d8ce482), built
  with the **CUDA** backend. Supports INPUT_EMBEDDING_NONE/PE_MAP/PE_DENSE. Binary:
  `build/cuda/lc0`. THIS is the one to use (GPU + all embedding modes).
  - Build gotcha: the box's `/usr/local/cuda` → CUDA 13, but the driver is 12.4, so
    pin everything to 12.4. Configure with:
    `env -i HOME=$HOME PATH=/usr/local/cuda-12.4/bin:/usr/bin:/bin:$HOME/.local/bin
     meson setup build/cuda --buildtype=release -Dcudnn=false -Dgtest=false
     -Dpython_bindings=false "-Dcudnn_libdirs=['/usr/local/cuda-12.4/lib64']"
     "-Dcudnn_include=['/usr/local/cuda-12.4/include']"` then `ninja -C build/cuda`.
    (Without pinning includes, nvcc 12.4 picks up CUDA-13 headers → `__cudaLaunch`
    macro errors.)
- `/home/admin/lc0/build/release` — old v0.31.2, eigen/CPU only (PE_DENSE nets only).
- Reference fork: `/home/admin/daniel/dm-lczero-training` (daniel-monroe/lczero-training,
  TF) — source of the embedding_style logic (`tf/tfprocess.py`).
  A copy of the final net is at `/home/admin/lc0/build/release/torchnet.pb.gz`, which
  lc0 **auto-discovers**, so plain `./lc0` (run from that dir) loads it with no flags.

## ⚠ NCCL on this 8×A100-PCIE box
GPU-to-GPU P2P over PCIe is broken here: any NCCL collective across >2 GPUs HANGS
(100% GPU busy-wait, no progress, no error). **Always set `NCCL_P2P_DISABLE=1`**
for multi-GPU runs. Verified: 2-GPU works without it; 4- and 8-GPU all-reduce +
DDP training work *with* it.

## Throughput / scale
- Data loader: ~22k pos/s with 8 workers single-process; lazy directory-level
  discovery (2770 source dirs, ~70M chunk files, no global file list — memory bounded).
- Single A100: ~13k pos/s end-to-end (15M-param net, bf16). DDP scales across the
  8×A100 box (data-parallel, per-rank dataloaders with distinct seeds).
- Large models: increase embedding_size/num_blocks/d_model; bf16 autocast keeps
  params in fp32 master copy; grad_accum for very large effective batch.

## Status log
- [done] Explored JAX codebase (model, loader, convert, training).
- [done] Created detached venv (torch 2.6 cu124), generated net_pb2.py.
- [done] Built lc0 BLAS/eigen-only binary for validation.
- [done] config.py, model (network.py + flags), loss.py, optim (nadamw/sgd/muon + lr).
- [done] data.py V7 loader — plane unpack verified against startpos (32 pieces, 20 legal moves).
- [done] export.py — VALIDATED end-to-end: lc0 loads exported net, searches, returns bestmove, no errors.
- [done] train.py — AMP/bf16, grad-accum, grad-clip, masked WD, SWA, checkpoint/resume, TensorBoard, DDP.
- [done] Trained run1 (15M params, 24000 steps, single A100, bf16): loss 4.58 -> 2.66,
  ~12.8k pos/s. Nets exported at steps 8000/16000/24000; checkpoints resumable.
- [done] VALIDATED trained net in lc0: bestmove d2d4, PV is a coherent Queen's Gambit
  line, policy ranks d4/Nf3/c4/e4 highest and g4/f3/Nh3 lowest, value WL≈+0.03 D≈0.68,
  MLH≈205 plies. No errors, no NaN.
- Multi-GPU fixed (NCCL_P2P_DISABLE=1); DDP verified on 2 GPUs + 8-GPU all-reduce.
- [done] FP8 training support (`src/fp8.py` + `training.fp8` config, torchao). Body
  GEMMs → Float8Linear (`tensorwise`/`rowwise`) or **MXLinear (`mxfp8`,
  Blackwell-native microscaling)**; master weights/ckpt/export unchanged. Scaled FP8
  needs cc≥8.9, mxfp8 needs cc≥10.0 (Blackwell) + torchao; can't run on this A100 box
  (guard raises). See the FP8 section above; example `configs/small_fp8.yaml`.

## RMSNorm + lc0 note
`rmsnorm_main_body: true` trains with RMSNorm in the body. lc0 0.31.2 applies
LayerNorm in the attention body, so an RMSNorm net is exported with gammas=weights
and betas=0 — it LOADS and RUNS, but is not numerically identical (no mean
subtraction in training vs lc0's LayerNorm). For an exactly-lc0-faithful net, keep
`rmsnorm_main_body: false` (the default). The flag is for large-scale training
efficiency experiments. `remove_qkv_bias: true` exports zero q/k/v biases so lc0
still loads the net.

## Layout (all under `/home/admin/daniel/lczero-training/torch_training/`)
```
torch_training/
  AGENTS.md                  # this file — restart-proof handoff
  requirements.txt
  .venv/                     # detached venv (torch 2.6 cu124)
  configs/
    small.yaml               # 15M net, batch 1024 (base template)
    run1.yaml                # the trained single-GPU run (24000 steps) — absolute paths
    run8.yaml / run4.yaml    # multi-GPU configs (batch 2048)
  src/
    __init__.py
    config.py                # dataclasses + YAML loader (RootConfig)
    proto/net_pb2.py         # generated from lc0 v0.31.2 net.proto (pblczero.Net)
    model/
      __init__.py
      network.py             # LczeroModel: embedding, encoder+smolgen, heads, flags
      utils.py               # activations, dtype, make_norm (LayerNorm/RMSNorm)
      policy_map.py          # fixed 1858 gather indices
    loss.py                  # LczeroLoss (policy CE/KL, value WDL, MLH, value_error, categorical, L2)
    optim/
      lr_schedule.py         # piecewise const/linear/cosine schedule
      muon.py                # Muon (Newton-Schulz) + AdamW fallback
      build.py               # build_optimizer (nadamw/sgd/muon), masked WD, apply_lr
    data.py                  # V7 IterableDataset + make_dataloader (lazy, multiprocess)
    train.py                 # training loop: AMP, grad-accum, clip, SWA, ckpt, TB, DDP
    export.py                # model_to_net / export_model -> .pb.gz (LINEAR16)
  runs/run1/                 # the completed trained run (see Artifacts)
```

## Artifacts produced (already on disk)
- Trained nets: `runs/run1/exported/net-000{08000,16000,24000}.pb.gz` (24000 = final).
- Checkpoints (resumable, full state): `runs/run1/checkpoint/ckpt-{8000,16000,24000}.pt`.
  Each is `torch.save({step, model, optimizer, num_averages, swa_model})`.
- TensorBoard logs: `runs/run1/tb/`.
- lc0 auto-discovered copy: `/home/admin/lc0/build/release/torchnet.pb.gz`.

## Reference sources (read these if re-deriving anything)
- Original JAX/Flax model: `/home/admin/daniel/lczero-training/src/lczero_training/model/*.py`.
- Original JAX↔Leela conversion: `.../src/lczero_training/convert/*.py` (param↔proto map).
- Original training/optimizer/lr: `.../src/lczero_training/training/*.py`.
- lc0 engine + proto: `/home/admin/lc0` (binary `build/release/lc0`, proto
  `libs/lczero-common/proto/net.proto`). lc0 git tag v0.31.2.
- V7 chunk parser reference: `/mnt/lczero-training-bt5/tf/chunkparser.py` (struct/offsets).
- Data: `/mnt/bt5-rescored-data/` (2770 dirs × ~25k `training.*.gz`, V7, already rescored).

## Gotchas / environment quirks (IMPORTANT for a restart)
- **Shell has `set -e`**: any command in a chain that returns nonzero (e.g. `pkill`
  with no match) aborts the whole chain. Append `|| true` to such commands.
- **Multi-GPU needs `NCCL_P2P_DISABLE=1`** (see NCCL section) or it hangs silently.
- **py-spy / sudo unavailable** (no passwordless sudo, ptrace restricted) — can't
  attach to stuck processes; diagnose by logging instead.
- **lc0 CUDA backend doesn't build** here (CUDA toolkit too new for 0.31.2); use
  `--backend=eigen` (CPU) for validation.
- **protoc**: system `protoc` is 3.12 (too old for protobuf 7 runtime). net_pb2.py was
  generated with `grpcio-tools` (protobuf 5.29) — regenerate with
  `python -m grpc_tools.protoc --proto_path=/home/admin/lc0/libs/lczero-common/proto --python_out=src/proto net.proto`.
- Background launches: prefer the harness `run_in_background` or `nohup ... &` with an
  absolute log path; the harness resets cwd between Bash calls.

## Verified facts (don't re-litigate)
- Plane unpack is correct (startpos → 32 pieces, 20 legal moves, king planes==1).
- Export round-trips: lc0 loads the net, searches, returns bestmove, no NaN/errors.
- Trained net is genuinely good: policy ranks d4/Nf3/c4/e4 top, g4/f3/Nh3 bottom;
  value WL≈+0.03, D≈0.68; MLH≈205 plies.
- 8-GPU NCCL all-reduce + 2-GPU DDP training both pass (with P2P disabled).

## Suggested next steps (if continuing)
1. **Large/fast run**: launch `run8.yaml` with `NCCL_P2P_DISABLE=1` on all 8 GPUs;
   bump `embedding_size`/`num_blocks`/`d_model` and `batch_size` for a bigger net.
2. Add more heads/losses already supported: value `has_error_output` + `value_error`
   loss; `num_categorical_buckets` + `value_categorical`; a second policy KL loss;
   extra value heads named `q`/`st` (proto fields exist).
3. Try **Muon** end-to-end on a real run (`optimizer.kind: muon`) and compare vs nadamw.
4. Try the user flags on a real run: `remove_qkv_bias: true`, `rmsnorm_main_body: true`
   (RMSNorm export is approximate in lc0 — see RMSNorm note).
5. Resume works automatically: re-run the same config; `train.py` picks up the latest
   `ckpt-*.pt` in `checkpoint_path`.
6. Optional: SWA (`training.swa.enabled: true`) and `export.export_swa: true`.

## Research program: maximize 12h single-GPU final performance (run4_soft baseline)

**Goal:** starting from the `run4_soft` config, maximize eval performance after **12 h
wall-clock on ONE A100-PCIE-40GB**. Baseline `run4_soft` = vanilla policy + soft policy
(T=4, weight 8) + future heads (opponent, next; weight 0.1) + value winner/q/st (+error
+categorical) + movesleft; aux_policy_depth 64; 8×256 encoder + smolgen; RMSNorm body;
Muon. Note: soft-policy-T4×8 and the future/opponent heads and value-error heads are
already KataGo acceleration tricks (Wu 2019, arXiv:1902.10565; KataGoMethods.md).

**Hard constraint:** the `training.optimizer` block AND `training.lr_schedule` are FROZEN
(no edits). Gains must come only from (a) architecture, (b) throughput / training speed
(more useful steps inside the 12 h), or (c) auxiliary losses/targets. Data-generation
tricks (playout-cap randomization, policy-target pruning) do not apply — fixed dataset.
**Architectural changes may increase FLOPs/params by at most a few percent** — i.e. they
must reshape optimization/representation (norms, caps, small biases), NOT add capacity
(capacity/width scaling is already confirmed not to help; see below). Auxiliary-loss and
distillation variants add training-time compute (extra forward passes) which competes with
the wall-clock step budget — prefer cheap-per-step forms (or precompute teacher targets).

**Horizon/metric:** ~1.6 steps/s on 1 GPU (batch 4096) → ~67k steps in 12 h; budget is
wall-clock so throughput wins buy more steps. Primary metric: eval `policy/ce` (vanilla)
at 12 h; secondary eval `value/winner` + composite loss + policy top-1.

### Tried — does NOT help (do not re-propose)
- Gated FFN / GLU / SwiGLU.
- Global pooling (KataGo-style).
- Varying FFN width (`dff`) and head size / head count / d_model width — capacity/width
  scaling in general.
- Deeper / more future heads (extra lookahead plies, e.g. i+3/i+4).
Implication: capacity/width and pooling are exhausted → the net is not obviously
capacity-limited. Pivot the search to (1) throughput → more steps, (2) richer supervision
(e.g. teacher distillation), (3) optimization efficiency that helps the FROZEN optimizer.

**Distillation is ruled out** — this run is a proxy to prepare a LARGER run, so borrowing
another net's knowledge doesn't transfer. Prefer techniques proven in the literature that
also *matter more at scale* (validating them here de-risks the big run).

### Candidate levers still open (to evaluate)
- Throughput (tiny 8M model badly under-utilizes the A100 → launch/overhead-bound):
  `torch.compile`, `compile(mode="reduce-overhead")`/CUDA graphs, channels_last. No FP8
  (A100 = Ampere).
- Cheap optimization/stability architecture (≤ few % params/FLOPs; these help MORE at
  scale, hence high value for a large-run proxy):
  - QK-norm — RMSNorm on q,k (Henry 2020; Dehghani "ViT-22B" 2023).
  - Attention-logit soft-cap, tanh (Gemma-2, 2024).
  - Attention sink / softmax-off-by-one (Xiao 2023; Miller "off by one").
  - LayerScale (Touvron CaiT 2021) / ReZero (Bachlechner 2020) — learned residual scale.
  - Normformer (Shleifer 2021) — extra norms + head scale.
  - Value-residual learning (Zhou 2024) — residual for the attention value stream.
  - Register tokens (Darcet "ViT need registers" 2023) — a few learnable global tokens
    (distinct mechanism from global pooling, which failed here).
  - μP / μTransfer init (Yang 2021) — but interacts with the FROZEN LR; init-scaling part
    only. Highly scale-relevant; flagged for the caveat.
- Auxiliary heads — SELF-DESIGNED dense per-square targets from FUTURE BOARD PLANES. Key
  unlock: consecutive records in a chunk are the same game's plies with full 112-plane
  boards + played_idx@8344/best_idx@8346; the loader already reads them (only move-dists
  used today). Shared convention: map every future ply into the CURRENT player's frame —
  even offsets same orientation; odd offsets apply s^56 flip AND swap us/them plane groups
  (0-5 <-> 6-11) so "our/their" and squares are consistent. 13-class piece code =
  {empty, our P N B R Q K, their P N B R Q K}. Mask any target the game ends before.
  Heads: Linear(emb->C) per square (<0.1% params). Eval streams unchanged.

### Finalized 12h-race variants (baseline run4_soft + one change each)
- V1 torch.compile(reduce-overhead)+CUDA graphs — throughput.
- V2 QK-norm (RMSNorm q,k).
- V3 attention sink (0-logit column).
- V4 future OWNERSHIP at ply i+8: per square 3-way {ours, theirs, empty}.
- V5 piece DESTINATION/capture: per square occupied at ply i, predict the to-square of
  that piece's NEXT move (64-way), or if captured first, the capturing piece's identity
  (6-way). => 70-way per occupied square; empties masked. Needs move/capture tracking.
- V6 NEXT-OCCUPANT: per square, 13-class identity of the next piece to occupy it after
  the current occupant changes (first future ply the occupant differs).
- V7 4-PLY FORWARD DYNAMICS: per square, for each of plies i+1..i+4, 13-class occupancy
  (4×13 per square).

### Launch record (12h race)
Configs: `configs/exp_{base,v1_compile,v2_qknorm,v3_sink,v4_own,v5_dest,v6_nextocc,v7_dyn}.yaml`
(each = run4_soft + one change). GPU map: exp_base→0, v1→1 … v7→7 (1 A100 each),
`torchrun --standalone --nproc_per_node=1 --master_port=2953{0..7}`, output `runs/exp_*/`.
Board-aux is threaded model→loader via `train.py` `board_aux_t` (loader emits the target;
loss reads `cfg.model.board_aux`). Code: qk_norm/attention_sink in MHA; BoardAuxHead +
`ModelPrediction.board`; `_board_target`/`_occ13` in data.py; board CE in loss.py; compile
branch in train.py. All targets validated on real chunks (V5 dest matched the actual move;
occupancy perspective = s^56 flip + color swap, verified 2 squares change per ply).
Params all within +0.2%.

### Pivot: input-representation enrichment (compiled baseline)
Result so far: `compile(reduce-overhead)` ~3× throughput (≈5k→14.5k pos/s single-GPU) — a
keeper, fold into everything. The other V1-V7 levers were judged non-improving and stopped;
only `compiled` kept. New direction = enrich per-token input so the FIRST attention layer
isn't wasted (each square token starts with only its own local content). Runs (built on the
compiled baseline, `model.*` flags in src/model/network.py Embedding):
- `exp_absbias`  — learned per-square (64,emb) additive positional bias (`abs_pos_bias`).
- `exp_absglobal` — absbias + act(Linear(12*64=768 board planes -> emb)) added to ALL tokens
  (`global_board_embed`): a global board summary each token sees before layer 1.
Idea backlog (same spirit, cheap): global vector as a seeded REGISTER token (65th token) so
attention-1 is immediately useful; FiLM (scale+shift) conditioning from the global vector;
factorized rank[8]+file[8] positional embeddings; feed all 112 planes (not just 12) to the
global proj; cheap local neighbor/attack pre-mix before attention. Confirmed to user: body
omits qkv biases (`remove_qkv_bias`) and uses RMSNorm (`rmsnorm_main_body`); smolgen=LayerNorm.

⚠ RAM: 8 concurrent single-GPU runs OOM-killed a process at the default `num_workers:10 /
frame_shuffle_size:8192 / prefetch:6` (503GB box, 3 dataloaders/run × 8 runs = 240+ worker
procs → <45GB free). Fix applied to all exp configs: **num_workers 6, frame_shuffle_size
2048, prefetch 3** → ~319GB free, ~5.2k pos/s each (~55k steps/12h). Use these for any
8-wide fan-out on this box.
- Input representation: richer input embedding (currently simplest INPUT_EMBEDDING_NONE) /
  learned per-square positional embedding.
- Data weighting (not optimizer): diff-focus (implemented; run5), phase/difficulty.
