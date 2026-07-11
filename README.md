# lczero-pytorch

A from-scratch **PyTorch** reimplementation of the Leela Chess Zero training
pipeline (BT-style attention network with smolgen), designed to scale to large
models and train fast on multi-GPU boxes. It is fully self-contained — no
TensorFlow or JAX dependency — and exports trained networks to the Leela
`.pb.gz` format that the [lc0](https://github.com/LeelaChessZero/lc0) engine loads.

## Features
- **Model**: square embedding → DeepNorm-scaled transformer encoder tower with
  **smolgen** → attention (bilinear) policy head + WDL value head (+ optional
  error / categorical outputs) + moves-left head.
- **Data**: fast multiprocess loader for V7 training chunks (gzip), with a
  streaming reservoir shuffle. Scales by `num_workers`; memory-bounded even on
  datasets with tens of millions of chunk files.
- **Training**: AMP (bf16/fp16), gradient accumulation, global-norm clipping,
  the piecewise const/linear/cosine LR schedule, optional SWA, resumable
  checkpointing, TensorBoard logging, and **DDP** multi-GPU.
- **Optimizers**: NAdamW with masked (decoupled) weight decay, SGD, and **Muon**
  (Newton-Schulz orthogonalized momentum, AdamW fallback for non-matrix params).
- **Export**: LINEAR16-quantized `.pb.gz` with the multihead attention-body
  network format, loadable by lc0.

### Architecture options (config flags)
- `model.use_ma_gating` — toggle the multiplicative/additive input gating.
- `model.remove_qkv_bias` — drop the attention q/k/v biases.
- `model.rmsnorm_main_body` — use RMSNorm instead of LayerNorm in the encoder
  tower (smolgen always keeps LayerNorm).
- Embedding is the simplest lc0 `INPUT_EMBEDDING_NONE` (square embedding only,
  no positional-encoding matrix, no preprocess/LayerNorm/FFN).

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e .
```
For CUDA, install the torch build matching your driver, e.g.:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Train
```bash
# single GPU
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/run1.yaml

# multi-GPU (DDP)
PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=8 \
    -m src.train --config configs/run8.yaml
```
Point `data.input_dirs` in the config at a directory of V7 `.gz` training chunks.

> Note: on boxes where GPU-to-GPU PCIe P2P is broken, prefix multi-GPU runs with
> `NCCL_P2P_DISABLE=1` or NCCL collectives will hang.

## Export to lc0
The trainer exports `.pb.gz` networks automatically at each checkpoint. To export
a checkpoint by hand, build `LczeroModel(cfg.model)`, load the checkpoint's
`model` state dict, and call `src.export.export_model(model, path)`.

Load in lc0 (an lc0 build new enough to support `INPUT_EMBEDDING_NONE` is required):
```bash
lc0 --backend=cuda --weights=net.pb.gz
```

## Layout
```
src/
  config.py            # YAML-backed dataclass config
  data.py              # V7 chunk dataset / dataloader
  loss.py              # policy / value / movesleft / value-error / categorical / L2
  train.py             # training loop, DDP, AMP, checkpoint, export
  export.py            # torch model -> lc0 .pb.gz
  model/
    network.py         # LczeroModel
    policy_map.py       # 1858-move gather indices
    utils.py            # activations, norms
  optim/               # nadamw / sgd / muon + lr schedule
  proto/net_pb2.py     # generated lc0 Net protobuf
configs/               # example YAML configs
```

See `AGENTS.md` for implementation notes, the data/tensor format, and the
torch↔Leela weight mapping.
