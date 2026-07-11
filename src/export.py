"""Export a trained PyTorch LczeroModel to the Leela .pb.gz network format.

Mirrors ../src/lczero_training/convert/{jax_to_leela,leela_pytree_visitor}.py.

Weight encoding (LINEAR16): for each tensor we store per-tensor `min_val`,
`max_val` and a uint16 buffer `round((w-min)/(max-min)*65535)`. The stored buffer
is the **JAX-logical array transposed, then row-major flattened**. Concretely:
  - nn.Linear.weight has torch shape (out, in); the JAX kernel is (in, out) = W.T,
    so the stored buffer is `W.flatten()` (C-order).  [encode(W.T) == W.flatten()]
  - The ma_gating gates have shape (64, emb) in both frameworks, so the stored
    buffer is `gate.T.flatten()`.
  - 1-D tensors (biases, norm gains/betas) flatten unchanged.
Special case: the input-embedding kernel row 109 (the rule50 plane input) is
divided by 99.0 on export (lc0 feeds that plane as 99·value).
"""
from __future__ import annotations

import gzip
import math
from typing import Optional

import numpy as np
import torch

from .config import ModelConfig
from .model import LczeroModel
from .model.utils import ACTIVATION_ENUM
from .proto import net_pb2

MAGIC = 0x1C0
_EMBEDDING_PLANE_TO_SCALE = 109
_EMBEDDING_SCALE = 99.0


# ---------------------------------------------------------------------------
# LINEAR16 encoding
# ---------------------------------------------------------------------------
def _quantize_into(layer, flat: np.ndarray) -> None:
    flat = np.ascontiguousarray(flat, dtype=np.float32)
    mn = float(flat.min())
    mx = float(flat.max())
    rng = mx - mn
    if rng > 1e-8:
        norm = (flat - mn) / rng
    else:
        norm = np.full_like(flat, 0.5)
    q = np.round(norm * 65535.0).astype(np.uint16)
    layer.min_val = mn
    layer.max_val = mx
    layer.params = q.tobytes()


def _encode_jax_logical(layer, jax_logical: np.ndarray) -> None:
    """Encode a JAX-logical array: stored buffer = jax_logical.T.flatten()."""
    _quantize_into(layer, np.ascontiguousarray(jax_logical).T.reshape(-1))


def _t(p: torch.nn.Parameter) -> np.ndarray:
    return p.detach().to(torch.float32).cpu().numpy()


def _linear_w(layer, lin: torch.nn.Linear) -> None:
    # jax kernel = W_torch.T (shape (in,out)); encode -> W_torch.flatten().
    _encode_jax_logical(layer, _t(lin.weight).T)


def _bias(layer, lin: torch.nn.Linear) -> None:
    _encode_jax_logical(layer, _t(lin.bias))


def _vec(layer, p: torch.nn.Parameter) -> None:
    _encode_jax_logical(layer, _t(p))


def _norm(gamma_layer, beta_layer, norm: torch.nn.Module) -> None:
    """Write a norm's gamma (and beta). RMSNorm has no bias -> zero betas."""
    _encode_jax_logical(gamma_layer, _t(norm.weight))
    if getattr(norm, "bias", None) is not None:
        _encode_jax_logical(beta_layer, _t(norm.bias))
    else:
        _encode_jax_logical(beta_layer, np.zeros(norm.weight.shape[0], dtype=np.float32))


def _qkv_bias(layer, lin: torch.nn.Linear, out_features: int) -> None:
    """Export a qkv bias, writing zeros when remove_qkv_bias dropped it."""
    if lin.bias is not None:
        _encode_jax_logical(layer, _t(lin.bias))
    else:
        _encode_jax_logical(layer, np.zeros(out_features, dtype=np.float32))


# ---------------------------------------------------------------------------
# Model -> Weights
# ---------------------------------------------------------------------------
def _fill_weights(weights, model: LczeroModel) -> None:
    cfg = model.config

    # --- Embedding (INPUT_EMBEDDING_NONE: just the square embedding + gating) ---
    emb = model.embedding
    # embedding kernel with row-109/99 scaling on the (in,out) jax-logical array
    # (row 109 is the rule50 plane; lc0 feeds that plane as 99·value).
    jax_kernel = _t(emb.embedding.weight).T.copy()  # (in, out)
    jax_kernel[_EMBEDDING_PLANE_TO_SCALE, :] /= _EMBEDDING_SCALE
    _encode_jax_logical(weights.ip_emb_w, jax_kernel)
    _bias(weights.ip_emb_b, emb.embedding)

    if emb.ma_gating is not None:  # omitted => lc0 skips gating (optional layer).
        _vec(weights.ip_mult_gate, emb.ma_gating.mult_gate.gate)
        _vec(weights.ip_add_gate, emb.ma_gating.add_gate.gate)

    # --- Encoder tower ---
    weights.headcount = cfg.encoder.heads
    # Shared smolgen weight-gen dense (top-level, no bias).
    _linear_w(weights.smolgen_w, model.encoders.smolgen_weight_gen)

    d_model = cfg.encoder.d_model
    for blk in model.encoders.blocks:
        enc = weights.encoder.add()
        mha = blk.mha
        _linear_w(enc.mha.q_w, mha.q)
        _qkv_bias(enc.mha.q_b, mha.q, d_model)
        _linear_w(enc.mha.k_w, mha.k)
        _qkv_bias(enc.mha.k_b, mha.k, d_model)
        _linear_w(enc.mha.v_w, mha.v)
        _qkv_bias(enc.mha.v_b, mha.v, d_model)
        _linear_w(enc.mha.dense_w, mha.output_dense)
        _bias(enc.mha.dense_b, mha.output_dense)
        # smolgen submodule
        sm = mha.smolgen
        _linear_w(enc.mha.smolgen.compress, sm.compress)
        _linear_w(enc.mha.smolgen.dense1_w, sm.dense1)
        _bias(enc.mha.smolgen.dense1_b, sm.dense1)
        _norm(enc.mha.smolgen.ln1_gammas, enc.mha.smolgen.ln1_betas, sm.ln1)
        _linear_w(enc.mha.smolgen.dense2_w, sm.dense2)
        _bias(enc.mha.smolgen.dense2_b, sm.dense2)
        _norm(enc.mha.smolgen.ln2_gammas, enc.mha.smolgen.ln2_betas, sm.ln2)
        # block norms + ffn
        _norm(enc.ln1_gammas, enc.ln1_betas, blk.ln1)
        _linear_w(enc.ffn.dense1_w, blk.ffn.linear1)
        _bias(enc.ffn.dense1_b, blk.ffn.linear1)
        _linear_w(enc.ffn.dense2_w, blk.ffn.linear2)
        _bias(enc.ffn.dense2_b, blk.ffn.linear2)
        _norm(enc.ln2_gammas, enc.ln2_betas, blk.ln2)

    # --- Policy heads ---
    if model.policy_embedding_shared is not None:
        _linear_w(weights.policy_heads.ip_pol_w, model.policy_embedding_shared)
        _bias(weights.policy_heads.ip_pol_b, model.policy_embedding_shared)
    for name, head in model.policy_heads.items():
        if head.aux:  # aux policy heads are training-only (share the aux embedding)
            continue
        ph = getattr(weights.policy_heads, name)
        if not head.shared:
            _linear_w(ph.ip_pol_w, head.tokens)
            _bias(ph.ip_pol_b, head.tokens)
        _linear_w(ph.ip2_pol_w, head.q)
        _bias(ph.ip2_pol_b, head.q)
        _linear_w(ph.ip3_pol_w, head.k)
        _bias(ph.ip3_pol_b, head.k)
        _linear_w(ph.ip4_pol_w, head.promotion_dense)

    # --- Value heads ---
    for name, head in model.value_heads.items():
        vh = getattr(weights.value_heads, name)
        _linear_w(vh.ip_val_w, head.embed)
        _bias(vh.ip_val_b, head.embed)
        _linear_w(vh.ip1_val_w, head.dense1)
        _bias(vh.ip1_val_b, head.dense1)
        _linear_w(vh.ip2_val_w, head.wdl)
        _bias(vh.ip2_val_b, head.wdl)
        if head.error is not None:
            _linear_w(vh.ip_val_err_w, head.error)
            _bias(vh.ip_val_err_b, head.error)
        if head.categorical is not None:
            _linear_w(vh.ip_val_cat_w, head.categorical)
            _bias(vh.ip_val_cat_b, head.categorical)

    # --- Moves-left head (only "main", top-level) ---
    if "main" in model.movesleft_heads:
        ml = model.movesleft_heads["main"]
        _linear_w(weights.ip_mov_w, ml.embed)
        _bias(weights.ip_mov_b, ml.embed)
        _linear_w(weights.ip1_mov_w, ml.dense1)
        _bias(weights.ip1_mov_b, ml.dense1)
        _linear_w(weights.ip2_mov_w, ml.out)
        _bias(weights.ip2_mov_b, ml.out)


def _make_format(cfg: ModelConfig):
    fmt = net_pb2.Format()
    fmt.weights_encoding = net_pb2.Format.LINEAR16
    nf = fmt.network_format
    nf.input = net_pb2.NetworkFormat.INPUT_CLASSICAL_112_PLANE
    nf.output = net_pb2.NetworkFormat.OUTPUT_WDL
    nf.network = net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    nf.policy = net_pb2.NetworkFormat.POLICY_ATTENTION
    nf.value = net_pb2.NetworkFormat.VALUE_WDL
    nf.moves_left = net_pb2.NetworkFormat.MOVES_LEFT_V1
    nf.input_embedding = net_pb2.NetworkFormat.INPUT_EMBEDDING_NONE
    # default_activation: map config activation; lc0 expects MISH/RELU here.
    act = cfg.defaults.activation.lower()
    if act == "relu":
        nf.default_activation = net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU
    else:
        nf.default_activation = net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH
    nf.smolgen_activation = ACTIVATION_ENUM.get(
        (cfg.encoder.smolgen.activation or "swish").lower(),
        net_pb2.NetworkFormat.ACTIVATION_SWISH)
    ffn_act = cfg.defaults.ffn_activation.lower()
    if ffn_act == cfg.defaults.activation.lower():
        nf.ffn_activation = net_pb2.NetworkFormat.ACTIVATION_DEFAULT
    else:
        nf.ffn_activation = ACTIVATION_ENUM.get(ffn_act, net_pb2.NetworkFormat.ACTIVATION_DEFAULT)
    return fmt


def model_to_net(model: LczeroModel, min_version: str = "0.31.0",
                 training_steps: Optional[int] = None,
                 license_str: Optional[str] = None) -> "net_pb2.Net":
    net = net_pb2.Net()
    net.magic = MAGIC
    if license_str:
        net.license = license_str
    maj, minr, patch = (list(map(int, min_version.split("."))) + [0, 0, 0])[:3]
    net.min_version.major = maj
    net.min_version.minor = minr
    net.min_version.patch = patch
    net.format.CopyFrom(_make_format(model.config))
    if training_steps is not None:
        net.training_params.training_steps = int(training_steps)
    _fill_weights(net.weights, model)
    return net


def export_model(model: LczeroModel, path: str, min_version: str = "0.31.0",
                 training_steps: Optional[int] = None) -> None:
    model.eval()
    net = model_to_net(model, min_version=min_version, training_steps=training_steps)
    data = gzip.compress(net.SerializeToString())
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
