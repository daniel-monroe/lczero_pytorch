"""PyTorch port of the LczeroModel (BT-style attention network with smolgen).

Faithfully mirrors ../../src/lczero_training/model/*.py so that trained weights can
be exported to the Leela .pb.gz format and loaded by lc0. Adds two user-requested
flags: `remove_qkv_bias` and `rmsnorm_main_body`.
"""
from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .policy_map import POLICY_MAP
from .utils import get_activation, get_dtype, make_layernorm, make_norm

INPUT_CHANNELS = 112


# ---------------------------------------------------------------------------
# Initialization: replicate flax variance_scaling(scale, "fan_avg",
# "truncated_normal") used for DeepNorm-initialized layers, and lecun_normal
# (scale=1, fan_in) for the rest.
# ---------------------------------------------------------------------------
_TRUNC_STD_CORRECTION = 0.8796256610342398  # std of truncated normal in [-2, 2]


def _variance_scaling_(weight: torch.Tensor, scale: float, mode: str) -> None:
    # torch Linear weight: (out_features, in_features). fan_in=in, fan_out=out.
    fan_out, fan_in = weight.shape[0], weight.shape[1]
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    else:  # fan_avg
        denom = (fan_in + fan_out) / 2.0
    variance = scale / max(1.0, denom)
    std = math.sqrt(variance) / _TRUNC_STD_CORRECTION
    with torch.no_grad():
        weight.normal_(0.0, std).clamp_(-2.0 * std, 2.0 * std)


def _lecun_normal_(weight: torch.Tensor) -> None:
    _variance_scaling_(weight, 1.0, "fan_in")


def _deepnorm_linear(in_f: int, out_f: int, beta: float, bias: bool = True) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=bias)
    _variance_scaling_(lin.weight, beta, "fan_avg")
    if bias and lin.bias is not None:
        nn.init.zeros_(lin.bias)
    return lin


def _default_linear(in_f: int, out_f: int, bias: bool = True) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=bias)
    _lecun_normal_(lin.weight)
    if bias and lin.bias is not None:
        nn.init.zeros_(lin.bias)
    return lin


class ModelPrediction(NamedTuple):
    value: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]
    policy: Dict[str, torch.Tensor]
    movesleft: Dict[str, torch.Tensor]
    # name -> masked 1858-move policy logits[B,1858] for the auxiliary future heads.
    future: Dict[str, torch.Tensor] = {}
    # (B, 64, C) per-square logits for the optional dense board-aux head (V4-V7), else None.
    board: Optional[torch.Tensor] = None


class Ffn(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, activation: str, beta: float):
        super().__init__()
        self.linear1 = _deepnorm_linear(in_features, hidden_features, beta)
        self.linear2 = _deepnorm_linear(hidden_features, in_features, beta)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class Gating(nn.Module):
    def __init__(self, shape: Tuple[int, ...], additive: bool):
        super().__init__()
        self.additive = additive
        init_val = 0.0 if additive else 1.0
        self.gate = nn.Parameter(torch.full(shape, init_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.additive:
            return x + self.gate
        return x * F.relu(self.gate)


class MaGating(nn.Module):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.mult_gate = Gating(shape, additive=False)
        self.add_gate = Gating(shape, additive=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.add_gate(self.mult_gate(x))


class Embedding(nn.Module):
    """The simplest lc0-compatible input embedding (INPUT_EMBEDDING_NONE).

    No positional encoding, no preprocess, no LayerNorm, no FFN: just a square
    embedding Linear(112 -> embedding_size) + activation, with optional MaGating.
    (This is the daniel-monroe fork's "old" embedding style.)
    """

    def __init__(self, cfg: ModelConfig, deepnorm_alpha: float, deepnorm_beta: float):
        super().__init__()
        embedding_size = cfg.embedding.embedding_size
        self.act = get_activation(cfg.defaults.activation)
        self.embedding = _default_linear(INPUT_CHANNELS, embedding_size)
        self.ma_gating = MaGating((64, embedding_size)) if cfg.use_ma_gating else None
        # Learned absolute per-square positional bias (added to every token).
        self.abs_pos_bias = (nn.Parameter(torch.zeros(64, embedding_size))
                             if cfg.abs_pos_bias else None)
        # Learned per-(square, piece-type) embedding (64*12=768 vectors), added to the
        # occupied square's token. Captures square-specific piece encoding the SHARED
        # linear embedding cannot (a learned piece-square table).
        self.piece_square = (nn.Parameter(torch.zeros(64, 12, embedding_size))
                             if cfg.piece_square_embed else None)
        # Global board summary: nonlinearity from the 12 piece planes (12*64=768) to emb,
        # broadcast-added to all 64 tokens so the first attention layer has global context.
        if cfg.global_board_embed:
            self.global_proj = _default_linear(12 * 64, embedding_size)
            self.global_act = get_activation(cfg.defaults.activation)
        else:
            self.global_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, 112). First 12 channels per square = piece one-hots (6 ours + 6 theirs).
        b = x.shape[0]
        pieces = x[:, :, :12] if self.piece_square is not None else None   # (B,64,12) one-hots
        board768 = x[:, :, :12].reshape(b, 12 * 64) if self.global_proj is not None else None
        x = self.act(self.embedding(x))
        if self.abs_pos_bias is not None:
            x = x + self.abs_pos_bias
        if self.piece_square is not None:
            x = x + torch.einsum("bsc,sce->bse", pieces, self.piece_square)
        if self.global_proj is not None:
            g = self.global_act(self.global_proj(board768))    # (B, emb)
            x = x + g.unsqueeze(1)                              # broadcast over 64 tokens
        if self.ma_gating is not None:
            x = self.ma_gating(x)
        return x


class Smolgen(nn.Module):
    def __init__(self, cfg: ModelConfig, weight_gen_dense: nn.Linear):
        super().__init__()
        enc = cfg.encoder
        sm = enc.smolgen
        in_features = enc.d_model  # operates on encoder hidden (== embedding_size)
        self.heads = enc.heads
        self.gen_size = sm.gen_size
        self.compress = nn.Linear(in_features, sm.hidden_channels, bias=False)
        _lecun_normal_(self.compress.weight)
        self.dense1 = _default_linear(sm.hidden_channels * 64, sm.hidden_size)
        self.ln1 = make_layernorm(sm.hidden_size)  # smolgen ALWAYS LayerNorm
        self.dense2 = _default_linear(sm.hidden_size, sm.gen_size * enc.heads)
        self.ln2 = make_layernorm(sm.gen_size * enc.heads)
        self.weight_gen_dense = weight_gen_dense  # shared, (gen_size -> 64*64), no bias
        self.act = get_activation(sm.activation or cfg.defaults.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        c = self.compress(x).reshape(b, -1)
        h = self.act(self.dense1(c))
        h = self.ln1(h)
        g = self.act(self.dense2(h))
        g = self.ln2(g)
        g = g.reshape(b, self.heads, self.gen_size)
        out = self.weight_gen_dense(g)  # (b, heads, 4096)
        return out.reshape(b, self.heads, 64, 64)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, in_features: int, deepnorm_beta: float,
                 smolgen_weight_gen: Optional[nn.Linear]):
        super().__init__()
        enc = cfg.encoder
        depth = enc.d_model
        assert depth % enc.heads == 0
        self.depth = depth
        self.num_heads = enc.heads
        self.head_depth = depth // enc.heads
        qkv_bias = not cfg.remove_qkv_bias
        self.q = _default_linear(in_features, depth, bias=qkv_bias)
        self.k = _default_linear(in_features, depth, bias=qkv_bias)
        self.v = _deepnorm_linear(in_features, depth, deepnorm_beta, bias=qkv_bias)
        self.output_dense = _deepnorm_linear(depth, in_features, deepnorm_beta)
        self.smolgen = (Smolgen(cfg, smolgen_weight_gen)
                        if smolgen_weight_gen is not None else None)
        self.qk_norm = cfg.qk_norm
        self.attention_sink = cfg.attention_sink
        if self.qk_norm:  # RMSNorm on q,k per head (over head_depth)
            self.q_norm = make_norm(self.head_depth, True)
            self.k_norm = make_norm(self.head_depth, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        q = self.q(x).reshape(b, 64, self.num_heads, self.head_depth).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, 64, self.num_heads, self.head_depth).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, 64, self.num_heads, self.head_depth).permute(0, 2, 1, 3)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_depth)
        if self.smolgen is not None:
            logits = logits + self.smolgen(x)
        if self.attention_sink:
            # append one constant-0 "sink" key (value 0); softmax over 65, then drop it.
            sink = logits.new_zeros(*logits.shape[:-1], 1)
            attn = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)[..., :64]
        else:
            attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)  # (b, heads, 64, hd)
        out = out.permute(0, 2, 1, 3).reshape(b, 64, self.depth)
        return self.output_dense(out)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, in_features: int, num_blocks: int,
                 deepnorm_beta: float, smolgen_weight_gen: Optional[nn.Linear]):
        super().__init__()
        self.alpha = math.pow(2.0 * num_blocks, -0.25)
        self.mha = MultiHeadAttention(cfg, in_features, deepnorm_beta, smolgen_weight_gen)
        self.ln1 = make_norm(in_features, cfg.rmsnorm_main_body)
        self.ffn = Ffn(in_features, cfg.encoder.dff, cfg.defaults.ffn_activation, deepnorm_beta)
        self.ln2 = make_norm(in_features, cfg.rmsnorm_main_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(x) * self.alpha
        out1 = self.ln1(x)
        ffn_out = self.ffn(out1)
        return self.ln2(out1 + ffn_out * self.alpha)


class EncoderTower(nn.Module):
    def __init__(self, cfg: ModelConfig, in_features: int, deepnorm_beta: float):
        super().__init__()
        enc = cfg.encoder
        assert enc.smolgen is not None, "smolgen config is required"
        # Shared weight-generation dense, created once and shared across blocks.
        self.smolgen_weight_gen = nn.Linear(enc.smolgen.gen_size, 64 * 64, bias=False)
        _lecun_normal_(self.smolgen_weight_gen.weight)
        self.blocks = nn.ModuleList([
            EncoderBlock(cfg, in_features, enc.num_blocks, deepnorm_beta,
                         self.smolgen_weight_gen)
            for _ in range(enc.num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, cfg: ModelConfig, in_features: int, hcfg):
        super().__init__()
        self.act = get_activation(cfg.defaults.activation)
        self.has_error = hcfg.has_error_output
        self.num_buckets = hcfg.num_categorical_buckets
        self.embed = _default_linear(in_features, hcfg.num_channels)
        self.dense1 = _default_linear(hcfg.num_channels * 64, 128)
        self.wdl = _default_linear(128, 3)
        self.error = _default_linear(128, 1) if self.has_error else None
        self.categorical = (_default_linear(128, self.num_buckets)
                            if self.num_buckets > 0 else None)

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        x = self.embed(x).reshape(b, -1)
        x = self.act(x)
        x = self.act(self.dense1(x))
        wdl = self.wdl(x)
        error = torch.sigmoid(self.error(x)) if self.error is not None else None
        categorical = self.categorical(x) if self.categorical is not None else None
        return (wdl, error, categorical)


class MovesLeftHead(nn.Module):
    def __init__(self, cfg: ModelConfig, in_features: int, hcfg):
        super().__init__()
        self.act = get_activation(cfg.defaults.activation)
        self.embed = _default_linear(in_features, hcfg.num_channels)
        self.dense1 = _default_linear(hcfg.num_channels * 64, 128)
        self.out = _default_linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.act(self.embed(x).reshape(b, -1))
        x = self.act(self.dense1(x))
        return F.relu(self.out(x))


class FutureHead(nn.Module):
    """Auxiliary, training-only head predicting a future ply's move policy.

    Emits a full 1858-move policy from the shared aux policy embedding using the
    exact same machinery as an aux policy head (q/k attention + promotion + policy
    map), so it can be masked and trained with the same masked cross-entropy as the
    main policy head. Not exported to lc0.

    For opponent-perspective plies (`opponent=True`), the 64 square tokens are
    vertically mirrored (s -> s^56: rank-flipped, files preserved) before the qk,
    so the head's query/key squares still correspond to the opponent's pick-up/
    put-down squares and promotions stay on the 8th-rank tokens.
    """

    def __init__(self, cfg: ModelConfig, in_features: int, aux_depth: int,
                 opponent: bool = False):
        super().__init__()
        self.opponent = opponent
        # Reuse the aux policy-head path (runs entirely at aux_depth; hcfg unused
        # on the aux branch of PolicyHead).
        self.policy = PolicyHead(cfg, in_features, None,
                                 shared_embedding=None, aux=True, aux_depth=aux_depth)
        if opponent:
            # Vertical board mirror on the 64 square tokens: square rank*8+file ->
            # (7-rank)*8+file == s ^ 56. Files preserved so promotions stay aligned.
            self.register_buffer("flip",
                                 torch.arange(64).view(8, 8).flip(0).reshape(64),
                                 persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.opponent:
            x = x.index_select(1, self.flip)   # mirror square tokens (B,64,aux_depth)
        return self.policy(x)                  # (B, 1858)


class BoardAuxHead(nn.Module):
    """Dense per-square auxiliary head (V4-V7): a single Linear(emb -> C) applied to each
    of the 64 encoder tokens. C depends on the target kind (see BoardAuxConfig)."""

    _C = {"ownership": 3, "next_occ": 12, "dest": 70}

    def __init__(self, ba, in_features: int):
        super().__init__()
        self.kind = ba.kind
        self.horizon = ba.horizon
        c = 13 * ba.horizon if ba.kind == "dynamics" else self._C[ba.kind]
        self.channels = c
        self.out = _default_linear(in_features, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)  # (B, 64, C)


class PolicyHead(nn.Module):
    def __init__(self, cfg: ModelConfig, in_features: int, hcfg,
                 shared_embedding: Optional[nn.Linear], aux: bool = False,
                 aux_depth: Optional[int] = None):
        super().__init__()
        self.act = get_activation(cfg.defaults.activation)
        self.aux = aux
        if aux:
            # Pure projection off the shared aux policy embedding (already act'd):
            # no own token embedding; q/k/promo operate at aux_depth.
            self.tokens = None
            self.shared = False
            embedding_size = aux_depth
            d_model = aux_depth
        elif shared_embedding is not None:
            self.tokens = shared_embedding
            self.shared = True
            embedding_size = shared_embedding.out_features
            d_model = hcfg.d_model
        else:
            assert hcfg.embedding_size is not None
            self.tokens = _default_linear(in_features, hcfg.embedding_size)
            self.shared = False
            embedding_size = hcfg.embedding_size
            d_model = hcfg.d_model
        self.q = _default_linear(embedding_size, d_model)
        self.k = _default_linear(embedding_size, d_model)
        self.dk = math.sqrt(d_model)
        self.promotion_dense = nn.Linear(d_model, 4, bias=False)
        _lecun_normal_(self.promotion_dense.weight)
        self.register_buffer("policy_map",
                             torch.tensor(POLICY_MAP, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        if not self.aux:               # aux heads receive the shared embedding directly
            x = self.act(self.tokens(x))
        q = self.q(x)
        k = self.k(x)
        qk = torch.matmul(q, k.transpose(-1, -2))  # (b,64,64)

        promotion_keys = k[:, -8:, :]
        promotion_offsets = self.promotion_dense(promotion_keys)  # (b,8,4)
        promotion_offsets = promotion_offsets.transpose(1, 2) * self.dk  # (b,4,8)
        promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4, :]  # (b,3,8)

        n_promo = qk[:, -16:-8, -8:]  # (b,8,8)
        q_promo = (n_promo + promotion_offsets[:, 0:1, :]).unsqueeze(-1)
        r_promo = (n_promo + promotion_offsets[:, 1:2, :]).unsqueeze(-1)
        b_promo = (n_promo + promotion_offsets[:, 2:3, :]).unsqueeze(-1)
        promotion_logits = torch.cat([q_promo, r_promo, b_promo], dim=-1)  # (b,8,8,3)

        policy_attn = qk / self.dk  # (b,64,64)
        promotion_logits = promotion_logits.reshape(b, 8, 24) / self.dk
        logits = torch.cat([policy_attn.reshape(b, -1),
                            promotion_logits.reshape(b, -1)], dim=-1)  # (b,4288)
        return logits[:, self.policy_map]  # (b,1858)


class LczeroModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        n = cfg.encoder.num_blocks
        assert n > 0
        deepnorm_alpha = math.pow(2.0 * n, -0.25)
        deepnorm_beta = math.pow(8.0 * n, -0.25)
        emb_size = cfg.embedding.embedding_size

        self.embedding = Embedding(cfg, deepnorm_alpha, deepnorm_beta)
        self.encoders = EncoderTower(cfg, emb_size, deepnorm_beta)

        self.value_heads = nn.ModuleDict({
            h.name: ValueHead(cfg, emb_size, h) for h in cfg.value_head})

        if cfg.shared_policy_embedding_size is not None:
            self.policy_embedding_shared = _default_linear(
                emb_size, cfg.shared_policy_embedding_size)
        else:
            self.policy_embedding_shared = None

        # Shared auxiliary policy embedding: one act(Linear(emb->aux_depth)) that
        # the aux policy heads + future heads project from (pure projections).
        self.aux_depth = cfg.aux_policy_depth or emb_size
        has_aux = any(h.aux for h in cfg.policy_head) or len(cfg.future_head) > 0
        if has_aux:
            self.aux_policy_embed = _default_linear(emb_size, self.aux_depth)
            self.aux_act = get_activation(cfg.defaults.activation)
        else:
            self.aux_policy_embed = None

        self.policy_heads = nn.ModuleDict({
            h.name: PolicyHead(cfg, emb_size, h, self.policy_embedding_shared,
                               aux=h.aux, aux_depth=self.aux_depth)
            for h in cfg.policy_head})

        self.movesleft_heads = nn.ModuleDict({
            h.name: MovesLeftHead(cfg, emb_size, h) for h in cfg.movesleft_head})

        # Auxiliary, training-only future heads (masked 1858-move policies for
        # future plies). Opponent-perspective heads mirror their square tokens.
        self.future_heads = nn.ModuleDict({
            h.name: FutureHead(cfg, emb_size, self.aux_depth, opponent=h.opponent)
            for h in cfg.future_head})

        # Optional dense per-square board-aux head (V4-V7), off the encoder output.
        self.board_aux_head = (BoardAuxHead(cfg.board_aux, emb_size)
                               if cfg.board_aux is not None else None)

        self._compute_dtype = get_dtype(cfg.defaults.compute_dtype)

    def forward(self, x: torch.Tensor) -> ModelPrediction:
        # x: (B, 112, 8, 8). Match param dtype (float32); mixed precision is
        # handled by torch.autocast in the training loop, not by hard casting.
        x = x.to(self.embedding.embedding.weight.dtype)
        b = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(b, 64, INPUT_CHANNELS)
        x = self.embedding(x)
        x = self.encoders(x)
        # Shared aux policy embedding (computed once), fed to aux + future heads.
        aux = self.aux_act(self.aux_policy_embed(x)) if self.aux_policy_embed is not None else None
        value = {name: head(x) for name, head in self.value_heads.items()}
        policy = {name: head(aux if head.aux else x)
                  for name, head in self.policy_heads.items()}
        movesleft = {name: head(x) for name, head in self.movesleft_heads.items()}
        future = {name: head(aux) for name, head in self.future_heads.items()}
        board = self.board_aux_head(x) if self.board_aux_head is not None else None
        return ModelPrediction(value=value, policy=policy,
                               movesleft=movesleft, future=future, board=board)
