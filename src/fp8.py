"""FP8 training support for Blackwell / Hopper / Ada tensor cores.

Thin wrapper over ``torchao`` that swaps the network's body ``nn.Linear`` GEMMs in
place for an FP8 linear. Two families of recipe are supported:

  * **Scaled FP8** (``tensorwise`` / ``rowwise``) via ``torchao.float8`` --
    ``Float8Linear`` casts inputs+weights to e4m3 with a per-tensor or per-row
    scale. Runs on Ada (8.9) / Hopper (9.0) / Blackwell (10.0+).
  * **Microscaling FP8** (``mxfp8``) via ``torchao.prototype.mx_formats`` --
    ``MXLinear`` uses an e8m0 scale shared across each block of ``mx_block_size``
    (32) elements. This is the format **Blackwell** tensor cores accelerate
    natively; the hardware GEMM needs cc >= 10.0 (or ``mx_gemm_kernel="emulated"``).

In both cases the swapped module is an ``nn.Linear`` subclass that keeps a
*high-precision* master ``weight`` and preserves the ``weight``/``bias`` parameter
names, so the change is transparent to the optimizer (same FQNs -> masked-WD globs
still match), checkpointing (same state_dict keys) and lc0 export (``export.py``
reads the high-precision ``.weight``). FP8 is purely a compute speedup: the
high-precision master weights are what gets trained, checkpointed and exported, so
an FP8-trained net is identical in format to a bf16-trained one.

Requirements: an FP8-capable GPU and the ``torchao`` package. Keep
``model.defaults.compute_dtype: bfloat16`` (FP8 runs under bf16 autocast) and
``torch.compile`` on (FP8 needs the scaling/cast fusion compile provides to beat
bf16).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

# Min CUDA compute capability for scaled-FP8 matmul (Ada Lovelace) ...
_MIN_FP8_CAPABILITY: Tuple[int, int] = (8, 9)
# ... and for hardware microscaling-FP8 (mxfp8) matmul (Blackwell).
_MIN_MXFP8_CAPABILITY: Tuple[int, int] = (10, 0)

_SCALED_RECIPES = ("tensorwise", "rowwise")
_MX_RECIPES = ("mxfp8",)


def _capability(device: torch.device) -> Tuple[int, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(device.index)


def fp8_capability_ok(device: torch.device) -> bool:
    """True iff `device` is a CUDA GPU with scaled-FP8 tensor cores (cc >= 8.9)."""
    cap = _capability(device)
    return cap is not None and cap >= _MIN_FP8_CAPABILITY


def _make_filter(skip_fqns: List[str], block_size: int = 1):
    """Build a torchao `filter_fn(module, fqn) -> bool` (True => convert).

    Converts an `nn.Linear` only when (a) no entry of `skip_fqns` is a substring of
    its FQN and (b) its in/out features are both divisible by `align` (16 for scaled
    FP8; the MX block size for mxfp8). Everything else stays in high precision.
    """
    align = max(16, block_size)

    def keep_in_fp8(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if any(s and s in fqn for s in skip_fqns):
            return False
        if mod.in_features % align != 0 or mod.out_features % align != 0:
            return False
        return True

    return keep_in_fp8


# ---------------------------------------------------------------------------
# Scaled FP8 (tensorwise / rowwise) -- torchao.float8
# ---------------------------------------------------------------------------
def _convert_scaled_fp8(model: nn.Module, cfg, filter_fn) -> None:
    try:
        from torchao.float8 import (Float8LinearConfig,
                                    convert_to_float8_training)
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "training.fp8 (tensorwise/rowwise) requires the 'torchao' package "
            "(pip install torchao). Original import error: " + repr(e))

    recipe = cfg.recipe.lower()
    if recipe == "tensorwise" or not hasattr(Float8LinearConfig, "from_recipe_name"):
        if recipe == "rowwise":
            raise RuntimeError(
                "fp8.recipe='rowwise' requires a newer torchao with "
                "Float8LinearConfig.from_recipe_name; upgrade torchao or use "
                "recipe='tensorwise'.")
        float8_config = Float8LinearConfig()
    else:
        float8_config = Float8LinearConfig.from_recipe_name(recipe)

    convert_to_float8_training(
        model, config=float8_config, module_filter_fn=filter_fn)


# ---------------------------------------------------------------------------
# Microscaling FP8 (mxfp8) -- torchao.prototype.mx_formats
# ---------------------------------------------------------------------------
def _mx_config(cfg):
    """Build a torchao MXLinearConfig for the requested mxfp8 kernel."""
    kernel = (cfg.mx_gemm_kernel or "auto").lower()
    valid_kernels = {"auto", "cublas", "cutlass", "emulated"}
    if kernel not in valid_kernels:
        raise ValueError(
            f"Unknown fp8.mx_gemm_kernel '{cfg.mx_gemm_kernel}', expected one of "
            f"{valid_kernels}")

    try:
        from torchao.prototype.mx_formats import MXLinearConfig
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "fp8.recipe='mxfp8' requires a torchao with the mx_formats prototype "
            "(pip install -U torchao). Original import error: " + repr(e))

    # Prefer the recipe-name constructor (encodes dtype + kernel defaults; the MX
    # spec fixes block_size=32). Recipe names: mxfp8_cublas/cutlass/emulated.
    if hasattr(MXLinearConfig, "from_recipe_name"):
        name = {"auto": "mxfp8_cublas", "cublas": "mxfp8_cublas",
                "cutlass": "mxfp8_cutlass", "emulated": "mxfp8_emulated"}[kernel]
        try:
            return MXLinearConfig.from_recipe_name(name)
        except Exception:
            pass  # fall through to explicit construction below

    # Explicit construction fallback (older torchao).
    from torchao.prototype.mx_formats import MXGemmKernelChoice
    choice_name = {"auto": "CUBLAS", "cublas": "CUBLAS",
                   "cutlass": "CUTLASS", "emulated": "EMULATED"}[kernel]
    gemm = getattr(MXGemmKernelChoice, choice_name)
    return MXLinearConfig(block_size=cfg.mx_block_size,
                          elem_dtype=torch.float8_e4m3fn,
                          gemm_kernel_choice=gemm)


def _convert_mxfp8(model: nn.Module, cfg, filter_fn) -> None:
    mx_config = _mx_config(cfg)
    # Modern torchao swaps Linear -> MXLinear via the unified quantize_ API; older
    # releases expose a dedicated swap_linear_with_mx_linear helper.
    try:
        from torchao.quantization import quantize_
        quantize_(model, mx_config, filter_fn=filter_fn)
    except (ImportError, TypeError):
        from torchao.prototype.mx_formats.mx_linear import \
            swap_linear_with_mx_linear
        swap_linear_with_mx_linear(model, config=mx_config, filter_fn=filter_fn)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def convert_model_to_fp8(model: nn.Module, cfg, device: torch.device) -> int:
    """Swap eligible body `nn.Linear` layers in `model` for an FP8 linear.

    Mutates `model` in place. Returns the number of converted layers. Raises a clear
    error if the recipe is unknown, the hardware lacks the required FP8 support, or
    `torchao` is not installed -- enabling FP8 is an explicit opt-in, so an
    unsupported environment is a configuration error, not a silent downgrade.
    """
    recipe = (cfg.recipe or "tensorwise").lower()
    if recipe not in _SCALED_RECIPES + _MX_RECIPES:
        raise ValueError(
            f"Unknown fp8.recipe '{cfg.recipe}', expected one of "
            f"{_SCALED_RECIPES + _MX_RECIPES}")

    is_mx = recipe in _MX_RECIPES
    emulated = is_mx and (cfg.mx_gemm_kernel or "auto").lower() == "emulated"
    min_cap = _MIN_MXFP8_CAPABILITY if is_mx else _MIN_FP8_CAPABILITY
    cap = _capability(device)
    # Emulated mxfp8 is a bit-accurate software reference -- it runs on any CUDA GPU
    # (for correctness testing off-Blackwell), so it only needs a CUDA device.
    cap_ok = cap is not None and (emulated or cap >= min_cap)
    if not cap_ok:
        kind = "mxfp8 hardware" if is_mx else "FP8"
        hw = "Blackwell" if is_mx else "Ada/Hopper/Blackwell"
        raise RuntimeError(
            f"training.fp8.enabled with recipe='{recipe}' needs a {kind}-capable "
            f"GPU (CUDA compute capability >= {min_cap}, got {cap}); requires {hw}. "
            + ("Use mx_gemm_kernel='emulated' to test off-Blackwell, or "
               if is_mx else "")
            + "disable fp8 or run on a supported GPU.")

    block = cfg.mx_block_size if is_mx else 1
    filter_fn = _make_filter(cfg.skip_fqns, block_size=block)

    before = _count_linears(model)
    if is_mx:
        _convert_mxfp8(model, cfg, filter_fn)
    else:
        _convert_scaled_fp8(model, cfg, filter_fn)
    converted = _count_converted(model)
    kept = before - converted
    detail = recipe + (f"/{cfg.mx_gemm_kernel}" if is_mx else "")
    print(f"[fp8] recipe={detail} converted {converted}/{before} body Linear "
          f"layers ({kept} kept in high precision)", flush=True)
    return converted


def _count_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.Linear))


def _count_converted(model: nn.Module) -> int:
    """Count linears swapped to an FP8 variant (Float8Linear / MXLinear).

    Both are nn.Linear subclasses; match by class name to avoid a hard torchao
    import here.
    """
    fp8_types = {"Float8Linear", "MXLinear"}
    return sum(1 for m in model.modules()
               if isinstance(m, nn.Linear) and type(m).__name__ in fp8_types)
