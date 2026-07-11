"""Fast multiprocess V7 chunk data loader.

Pure PyTorch / NumPy — no dependency on the C++ loader, TensorFlow or JAX. Each
DataLoader worker streams gzipped V7 chunks, samples positions, runs a streaming
reservoir shuffle, and yields float32 (planes, probs, values) tuples that the
default collate stacks into batches.

Throughput scales with `num_workers`; on the 8×A100 box this comfortably feeds
even large transformers since the per-record work is vectorized NumPy.
"""
from __future__ import annotations

import gzip
import math
import os
import random
import struct
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

V7_RECORD_SIZE = 8396
NUM_PLANES = 112

# Bit-unpacking: plane[square] = (u64 >> (square ^ 7)) & 1.
_SHIFTS = (np.arange(64, dtype=np.uint64) ^ np.uint64(7))

# Byte offsets within a V7 record.
_OFF_PROBS = 8       # 1858 float32
_OFF_PLANES = 7440   # 104 uint64
_OFF_CASTLE = 8272   # 4 bytes castling + stm + rule50 + invariance + dep_result
_OFF_FLOATS = 8280   # 15 float32: root_q best_q root_d best_d root_m best_m plies
                     #             result_q result_d played_q played_d played_m
                     #             orig_q orig_d orig_m
_OFF_BEST_Q = 8284   # float32 best_q  (post-search value)   -- for Leela diff focus
_OFF_ORIG_Q = 8328   # float32 orig_q  (original network value)
_OFF_POL_KLD = 8348  # float32 policy KLD (original vs searched policy)
_OFF_QST = 8352
_OFF_DST = 8356


def record_probs(rec: bytes) -> np.ndarray:
    """Just the 1858-move policy (visit) distribution of a record (negatives=illegal)."""
    return np.frombuffer(rec, dtype="<f4", count=1858, offset=_OFF_PROBS).copy()


def parse_record(rec: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one 8396-byte V7 record into (planes[112,8,8], probs[1858], values[6,3])."""
    planes = np.empty((NUM_PLANES, 64), dtype=np.float32)
    u64 = np.frombuffer(rec, dtype="<u8", count=104, offset=_OFF_PLANES)
    planes[:104] = ((u64[:, None] >> _SHIFTS[None, :]) & np.uint64(1)).astype(np.float32)
    planes[104] = rec[_OFF_CASTLE + 0]   # castling_us_ooo
    planes[105] = rec[_OFF_CASTLE + 1]   # castling_us_oo
    planes[106] = rec[_OFF_CASTLE + 2]   # castling_them_ooo
    planes[107] = rec[_OFF_CASTLE + 3]   # castling_them_oo
    planes[108] = rec[_OFF_CASTLE + 4]   # side_to_move_or_enpassant
    planes[109] = rec[_OFF_CASTLE + 5] / 99.0  # rule50_count / 99
    planes[110] = 0.0
    planes[111] = 1.0
    planes = planes.reshape(NUM_PLANES, 8, 8)

    probs = np.frombuffer(rec, dtype="<f4", count=1858, offset=_OFF_PROBS).copy()

    f = np.frombuffer(rec, dtype="<f4", count=15, offset=_OFF_FLOATS)
    root_q, best_q, root_d, best_d, root_m, best_m, plies, \
        result_q, result_d, played_q, played_d, played_m, \
        orig_q, orig_d, orig_m = f
    q_st = np.frombuffer(rec, dtype="<f4", count=1, offset=_OFF_QST)[0]
    d_st = np.frombuffer(rec, dtype="<f4", count=1, offset=_OFF_DST)[0]
    values = np.array([
        [result_q, result_d, plies],
        [best_q, best_d, best_m],
        [played_q, played_d, played_m],
        [orig_q, orig_d, orig_m],
        [root_q, root_d, root_m],
        [q_st, d_st, np.nan],
    ], dtype=np.float32)
    return planes, probs, values


def _read_chunk_records(path: str) -> List[bytes]:
    """Decompress a chunk file and split into V7 records. Returns [] on any error."""
    try:
        with gzip.open(path, "rb") as fh:
            data = fh.read()
    except Exception:
        return []
    if len(data) < 4:
        return []
    version = int(np.frombuffer(data, dtype="<i4", count=1)[0])
    if version != 7 or (len(data) % V7_RECORD_SIZE) != 0:
        return []
    n = len(data) // V7_RECORD_SIZE
    return [data[i * V7_RECORD_SIZE:(i + 1) * V7_RECORD_SIZE] for i in range(n)]


def discover_source_dirs(input_dirs: List[str]) -> List[str]:
    """Find 'chunk source' directories cheaply, without enumerating every file.

    This dataset has ~25k .gz files in each of ~2770 subdirectories (tens of
    millions of files total), so we never build a global file list. Instead each
    immediate subdirectory of an input dir is treated as a chunk source; an input
    dir that itself holds .gz files is also a source. Workers list files lazily.
    """
    sources: List[str] = []
    for root in input_dirs:
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        has_gz_here = False
        for e in entries:
            full = os.path.join(root, e)
            if os.path.isdir(full):
                sources.append(full)
            elif e.endswith(".gz"):
                has_gz_here = True
        if has_gz_here:
            sources.append(root)
    if not sources:
        sources = list(input_dirs)
    return sources


def _stable_hash(name: str) -> float:
    import hashlib
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _split_dirs(dirs: List[str], train_ratio: float, train: bool) -> List[str]:
    # Deterministic split by hash of the source-dir basename. Identical across
    # workers/runs and non-overlapping between train and test.
    return [d for d in dirs
            if (_stable_hash(os.path.basename(d.rstrip("/"))) < train_ratio) == train]


def _list_gz(dirpath: str) -> List[str]:
    try:
        return [os.path.join(dirpath, f) for f in os.listdir(dirpath)
                if f.endswith(".gz")]
    except OSError:
        return []


class V7IterableDataset(IterableDataset):
    """Streams positions from a set of chunk-source directories.

    Memory-bounded: lists files one source-dir at a time, into a frame reservoir
    of `frame_shuffle_size`.
    """

    def __init__(self, source_dirs: List[str], positions_per_chunk: int,
                 frame_shuffle_size: int,
                 skip_factor: int, infinite: bool = True, seed: int = 0,
                 future_plies: int = 2,
                 diff_focus: Optional[Tuple[float, float, float, float]] = None,
                 board_aux: Optional[Tuple[str, int]] = None):
        super().__init__()
        self.source_dirs = source_dirs
        self.positions_per_chunk = positions_per_chunk
        self.frame_shuffle_size = max(1, frame_shuffle_size)
        self.skip_factor = max(1, skip_factor)
        self.infinite = infinite
        self.seed = seed
        self.future_plies = max(1, future_plies)
        # (diff_focus_min, diff_focus_slope, q_weight, pol_scale) or None (disabled).
        self.diff_focus = diff_focus
        # (kind, horizon) for the dense future-board aux target, or None.
        self.board_aux = board_aux

    def _worker_dirs(self) -> List[str]:
        info = get_worker_info()
        if info is None:
            return list(self.source_dirs)
        return self.source_dirs[info.id::info.num_workers]

    def _file_stream(self, dirs: List[str], rng: random.Random) -> Iterator[str]:
        """Yield shuffled chunk files lazily: shuffle dirs, then files within each
        dir, holding only one dir's listing at a time."""
        while True:
            rng.shuffle(dirs)
            for d in dirs:
                files = _list_gz(d)
                rng.shuffle(files)
                for f in files:
                    yield f
            if not self.infinite:
                return

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = random.Random(self.seed * 100003 + wid)
        dirs = self._worker_dirs()
        if not dirs:
            return

        buf: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        cap = self.frame_shuffle_size

        def sample_positions(records: List[bytes]):
            if not records:
                return
            n = len(records)
            pool = list(range(0, n, self.skip_factor)) if self.skip_factor > 1 else list(range(n))
            k = min(self.positions_per_chunk, len(pool))
            for i in rng.sample(pool, k):
                # Leela "diff focus": upweight positions the original net got wrong
                # by rejection-sampling (peek at best_q/orig_q/pol_kld, skip records
                # the net already predicted well). Matches lc0 chunkparser exactly.
                if self.diff_focus is not None:
                    rec = records[i]
                    best_q = struct.unpack_from("<f", rec, _OFF_BEST_Q)[0]
                    orig_q = struct.unpack_from("<f", rec, _OFF_ORIG_Q)[0]
                    pol_kld = struct.unpack_from("<f", rec, _OFF_POL_KLD)[0]
                    if not math.isnan(orig_q) and pol_kld > 0.0:
                        dmin, dslope, qw, ps = self.diff_focus
                        diff_q = abs(best_q - orig_q)
                        total = (qw * diff_q + pol_kld) / (qw + ps)
                        thresh_p = dmin + dslope * total
                        if thresh_p < 1.0 and rng.random() > thresh_p:
                            continue  # skip: original net was already ~correct here
                planes, probs, values = parse_record(records[i])
                # Future-head targets are the full move DISTRIBUTIONS of the next
                # `future_plies` plies of this game (records are sequential):
                # row k = ply i+1+k (row 0 = opponent reply i+1, row 1 = our next
                # move i+2, row 2 = i+3, row 3 = i+4, ...). Missing (game end) ->
                # all -1 (treated as fully illegal -> masked in the loss).
                future = np.full((self.future_plies, 1858), -1.0, dtype=np.float32)
                for k in range(self.future_plies):
                    if i + 1 + k < n:
                        future[k] = record_probs(records[i + 1 + k])
                if self.board_aux is not None:
                    board = _board_target(records, i, self.board_aux[0], self.board_aux[1])
                else:
                    board = np.zeros(1, dtype=np.int64)   # placeholder (unused)
                yield planes, probs, values, future, board

        warmup = max(1, cap // 4)
        for path in self._file_stream(dirs, rng):
            records = _read_chunk_records(path)
            for sample in sample_positions(records):
                if len(buf) < cap:
                    buf.append(sample)
                    # Start emitting once warm so the GPU is never starved during the
                    # long initial fill; net growth (+0.5/iter) still reaches cap.
                    if len(buf) > warmup and rng.random() < 0.5:
                        j = rng.randrange(len(buf))
                        buf[j], buf[-1] = buf[-1], buf[j]
                        yield buf.pop()
                else:
                    j = rng.randrange(cap)
                    out = buf[j]
                    buf[j] = sample
                    yield out
        # finite (eval) mode: drain the reservoir
        if not self.infinite:
            rng.shuffle(buf)
            for s in buf:
                yield s


# --- Dense future-board auxiliary targets (V4-V7) -------------------------------------
# All targets are built in the CURRENT player's frame: future ply i+k is mapped back by
# a vertical mirror (s^56) + us/them color swap when k is odd. 13-class piece code:
# 0=empty, 1..6 = ours {P,N,B,R,Q,K}, 7..12 = theirs. Masked entries use -100 (ignored by
# cross-entropy). Validated: consecutive plies differ by exactly one move once mapped.

def _occ12(rec: bytes) -> np.ndarray:
    """Lean unpack of just the 12 current-position piece planes -> (12, 64) float."""
    u64 = np.frombuffer(rec, dtype="<u8", count=12, offset=_OFF_PLANES)
    return ((u64[:, None] >> _SHIFTS[None, :]) & np.uint64(1)).astype(np.float32)


def _occ13(rec: bytes, odd: bool) -> np.ndarray:
    """13-class occupancy per square in the CURRENT player's frame."""
    p = _occ12(rec)
    occ = np.where(p.any(0), p.argmax(0) + 1, 0).astype(np.int64)
    if odd:
        occ = occ.reshape(8, 8)[::-1, :].reshape(64)          # s^56 vertical mirror
        ours = (occ >= 1) & (occ <= 6)
        theirs = (occ >= 7) & (occ <= 12)
        swapped = occ.copy()
        swapped[ours] = occ[ours] + 6
        swapped[theirs] = occ[theirs] - 6
        occ = swapped
    return occ


def _board_target(records: List[bytes], i: int, kind: str, horizon: int) -> np.ndarray:
    n = len(records)
    if kind == "ownership":                       # V4: 3-way {ours=0, theirs=1, empty=2} at i+H
        H = horizon
        if i + H >= n:
            return np.full(64, -100, dtype=np.int64)
        occ = _occ13(records[i + H], (H % 2 == 1))
        return np.where(occ == 0, 2, np.where(occ <= 6, 0, 1)).astype(np.int64)

    if kind == "dynamics":                        # V7: 13-way occupancy at each of i+1..i+K
        K = horizon
        tgt = np.full((K, 64), -100, dtype=np.int64)
        for k in range(1, K + 1):
            if i + k < n:
                tgt[k - 1] = _occ13(records[i + k], (k % 2 == 1))
        return tgt

    cap = horizon if horizon > 0 else n
    if kind == "next_occ":                        # V6: 12-way id of next distinct occupant
        cur = _occ13(records[i], False)
        tgt = np.full(64, -100, dtype=np.int64)
        done = np.zeros(64, dtype=bool)
        for k in range(1, cap + 1):
            if i + k >= n:
                break
            occ = _occ13(records[i + k], (k % 2 == 1))
            new = (~done) & (occ != 0) & (occ != cur)
            tgt[new] = occ[new] - 1                # 0..11
            done |= new
            if done.all():
                break
        return tgt

    if kind == "dest":                            # V5: per OUR occupied square -> next-move
        occs = [_occ13(records[i], False)]        # to-square (0..63), or 64+type if captured
        for k in range(1, cap + 1):
            if i + k >= n:
                break
            occs.append(_occ13(records[i + k], (k % 2 == 1)))
        cur = occs[0]
        tgt = np.full(64, -100, dtype=np.int64)
        for s in np.where((cur >= 1) & (cur <= 6))[0]:   # our (side-to-move) pieces
            for k in range(len(occs) - 1):
                a, b = occs[k], occs[k + 1]
                if not (1 <= a[s] <= 6):
                    break                          # piece already gone (shouldn't precede event)
                if b[s] == 0 and k % 2 == 0:       # our move (even k) vacated s -> find dest
                    cand = np.where((b >= 1) & (b <= 6) & ~((a >= 1) & (a <= 6)))[0]
                    if len(cand) == 1:
                        tgt[s] = int(cand[0])
                    break
                if 7 <= b[s] <= 12 and k % 2 == 1: # captured by opponent (odd k)
                    tgt[s] = 64 + int(b[s] - 7)
                    break
                if b[s] != a[s]:
                    break
        return tgt
    raise ValueError(f"unknown board_aux kind {kind!r}")


def _collate(batch):
    planes = torch.from_numpy(np.stack([b[0] for b in batch]))
    probs = torch.from_numpy(np.stack([b[1] for b in batch]))
    values = torch.from_numpy(np.stack([b[2] for b in batch]))
    future = torch.from_numpy(np.stack([b[3] for b in batch]))  # (B, future_plies, 1858)
    board = torch.from_numpy(np.stack([b[4] for b in batch]))    # (B, ...) board-aux target
    return planes, probs, values, future, board


def make_dataloader(cfg, batch_size: int, train: bool = True, seed: int = 0,
                    infinite: bool = True, diff_focus: bool = False,
                    board_aux: Optional[Tuple[str, int]] = None) -> DataLoader:
    """Build a training (infinite) or eval DataLoader from a DataConfig.

    `diff_focus=True` enables Leela's diff-focus rejection sampling (only meaningful
    for the training stream); it is a no-op unless cfg.diff_focus_slope > 0.
    `board_aux=(kind, horizon)` makes the sampler emit the dense future-board target.
    """
    sources = discover_source_dirs(cfg.input_dirs)
    if not sources:
        raise RuntimeError(f"No chunk source directories found under {cfg.input_dirs}")
    sources = _split_dirs(sources, cfg.train_ratio, train)
    if not sources:
        raise RuntimeError("Train/test split produced no source dirs; check train_ratio")
    ds = V7IterableDataset(
        source_dirs=sources,
        positions_per_chunk=cfg.positions_per_chunk,
        frame_shuffle_size=cfg.frame_shuffle_size,
        skip_factor=cfg.skip_factor,
        infinite=infinite,
        seed=seed,
        future_plies=getattr(cfg, "future_plies", 2),
        diff_focus=(
            (cfg.diff_focus_min, cfg.diff_focus_slope,
             cfg.diff_focus_q_weight, cfg.diff_focus_pol_scale)
            if diff_focus and getattr(cfg, "diff_focus_slope", 0.0) > 0.0
            else None
        ),
        board_aux=board_aux,
    )
    num_workers = cfg.num_workers
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0 and infinite),
        prefetch_factor=(cfg.prefetch_factor if num_workers > 0 else None),
    )
