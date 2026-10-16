"""Video pipeline — reconstructed support module.

NOTE: This file was ABSENT from the uploaded snapshot. Reconstructed to
satisfy the seam contract pinned by tests/test_video_conceptgraph_seam.py:

  * VideoPipelineConfig(gate=..., min_area=...)
  * NumpySequenceSource(frames, fps)
  * generate_synthetic_frames(num_frames, size, seed) — one persistently
    moving bright sprite on a constant dark background (deterministic).
  * frames_from_sprites(...) — composited sprite clip (used only when the
    sprite images exist in uploads/; they do not in this snapshot).
  * VideoPipeline(source, cfg) — context manager; process_all() ->
    VideoSnapshot with persistent track identities.
  * TemporalType — observation contract enum.

The pipeline is real, if minimal: per-frame foreground extraction by
intensity gate, 4-connected-component labelling, area filter, and
greedy nearest-centroid track association with a per-frame jump cap.
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class TemporalType(Enum):
    """What kind of temporal entity a track-level observation describes."""

    SPATIAL_OBJECT = auto()
    EVENT_CAUSAL = auto()
    TEMPORAL_CHAIN = auto()


# ---------------------------------------------------------------------------
# Frame generators
# ---------------------------------------------------------------------------

def generate_synthetic_frames(num_frames: int = 30, size: int = 64, seed: int = 0) -> np.ndarray:
    """One bright square traversing a constant dark background.

    Deterministic by construction (the seed perturbs only the phase of the
    trajectory, never the per-frame rendering), so a given (num_frames,
    size, seed) triple renders identically everywhere.
    """
    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0.0, 1.0))
    frames = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
    frames[:] = 10  # constant dark background (below any positive gate)
    side = max(6, size // 6)
    amp = (size - side) // 2 - 2
    cx = cy = (size - side) // 2
    for f in range(num_frames):
        theta = 2.0 * math.pi * (f / max(1, num_frames)) + phase * 2.0 * math.pi
        x = int(round(cx + amp * math.cos(theta)))
        y = int(round(cy + amp * math.sin(theta)))
        frames[f, y : y + side, x : x + side, :] = 235
    return frames


def frames_from_sprites(
    sprite_paths: Sequence[str],
    num_frames: int = 60,
    size: int = 96,
    seed: int = 7,
    static_path: Optional[str] = None,
) -> np.ndarray:
    """Composite real sprite images onto lane-separated trajectories.

    Each moving sprite slides horizontally across its own lane; the static
    sprite pulses (intensity-modulated) in the bottom corner. Deterministic
    given the inputs.
    """
    from PIL import Image

    frames = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
    frames[:] = 10
    box = max(6, size // 6)
    sprites: List[np.ndarray] = []
    for p in sprite_paths:
        img = Image.open(p).convert("RGB").resize((box, box))
        sprites.append(np.array(img, dtype=np.uint8))
    lanes = [size // 8, size // 2 - box // 2]
    for i, spr in enumerate(sprites):
        start_f = 0 if i == 0 else num_frames // 4
        y = lanes[i % len(lanes)]
        for f in range(start_f, num_frames):
            u = (f - start_f) / max(1, num_frames - start_f - 1)
            x = int(round(u * (size - box)))
            frames[f, y : y + box, x : x + box, :] = spr
    if static_path:
        img = Image.open(static_path).convert("RGB").resize((box, box))
        sta = np.array(img, dtype=np.uint8)
        y = size - box - 2
        x = size - box - 2
        for f in range(num_frames):
            gain = 1.0 if (f // 5) % 2 == 0 else 0.55
            frames[f, y : y + box, x : x + box, :] = (sta * gain).astype(np.uint8)
    return frames


# ---------------------------------------------------------------------------
# Source + config
# ---------------------------------------------------------------------------

class NumpySequenceSource:
    def __init__(self, frames: np.ndarray, fps: float = 15.0) -> None:
        self.frames = frames
        self.fps = fps

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield self.frames[i]


@dataclass
class VideoPipelineConfig:
    gate: float = 60.0          # foreground intensity gate (mean over RGB)
    min_area: int = 6           # minimum component area in pixels
    max_jump: float = 16.0      # max centroid displacement per frame (px)
    min_track_len: int = 5      # frames required for a track to be "active"


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def _components(mask: np.ndarray, min_area: int) -> List[Dict[str, Any]]:
    """4-connected-component labelling (iterative flood fill)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: List[Dict[str, Any]] = []
    for y0 in range(H):
        for x0 in range(W):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            q = deque([(y0, x0)])
            seen[y0, x0] = True
            ys: List[int] = []
            xs: List[int] = []
            while q:
                y, x = q.popleft()
                ys.append(y)
                xs.append(x)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            area = len(ys)
            if area < min_area:
                continue
            out.append(
                {
                    "area": area,
                    "centroid": (float(np.mean(xs)), float(np.mean(ys))),
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "pixels": (ys, xs),
                }
            )
    return out


@dataclass
class TrackObservation:
    frame: int
    centroid: Tuple[float, float]
    area: int
    mean_rgb: Tuple[float, float, float]


@dataclass
class Track:
    track_id: int
    observations: List[TrackObservation] = field(default_factory=list)
    last_centroid: Tuple[float, float] = (0.0, 0.0)


@dataclass
class VideoSnapshot:
    num_frames: int
    fps: float
    tracks: List[Track]

    @property
    def active_track_ids(self) -> List[int]:
        return [t.track_id for t in self.tracks]


class VideoPipeline:
    def __init__(self, source: NumpySequenceSource, config: Optional[VideoPipelineConfig] = None) -> None:
        self.source = source
        self.config = config or VideoPipelineConfig()
        self._next_track_id = 1

    def __enter__(self) -> "VideoPipeline":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    # --
    def process_all(self) -> VideoSnapshot:
        cfg = self.config
        live: List[Track] = []
        finished: List[Track] = []
        n_frames = 0
        for f_idx, frame in enumerate(self.source):
            n_frames += 1
            gray = frame[..., :3].mean(axis=-1)
            mask = gray > cfg.gate
            comps = _components(mask, cfg.min_area)
            dets = []
            for c in comps:
                ys, xs = c["pixels"]
                mean_rgb = (
                    float(np.mean(frame[ys, xs, 0])),
                    float(np.mean(frame[ys, xs, 1])),
                    float(np.mean(frame[ys, xs, 2])),
                )
                dets.append(
                    TrackObservation(
                        frame=f_idx,
                        centroid=c["centroid"],
                        area=c["area"],
                        mean_rgb=mean_rgb,
                    )
                )
            # Greedy nearest-centroid association.
            assigned_tracks: set = set()
            assigned_dets: set = set()
            pairs = []
            for ti, tr in enumerate(live):
                for di, det in enumerate(dets):
                    dx = tr.last_centroid[0] - det.centroid[0]
                    dy = tr.last_centroid[1] - det.centroid[1]
                    d = math.hypot(dx, dy)
                    if d <= cfg.max_jump:
                        pairs.append((d, ti, di))
            for d, ti, di in sorted(pairs):
                if ti in assigned_tracks or di in assigned_dets:
                    continue
                tr = live[ti]
                det = dets[di]
                tr.observations.append(det)
                tr.last_centroid = det.centroid
                assigned_tracks.add(ti)
                assigned_dets.add(di)
            still_live: List[Track] = []
            for ti, tr in enumerate(live):
                if ti in assigned_tracks:
                    still_live.append(tr)
                else:
                    finished.append(tr)
            for di, det in enumerate(dets):
                if di in assigned_dets:
                    continue
                tr = Track(track_id=self._next_track_id)
                self._next_track_id += 1
                tr.observations.append(det)
                tr.last_centroid = det.centroid
                still_live.append(tr)
            live = still_live
        all_tracks = finished + live
        persistent = [t for t in all_tracks if len(t.observations) >= cfg.min_track_len]
        return VideoSnapshot(num_frames=n_frames, fps=self.source.fps, tracks=persistent)
