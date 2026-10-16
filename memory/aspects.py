"""memory/aspects.py -- extracted from consolidator.py (Move 5a).

MOVE ONLY: whole top-level statements, verbatim.

[P4 classifier of record] This module IS the category scorer of record.
`AspectGraph.category_scores` is the only path that turns a view descriptor
into per-category scores; nothing in the default scoring path reads the
consolidator's core machinery, which stays a memory/operator store (core
growth, containment, margins). The demoted virtual-view branch is off by
default (see category_scores) and no other scorer is consulted anywhere in
the pipeline.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Sequence
import parameters as P
from core.registries import (  # noqa: F401
    CONSTITUTIONAL_RANGES,
)
from perception.contours import (  # noqa: F401
    ribbon_bins,
)

_RIBBON_R = CONSTITUTIONAL_RANGES.get("RIBBON", (4.0, 8.0, 8.0, 8.0, 8.0))

_ENSEMBLE_R = CONSTITUTIONAL_RANGES.get("ENSEMBLE", (4.0, 4.0, 4.0, 4.0, 3.0, 8.0, 8.0, 8.0))

# [Stage 5 rewire] Slot ranges for the PEAK_CHAIN descriptor source. A
# PEAK_CHAIN carries six slots (n_peaks, mean_theta, theta_var, closure,
# curvature, length); slot 1 is circular over 12, which under the tree's N/2
# convention declares range 6.0 while the bin index runs 0..11 -- so it is
# looked up explicitly rather than reused verbatim from the registry.
_CHAIN_R = (15.0, 6.0, 7.0, 1.0, 7.0, 7.0)


def _ranges_for(n_slots: int) -> Tuple[float, ...]:
    """Slot ranges by descriptor arity, so both sources share one matcher.

    Length 6 is a PEAK_CHAIN; anything else is a shock RIBBON. Dispatching on
    arity (rather than on a global mode) means a view built from chains and a
    view built from ribbons stay distinguishable to the matcher instead of
    silently mismatching.
    """
    return _CHAIN_R if n_slots == 6 else _RIBBON_R

@dataclass
class PartDesc:
    bins: Tuple[float, ...]
    pos: Tuple[float, float]      # Object-centroid relative, RMS-spread normalized
    ang: float                    # Axial axis orientation in radians [0, pi)
    ens: int                      # Ensemble index within view (-1 if unassigned)

@dataclass(eq=False)
class ViewRecord:
    vid: int
    t: int
    instance: str
    label: Any
    az: float
    el: float
    parts: List[PartDesc]
    ens_bins: List[Tuple[float, ...]]
    proto: int = -1

def build_view_descriptor(shock_struct: Any) -> Optional[Tuple[List[PartDesc], List[Tuple[float, ...]]]]:
    """Convert Stage 4 shock structure into invariant part descriptors and ensemble bins."""
    if not shock_struct:
        return None
    segs, _joints, ens, meta = shock_struct
    if not segs or not meta or "R" not in meta:
        return None
    R = meta["R"]

    pts = [(a[0], a[1]) for s in segs for a in s.get("axis", [])]
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    ens_of: Dict[int, int] = {}
    if ens:
        for e in ens:
            for sid in e.get("segments", []):
                ens_of[sid] = e["id"]

    parts: List[PartDesc] = []
    for s in sorted(segs, key=lambda s: s["id"]):
        ax = s.get("axis", [])
        if not ax:
            continue
        mid = ax[len(ax) // 2]
        if len(ax) >= 2:
            ang = math.atan2(ax[-1][1] - ax[0][1], ax[-1][0] - ax[0][0]) % math.pi
        else:
            ang = 0.0
        parts.append(
            PartDesc(
                bins=ribbon_bins(s, R),
                pos=((mid[0] - cx) / R, (mid[1] - cy) / R),
                ang=ang,
                ens=ens_of.get(s["id"], -1),
            )
        )

    ens_bins = [tuple(e["bins"]) for e in sorted(ens, key=lambda e: e["id"])] if ens else []
    return parts, ens_bins

def _chain_members(det: Any) -> List[Any]:
    """Member anchors of a PEAK_CHAIN detection, in the chain's own order."""
    return [r[1] for r in getattr(det, "rho", ()) or ()
            if len(r) > 1 and r[0] == "member"]


def _chain_axis_angle(pts: Sequence[Tuple[float, float]]) -> float:
    """Circular-mean direction of a chain's member steps, on [0, pi).

    Each member contributes its step direction toward the next member, and
    the directions are combined on the pi-periodic axis (a contour axis has
    no head/tail), so a chain traversed in either direction yields the same
    axis. Falls back to the first step when the mean is degenerate.
    """
    if len(pts) < 2:
        return 0.0
    sx = sy = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        m = math.hypot(dx, dy)
        if m < 1e-9:
            continue
        sx += dx / m
        sy += dy / m
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return 0.0
        return math.atan2(dy, dx) % math.pi
    return math.atan2(sy, sx) % math.pi


def build_view_descriptor_from_chains(
    chain_dets: Any,
    anchor_positions: Optional[Dict[Any, Any]],
    R: float,
) -> Optional[Tuple[List[PartDesc], List[Tuple[float, ...]]]]:
    """[Stage 5 rewire] View descriptor from the live field's PEAK_CHAINs.

    Replaces the shock-structure source, which is unreachable on this tree
    (the segmenter region path is empty, so no segments are ever produced).
    A PEAK_CHAIN is the same OBJECT CLASS a shock ribbon was -- a bounded
    contour fragment with an internal axis -- so it drops straight into the
    descriptor, matching, resonance, clustering and link machinery without
    changing any of it.

    Each chain becomes one part:
      bins  its six quantized slots
      pos   its member centroid, object-centre relative, RMS-spread normalized
      ang   the circular mean of its member step directions

    `ens` stays -1 for every chain: ensembles were a shock-graph grouping and
    do not exist at the chain level yet. `match_views` handles ens=-1 parts
    on its residual path, so ensembles are not required for matching to work.

    [Q2 bug fix] `pos` used to be divided by the median WAVELENGTH (R), which
    made every position ~15 units wide against a 0.35-wide position kernel:
    the position term was identically zero for every part pair, so matching
    degenerated into a pure bag-of-slots comparison (measured: 10226
    same-class and 21785 cross-class part pairs with slot similarity >= 0.90
    and median position distance 14.8 / 15.5). The normaliser is now the
    view's own RMS member spread about the object centroid, weighted by each
    chain's peak count, so a part position is O(1) and the position kernel can
    do its job.
    """
    if not chain_dets or not R or R <= 0.0:
        return None
    ordered = sorted(chain_dets, key=lambda d: getattr(d, "alpha", 0))
    raw: List[Tuple[Tuple[float, ...], List[Tuple[float, float]]]] = []
    for det in ordered:
        q = getattr(getattr(det, "sigma", None), "quantized", None)
        if not q or len(q) < 6:
            continue
        pts: List[Tuple[float, float]] = []
        for a in _chain_members(det):
            if anchor_positions is None:
                continue
            p = anchor_positions.get(a)
            if p is not None and len(p) >= 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except (TypeError, ValueError):
                    continue
        if len(pts) < 2:
            continue
        raw.append((tuple(float(v) for v in q[:6]), pts))
    if not raw:
        return None

    # global object centroid across all chains, matching the ribbon source's
    # "all axis points" convention, so pos is comparable between the two.
    all_pts = [p for _b, pts in raw for p in pts]
    cx = sum(p[0] for p in all_pts) / len(all_pts)
    cy = sum(p[1] for p in all_pts) / len(all_pts)

    # RMS member spread about that centroid, weighted by chain peak count
    # (bins[0]). Fallback to the wavelength when the view has no spread, so
    # the normaliser is always finite and positive.
    num = 0.0
    den = 0.0
    for bins, pts in raw:
        w = max(1.0, float(bins[0]))
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        num += w * ((mx - cx) ** 2 + (my - cy) ** 2)
        den += w
    rms = math.sqrt(num / den) if den > 0.0 else 0.0
    if not (rms > 1e-6):
        rms = float(R)

    parts: List[PartDesc] = []
    for bins, pts in raw:
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        parts.append(PartDesc(
            bins=bins,
            pos=((mx - cx) / rms, (my - cy) / rms),
            ang=_chain_axis_angle(pts),
            ens=-1,
        ))
    return parts, []


def _awrap(x: float) -> float:
    """Wrap angular differences into [-pi/2, pi/2)."""
    return (x + math.pi / 2.0) % math.pi - math.pi / 2.0

def _bins_sim(a: Sequence[float], b: Sequence[float], ranges: Sequence[float]) -> float:
    n = min(len(a), len(b), len(ranges))
    if n == 0:
        return 0.0
    return math.exp(-sum(((a[i] - b[i]) / (max(ranges[i], 1e-6) / 3.0)) ** 2 for i in range(n)) / n)

def _greedy_match(sim: Dict[Tuple[int, int], float], min_sim: float) -> List[Tuple[int, int, float]]:
    out: List[Tuple[int, int, float]] = []
    ui: Set[int] = set()
    uj: Set[int] = set()
    for (i, j), v in sorted(sim.items(), key=lambda kv: (-kv[1], kv[0])):
        if v >= min_sim and i not in ui and j not in uj:
            out.append((i, j, v))
            ui.add(i)
            uj.add(j)
    return out

def match_views(
    A: ViewRecord, B: ViewRecord
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float, int]]]:
    """Hierarchical part persistence matcher:
    1. Match macro-ensembles via greedy bipartite alignment.
    2. Match parts conditioned on matched parent ensembles.
    3. Match residual unassigned parts using stricter similarity thresholds.

    [Q2] When `P.ASPECT_GRADED_PARTS` is on, part similarity is computed on
    the coarsened slot set `P.ASPECT_PART_SLOTS` and the exact slot-0
    equality gate is replaced by the graded slot distance (which already
    carries slot 0). The flag ships DARK (off): this is a scoring-facing
    change, so per the operating policy it is not enabled until second
    instances per class confirm it. The legacy path (all six slots, exact
    slot-0 gate x ASPECT_ORDER_MISMATCH) stays the default.
    """
    ms = P.ASPECT_MATCH_MIN
    sp = P.ASPECT_SIGMA_POS

    esim = {
        (i, j): _bins_sim(ea, eb, _ENSEMBLE_R)
        for i, ea in enumerate(A.ens_bins)
        for j, eb in enumerate(B.ens_bins)
    }
    em = _greedy_match(esim, ms)
    emap = {i: j for i, j, _ in em}

    slots = tuple(P.ASPECT_PART_SLOTS) if P.ASPECT_GRADED_PARTS else ()
    v2_rng = tuple(_CHAIN_R[k] for k in slots if k < len(_CHAIN_R))

    def psim(i: int, j: int) -> float:
        pa, pb = A.parts[i], B.parts[j]
        if P.ASPECT_GRADED_PARTS:
            # [Q2] coarsened slot set + graded slot-0 (no exact identity gate)
            qa = tuple(pa.bins[k] for k in slots if k < len(pa.bins))
            qb = tuple(pb.bins[k] for k in slots if k < len(pb.bins))
            s = _bins_sim(qa, qb, v2_rng) if len(qa) == len(v2_rng) else 0.0
        else:
            # legacy: exact slot-0 identity gate ("shock order"), ranges by
            # descriptor arity (6 slots = PEAK_CHAIN, 5 = shock RIBBON)
            order_penalty = 1.0 if pa.bins[0] == pb.bins[0] else P.ASPECT_ORDER_MISMATCH
            s = _bins_sim(pa.bins, pb.bins, _ranges_for(len(pa.bins))) * order_penalty
        dist_sq = (pa.pos[0] - pb.pos[0]) ** 2 + (pa.pos[1] - pb.pos[1]) ** 2
        return s * math.exp(-(math.sqrt(dist_sq) / sp) ** 2)

    pm: List[Tuple[int, int, float, int]] = []
    ua: Set[int] = set()
    ub: Set[int] = set()

    # Level 1: Match within agreed ensemble contexts
    s1 = {
        (i, j): psim(i, j)
        for i, pa in enumerate(A.parts)
        for j, pb in enumerate(B.parts)
        if pa.ens >= 0 and emap.get(pa.ens, -2) == pb.ens
    }
    for i, j, v in _greedy_match(s1, ms):
        pm.append((i, j, v, 1))
        ua.add(i)
        ub.add(j)

    # Level 2: Match residual parts across unmapped contexts
    s2 = {
        (i, j): psim(i, j)
        for i in range(len(A.parts))
        if i not in ua
        for j in range(len(B.parts))
        if j not in ub
    }
    for i, j, v in _greedy_match(s2, 1.25 * ms):
        pm.append((i, j, v, 2))
        ua.add(i)
        ub.add(j)

    return em, sorted(pm, key=lambda x: (x[3], -x[2]))

def _dOmega(a_az: float, a_el: float, b_az: float, b_el: float) -> Tuple[float, float, float]:
    """Calculate (delta_azimuth in [-180, 180), delta_elevation, great_circle_distance_deg)."""
    daz = (b_az - a_az + 180.0) % 360.0 - 180.0
    ea, eb = math.radians(a_el), math.radians(b_el)
    c = math.sin(ea) * math.sin(eb) + math.cos(ea) * math.cos(eb) * math.cos(math.radians(daz))
    dist = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return daz, b_el - a_el, dist

class AspectGraph:
    """Aspect Graph Memory Engine.
    Clusters views into canonical Aspect Prototypes via ART resonance,
    links adjacent viewpoints with persistent transformation traces,
    and synthesizes virtual halfway views to support unseen view classification.
    """

    def __init__(self) -> None:
        self.views: List[ViewRecord] = []
        self.protos: Dict[int, Dict[str, Any]] = {}
        self.links: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.diag: Counter = Counter()
        self._by_instance: Dict[str, List[int]] = defaultdict(list)

    @staticmethod
    def make_view(
        desc: Tuple[List[PartDesc], List[Tuple[float, ...]]],
        instance: str = "",
        label: Any = None,
        az: float = 0.0,
        el: float = 0.0,
        vid: int = -1,
        t: int = 0,
    ) -> ViewRecord:
        return ViewRecord(vid, t, instance, label, az, el, desc[0], desc[1])

    @staticmethod
    def _resonance(A: ViewRecord, B: ViewRecord) -> float:
        """Balanced Sørensen–Dice resonance over matched parts:
        Resonance = (2 * sum(matched_sim)) / (|A.parts| + |B.parts|)
        Tolerates oversegmentation noise while strictly penalizing missing structures.
        """
        _em, pm = match_views(A, B)
        total_parts = len(A.parts) + len(B.parts)
        if total_parts == 0 or not pm:
            return 0.0
        return (2.0 * sum(s for _i, _j, s, _l in pm)) / float(total_parts)

    def observe(
        self, t: int, instance: str, label: Any, az: float, el: float, desc: Optional[Tuple[List[PartDesc], List[Tuple[float, ...]]]]
    ) -> Optional[ViewRecord]:
        if not P.ASPECT_ENABLED or desc is None:
            return None
        v = ViewRecord(len(self.views), int(t), str(instance), label, float(az), float(el), desc[0], desc[1])
        v.proto = self._assign(v)
        self._decay()
        for uid in self._link_candidates(v):
            self._trace_link(self.views[uid], v)
        self.views.append(v)
        self._by_instance[v.instance].append(v.vid)
        return v

    def _link_candidates(self, v: ViewRecord) -> List[int]:
        if P.ASPECT_PAIR_MODE == "temporal":
            return [self.views[-1].vid] if self.views and v.t - self.views[-1].t <= 1 else []
        return [
            u
            for u in self._by_instance[v.instance]
            if _dOmega(self.views[u].az, self.views[u].el, v.az, v.el)[2] <= P.ASPECT_LINK_MAX_DEG
        ]

    def _assign(self, v: ViewRecord) -> int:
        best_pid: Optional[int] = None
        best_r = -1.0
        for pid in sorted(self.protos):
            p = self.protos[pid]
            if _dOmega(p["az"], p["el"], v.az, v.el)[2] > P.ASPECT_ASSIGN_MAX_DEG:
                continue
            if v.label is not None and p["labels"]:
                majority_label = sorted(p["labels"].items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
                if majority_label != v.label:
                    continue  # Match tracking: prevent cross-class prototype pollution
            r = self._resonance(v, p["rep"])
            if r > best_r:
                best_pid = pid
                best_r = r

        if best_pid is not None and best_r >= P.ASPECT_VIGILANCE:
            p = self.protos[best_pid]
            n = p["n"]
            # Circular mean update for prototype viewpoint
            rad_p = math.radians(p["az"])
            rad_v = math.radians(v.az)
            # Spherical EMA: treats existing mean as n unit vectors.
            # Inflates weight of established prototypes to resist outlier drift.
            x = math.cos(rad_p) * n + math.cos(rad_v)
            y = math.sin(rad_p) * n + math.sin(rad_v)
            p["az"] = math.degrees(math.atan2(y, x)) % 360.0
            # Re-wrap: a tiny-negative atan2 result rounds to exactly 360.0
            # under the float modulo above; keep azimuth in [0, 360).
            p["az"] %= 360.0
            p["el"] = (p["el"] * n + v.el) / (n + 1)
            p["n"] = n + 1
            if v.label is not None:
                p["labels"][v.label] += 1
            p["views"].append(v.vid)
            self.diag["assign_existing"] += 1
            return best_pid

        new_pid = len(self.protos)
        self.protos[new_pid] = {
            "rep": v,
            "az": v.az,
            "el": v.el,
            "n": 1,
            "labels": Counter([v.label] if v.label is not None else []),
            "views": [v.vid],
        }
        self.diag["assign_new"] += 1
        return new_pid

    def _decay(self) -> None:
        for L in self.links.values():
            L["w"] *= P.ASPECT_TRACE_DECAY

    def _trace_link(self, U: ViewRecord, V: ViewRecord) -> None:
        pu, pv = U.proto, V.proto
        if pu == pv:
            self.diag["link_same_proto"] += 1
            return
        daz, dele, ang = _dOmega(U.az, U.el, V.az, V.el)
        _em, pm = match_views(U, V)
        a, b, flip = (pu, pv, False) if pu < pv else (pv, pu, True)
        if flip:
            daz, dele = -daz, -dele
        L = self.links.setdefault(
            (a, b),
            {
                "w": 0.0,
                "n": 0,
                "daz": 0.0,
                "del": 0.0,
                "m2az": 0.0,
                "m2el": 0.0,
                "persist": {},
                "virt": None,
            },
        )
        prox = math.exp(-(ang / P.ASPECT_LINK_SIGMA_DEG) ** 2)
        L["w"] += P.ASPECT_TRACE_ETA * prox * (1.0 - L["w"] / P.ASPECT_W_MAX)
        L["n"] += 1

        d1 = daz - L["daz"]
        L["daz"] += d1 / L["n"]
        L["m2az"] += d1 * (daz - L["daz"])
        d2 = dele - L["del"]
        L["del"] += d2 / L["n"]
        L["m2el"] += d2 * (dele - L["del"])

        for i, j, _s, _lvl in pm:
            pa, pb = U.parts[i], V.parts[j]
            if flip:
                pa, pb = pb, pa
            e = L["persist"].setdefault((pa.bins, pb.bins), {"n": 0, "dx": 0.0, "dy": 0.0, "da": 0.0})
            e["n"] += 1
            k = 1.0 / e["n"]
            e["dx"] += k * ((pb.pos[0] - pa.pos[0]) - e["dx"])
            e["dy"] += k * ((pb.pos[1] - pa.pos[1]) - e["dy"])
            e["da"] += k * (_awrap(pb.ang - pa.ang) - e["da"])

        while len(L["persist"]) > P.ASPECT_PERSIST_CAP:
            victim_key = min(L["persist"], key=lambda kk: (L["persist"][kk]["n"], repr(kk)))
            L["persist"].pop(victim_key)

        L["virt"] = None  # Invalidate cached interpolated view
        self.diag["links_updated"] += 1

    def _virtual(self, key: Tuple[int, int], L: Dict[str, Any]) -> ViewRecord:
        """Synthesize intermediate virtual view from the persistence map across prototypes."""
        if L["virt"] is not None:
            return L["virt"]
        A = self.protos[key[0]]["rep"]
        B = self.protos[key[1]]["rep"]
        _em, pm = match_views(A, B)

        parts: List[PartDesc] = []
        ua: Set[int] = set()
        ub: Set[int] = set()

        for i, j, _s, _l in pm:
            pa, pb = A.parts[i], B.parts[j]
            parts.append(
                PartDesc(
                    bins=pa.bins,
                    pos=(0.5 * (pa.pos[0] + pb.pos[0]), 0.5 * (pa.pos[1] + pb.pos[1])),
                    ang=(pa.ang + 0.5 * _awrap(pb.ang - pa.ang)) % math.pi,
                    ens=-1,
                )
            )
            ua.add(i)
            ub.add(j)

        # Append unmapped residual parts from both viewpoints
        parts.extend([PartDesc(p.bins, p.pos, p.ang, -1) for i, p in enumerate(A.parts) if i not in ua])
        parts.extend([PartDesc(p.bins, p.pos, p.ang, -1) for j, p in enumerate(B.parts) if j not in ub])

        # Exact circular mean for virtual view azimuth
        rad_a = math.radians(A.az)
        rad_b = math.radians(B.az)
        ax = math.cos(rad_a) + math.cos(rad_b)
        ay = math.sin(rad_a) + math.sin(rad_b)
        virt_az = math.degrees(math.atan2(ay, ax)) % 360.0
        # Re-wrap: a tiny-negative atan2 result rounds to exactly 360.0 under
        # the float modulo above; keep azimuth in [0, 360).
        virt_az %= 360.0

        L["virt"] = ViewRecord(
            vid=-1,
            t=0,
            instance="",
            label=None,
            az=virt_az,
            el=0.5 * (A.el + B.el),
            parts=parts,
            ens_bins=[],
        )
        return L["virt"]

    def category_scores(
        self,
        desc: Tuple[List[PartDesc], List[Tuple[float, ...]]],
        use_links: bool = True,
        use_virtual: bool = False,
    ) -> Dict[Any, float]:
        """Evaluate category recognition score using prototypes and (optionally)
        virtual views.

        [Aspect-graph demotion, 2026-10-10] `use_virtual` defaults to False:
        the midpoint-interpolated virtual view is no longer part of the scoring
        path. Evidence: on the ETH-80 equator ring the virtual view did not
        beat the better flank on F (-0.017 [-0.026, -0.009]); its apparent
        coverage win was a part-set-size effect (the mismatched-pair control
        absorbed ~80% of it); the true interpolation gain shrank as the
        position kernel tightened; and matched-part "motion" was null-sized
        against a same-viewpoint cross-instance control (0.159R within vs
        0.193R across), i.e. the stored displacement is matcher noise.

        So the claim supported by the data is about THIS construction
        (midpoint interpolation of shock-ribbon descriptors at this
        resolution), not "views cannot be interpolated" in general.

        Links and their persistence maps remain maintained as a
        transition-statistics store (part-type persistence and per-type mean
        motion); nothing in the default scoring path reads them.
        """
        q = ViewRecord(-1, 0, "", None, 0.0, 0.0, desc[0], desc[1])
        scores: Dict[Any, float] = defaultdict(float)

        def add(labels: Counter, r: float) -> None:
            tot = sum(labels.values())
            if tot <= 0:
                return
            for lab, c in labels.items():
                scores[lab] = max(scores[lab], r * (c / tot))

        # Direct prototype resonance
        for pid in sorted(self.protos):
            if self.protos[pid]["labels"]:
                add(self.protos[pid]["labels"], self._resonance(q, self.protos[pid]["rep"]))

        # Link-induced virtual view resonance (demoted: off by default)
        if use_links and use_virtual:
            for key in sorted(self.links):
                L = self.links[key]
                if L["w"] < P.ASPECT_LINK_MIN_W:
                    continue
                labs = self.protos[key[0]]["labels"] + self.protos[key[1]]["labels"]
                if labs:
                    weight_scale = math.sqrt(min(1.0, L["w"] / P.ASPECT_W_MAX))
                    virt_resonance = self._resonance(q, self._virtual(key, L))
                    add(labs, weight_scale * virt_resonance)

        return dict(scores)

    def summary(self) -> Dict[str, Any]:
        ws = [L["w"] for L in self.links.values()]
        return {
            "views": len(self.views),
            "protos": len(self.protos),
            "links": len(self.links),
            "mean_w": (sum(ws) / len(ws)) if ws else 0.0,
            **dict(self.diag),
        }

