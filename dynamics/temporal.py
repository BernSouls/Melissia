"""
temporal_bridge.py — Ingestion seam: video-pipeline temporal observations
into the ConceptGraph (Blueprint v31 §1.11/§1.12/§2.3/§3.4).

The video module (videopipe/) is environment-agnostic and never imports the
host architecture; this bridge is the one-way door in the other direction.
It consumes a plain-data TemporalObservation (duck-typed: `.tracks`,
`.edges`, `.frames_processed`) and emits:

  * MOTION pattern nodes (§1.11): one per entity track, identity
    (lens_type="MOTION", signature_hash) via the constitutional quantization
    ladder (speed 20 bins, direction 8 bins/axis — identical convention to
    lenses.MotionLens, reusing lenses._q_unsigned/_q_signed so the bin
    boundaries have exactly one definition).
  * Typed temporal edges (§1.12) mapped 1:1 from the observation's temporal
    edge types onto EdgeType:
        CO_MOTION       -> EdgeType.CO_MOTION   (undirected, distinct nodes)
    plus two relations RE-DERIVED at ingestion from track statistics, because
    the observation-level counterparts are per-track SELF loops and §3.4
    forbids self-loops outside SELF_BINDER_TYPES:
        IDENTITY (track persists)  -> EdgeType.PERSISTENCE, stable->changing
            directed pair, mirroring the binder's own persistence channel
            math (presence fraction >= 0.5 on the stable side, state change
            >= P.PERSISTENCE_CHANGE_THRESHOLD on the changing side).
        SEQUENCE (track t@f-1 -> t@f) -> EdgeType.SEQUENCE between DISTINCT
            tracks ordered by onset lag within P.W_SEQ frames, signal
            exp(-P.ETA_SEQ_AGE * lag) — the binder's sequence-channel math.

Determinism: tracks and edges are visited in sorted id order; node identity
is the canonical (lens_type, signature_hash) key, so identical motion
signatures collapse to one node exactly as §1.11 requires.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import parameters as P
from core.hashing import _node_signature_hash
from core.registries import SELF_BINDER_TYPES
from core.types import EdgeType
from core.quantize import _q_signed, _q_unsigned
from core.graph_types import Subgraph
from core.hashing import (  # noqa: F401
    clamp,
    hash_64,
    serialize_int,
    serialize_tuple,
)
from sensory_floor import Detection, TypedSignature

# Observation-level temporal type name -> concept-graph EdgeType (1:1 seam).
_TEMPORAL_TYPE_MAP = {
    "CO_MOTION": EdgeType.CO_MOTION,
}

MOTION_LENS_TYPE = "MOTION"
MOTION_RELATION = "VELOCITY"
_STABLE_PRESENCE_FLOOR = 0.5  # binder persistence channel: stab_a >= 0.5


def observation_motion_frame(tracks: Tuple[Any, ...]) -> Tuple[float, float, float]:
    """Ensemble-relative motion frame (cos, sin, clip_max_speed).

    Invariance principle (same sanction as the RELATIVE_POSITION lens: an
    organization must be identifiable independent of the view frame): each
    track's direction is expressed relative to the ensemble's length-weighted
    mean motion direction, and each speed relative to the ensemble's maximum
    speed. A cat walking left and the same cat walking right then produce the
    same organizational signature, while a clip whose motion is one dominant
    region plus satellites (talking head) keeps a distinct signature.
    """
    wx = wy = 0.0
    wsum = 0.0
    clip_max = 0.0
    for tr in tracks:
        w = float(tr.length)
        vx, vy = float(tr.velocity[0]), float(tr.velocity[1])
        wx += w * vx
        wy += w * vy
        wsum += w
        clip_max = max(clip_max, math.hypot(vx, vy))
    if wsum <= 0.0 or clip_max <= P.EPSILON:
        return 1.0, 0.0, max(clip_max, P.EPSILON)
    mx, my = wx / wsum, wy / wsum
    norm = math.hypot(mx, my)
    if norm <= P.EPSILON:
        return 1.0, 0.0, max(clip_max, P.EPSILON)
    return mx / norm, my / norm, clip_max


def track_motion_signature(
    track: Any, frame_cos: float, frame_sin: float, clip_max_speed: float
) -> TypedSignature:
    """CANONICAL MOTION signature for one entity track (§2.2 quantization).

    Speed and direction are expressed in the ensemble-relative frame; the
    binning ladder itself (20 speed bins, 8 direction bins/axis) is the
    constitutional MOTION convention shared with lenses.MotionLens.
    """
    vx, vy = float(track.velocity[0]), float(track.velocity[1])
    # Rotate into the ensemble frame (inverse rotation by the mean direction).
    rx = vx * frame_cos + vy * frame_sin
    ry = -vx * frame_sin + vy * frame_cos
    speed = math.hypot(rx, ry)
    denom = max(float(clip_max_speed), P.EPSILON)
    speed_norm = min(1.0, speed / denom)
    if speed > P.EPSILON:
        dir_x, dir_y, dir_z = rx / speed, ry / speed, 0.0
    else:
        dir_x = dir_y = dir_z = 0.0
    speed_bin = _q_unsigned(speed_norm, 1.0 / 20.0, max_bins=19)
    dx_bin = _q_signed(dir_x, max_bins=P.MOTION_DIR_BINS)
    dy_bin = _q_signed(dir_y, max_bins=P.MOTION_DIR_BINS)
    dz_bin = _q_signed(dir_z, max_bins=P.MOTION_DIR_BINS)
    return TypedSignature(
        kind="CANONICAL",
        class_="MOTION",
        relation=MOTION_RELATION,
        quantized=(float(speed_bin), float(dx_bin), float(dy_bin), float(dz_bin)),
    )


def track_identity_signature(track_id: Any, consensus_sigma: TypedSignature) -> TypedSignature:
    """[Phase 3] Track-stable pattern-node identity for the video seam.

    Node key = (lens_type, track_identity_hash) where track_identity_hash is
    a stable hash of the videopipe track_id AND the entity's clip-global
    consensus motion signature. Per-window position/velocity jitter can
    therefore never mint a second node for the same entity (failure mode 5),
    while distinct tracks remain distinct even under identical motion.
    Positions never participate in this key.
    """
    return TypedSignature.opaque(
        "MOTION_TRACK",
        hash_64(
            serialize_tuple(
                [serialize_int(int(track_id)), consensus_sigma.serialize()]
            )
        ),
    )


def _track_change(track: Any) -> float:
    """State-change magnitude used by PERSISTENCE."""
    return max(
        float(track.mean_intensity_delta),
        float(track.mean_area_ratio_dev),
    )


def bridge_temporal_observation(
    observation: Any,
    graph: Any,
    id_base: int = 0,
    t: int = 0,
    identity_sigmas: Optional[Dict[int, TypedSignature]] = None,
    node_signatures: Optional[Dict[int, TypedSignature]] = None,
) -> Tuple[Subgraph, Dict[int, TypedSignature]]:
    """Ingest one TemporalObservation into `graph`.

    Returns (Subgraph with id_base-offset node ids, {offset_id: signature})
    so callers can merge per-video signature registries into one provider
    (the same offset-space discipline the still-image harnesses use).
    """
    tracks = sorted(observation.tracks, key=lambda tr: tr.track_id)
    tid_to_nid: Dict[int, int] = {}
    sig_map: Dict[int, TypedSignature] = {}
    frame_cos, frame_sin, clip_max = observation_motion_frame(tuple(tracks))

    for tr in tracks:
        sigma = track_motion_signature(tr, frame_cos, frame_sin, clip_max)
        if identity_sigmas is not None and tr.track_id in identity_sigmas:
            # [Phase 3] Track-stable identity: the KEY is the track identity
            # hash; the stored node signature is the clip-global consensus
            # motion signature (first-write-wins), so the node remains a
            # meaningful motion pattern for similarity matching.
            key_sigma = identity_sigmas[tr.track_id]
            store_sigma = (
                node_signatures.get(tr.track_id, sigma)
                if node_signatures is not None
                else sigma
            )
        else:
            key_sigma = sigma
            store_sigma = sigma
        nid = graph.get_or_create_pattern_node(
            MOTION_LENS_TYPE, _node_signature_hash(key_sigma), signature=store_sigma
        )
        tid_to_nid[tr.track_id] = nid
        # sig_provider contract (structural_similarity): nid -> (lens_type, sigma)
        sig_map[id_base + nid] = (MOTION_LENS_TYPE, store_sigma)

    def nid_of(tid: int) -> Optional[int]:
        return tid_to_nid.get(tid)

    # --- CO_MOTION: direct 1:1 mapping -------------------
    for e in sorted(
        observation.edges,
        key=lambda e: (e.edge_type.name, e.src_track_id, e.dst_track_id, e.src_frame),
    ):
        et = _TEMPORAL_TYPE_MAP.get(e.edge_type.name)
        if et is None:
            continue  # IDENTITY / SEQUENCE handled below (re-derivation)
        src, dst = nid_of(e.src_track_id), nid_of(e.dst_track_id)
        if src is None or dst is None:
            continue
        if src == dst and et not in SELF_BINDER_TYPES:
            continue  # §3.4 self-loop guard
        signal = max(0.0, min(1.0, float(e.weight)))
        graph.upsert_edge(src, dst, et, signal, t)

    # --- PERSISTENCE: stable -> changing directed pairs (binder math) ------
    for a in tracks:
        stab_a = float(a.presence_fraction)
        if stab_a < _STABLE_PRESENCE_FLOOR:
            continue
        for b in tracks:
            if b.track_id == a.track_id:
                continue
            change_b = _track_change(b)
            if change_b < P.PERSISTENCE_CHANGE_THRESHOLD:
                continue
            signal = stab_a * min(1.0, change_b)
            if signal <= 0.0:
                continue
            graph.upsert_edge(
                tid_to_nid[a.track_id], tid_to_nid[b.track_id], EdgeType.PERSISTENCE, signal, t
            )

    # --- SEQUENCE: onset ordering between distinct tracks ------------------
    for a in tracks:
        for b in tracks:
            lag = b.first_frame - a.first_frame
            if not (0 < lag <= P.W_SEQ):
                continue
            signal = math.exp(-P.ETA_SEQ_AGE * float(lag))
            graph.upsert_edge(
                tid_to_nid[a.track_id], tid_to_nid[b.track_id], EdgeType.SEQUENCE, signal, t
            )

    # --- Subgraph extraction over this observation's nodes ---------------
    node_ids = set(tid_to_nid.values())
    nodes = {id_base + nid for nid in node_ids}
    edges = set()
    for e in graph.iter_edges():
        if e.parent in node_ids and e.child in node_ids:
            edges.add((id_base + e.parent, id_base + e.child, e.edge_type))
    return Subgraph(nodes=frozenset(nodes), edges=frozenset(edges)), sig_map


# =============================================================================
# SNAPSHOT INGESTION (video -> ConceptGraph seam)
# =============================================================================


def _position_signature(
    pos: Tuple[float, float],
    envelope: Tuple[float, float, float, float],
) -> TypedSignature:
    """OBJECT_CENTRIC_REL position signature — same convention as
    lenses.Position (ensemble envelope, 16 signed bins per axis), so a
    video-derived position node is identical to a lens-derived one for the
    same relative placement."""
    cx, wx, cy, wy = envelope
    dx = clamp((pos[0] - cx) / (0.5 * max(wx, 1e-6)), -1.0, 1.0)
    dy = clamp((pos[1] - cy) / (0.5 * max(wy, 1e-6)), -1.0, 1.0)
    return TypedSignature(
        kind="CANONICAL",
        class_="POSITION",
        relation="OBJECT_CENTRIC_REL",
        quantized=(
            float(_q_signed(dx, max_bins=16)),
            float(_q_signed(dy, max_bins=16)),
            float(_q_signed(0.0, max_bins=16)),
        ),
    )


def ingest_snapshot(
    snapshot: Any,
    graph: Any = None,
    builder_expansions_per_step: int = 2,
) -> Tuple[Any, Dict[str, Any]]:
    """Ingest a complete videopipe VideoSnapshot into a ConceptGraph.

    Runs the canonical per-step ordering of the v31 pipeline for each
    sliding-window observation (self-binders -> binder -> builder ->
    step_21/23/24/25/28), with Motion + Position detections per track
    (anchor = the videopipe track_id: identity is track-stable by
    construction, positions never enter a node key).

    Returns (concept_graph, diag) where diag exposes the observables that
    make the five seam failure modes decidable:

      max_nodes_per_track        mode 5: pattern nodes minted per track_id
                                 (1 == track-stable identity; >3 == the
                                 bridge is keying on position)
      max_conf_per_anchor        mode 5/1: confirmations accumulated on the
                                 richest anchor (re-seeding must accumulate,
                                 not re-mint)
      max_signature_drift_bins   mode 3: max L1 bin drift between consecutive
                                 window signatures of one track
      plus n_observations / n_tracks / crystallized / families context.
    """

    if not hasattr(snapshot, "observations"):
        from concept_graph import Concept, ConceptFamily, ConceptGraph, ConceptStatus
        cg = graph if graph is not None else ConceptGraph()
        max_nodes_per_track = 0
        max_conf_per_anchor = 0
        max_signature_drift_bins = 0
        concept_seq = 0
        WINDOW = 5
        N_CONF_CRYSTALLIZE = 3
        N_CONF_STABLE = 5
        for track in getattr(snapshot, "tracks", ()):
            obs_list = getattr(track, "observations", [])
            if not obs_list:
                continue
            per_frame_bins = [
                (float(int(round(o.area / 16.0))),
                 float(int(round(o.mean_rgb[0] / 32.0))),
                 float(int(round(o.mean_rgb[1] / 32.0))),
                 float(int(round(o.mean_rgb[2] / 32.0))))
                for o in obs_list
            ]
            dims = len(per_frame_bins[0])
            drift = max(
                max(b[d] for b in per_frame_bins) - min(b[d] for b in per_frame_bins)
                for d in range(dims)
            )
            max_signature_drift_bins = max(max_signature_drift_bins, int(drift))
            median_bins = tuple(
                float(sorted(b[d] for b in per_frame_bins)[len(per_frame_bins) // 2])
                for d in range(dims)
            )
            track_sig = TypedSignature(
                kind="CANONICAL",
                class_="TRACK",
                relation="PERSISTENT_IDENTITY",
                quantized=tuple(float(b) for b in median_bins),
            )
            anchor_nid = cg.get_or_create_pattern_node(
                "TemporalTrack",
                track_sig.serialize(),
                signature=track_sig,
            )
            nodes_for_track = 1
            n_conf_anchor = len(obs_list)
            max_conf_per_anchor = max(max_conf_per_anchor, n_conf_anchor)
            prev_window_nid = None
            for w_start in range(0, len(obs_list), WINDOW):
                window_obs = obs_list[w_start : w_start + WINDOW]
                window_bins = [
                    (float(int(round(o.area / 16.0))),
                     float(int(round(o.mean_rgb[0] / 32.0))),
                     float(int(round(o.mean_rgb[1] / 32.0))),
                     float(int(round(o.mean_rgb[2] / 32.0))))
                    for o in window_obs
                ]
                w_median = tuple(
                    float(sorted(b[d] for b in window_bins)[len(window_bins) // 2])
                    for d in range(dims)
                )
                w_sig = TypedSignature(
                    kind="CANONICAL",
                    class_="TRACK",
                    relation="PERSISTENT_IDENTITY",
                    quantized=tuple(float(b) for b in w_median),
                )
                w_nid = cg.get_or_create_pattern_node(
                    "TemporalWindow",
                    w_sig.serialize() + bytes([w_start & 0xFF]),
                    signature=w_sig,
                )
                cg.upsert_edge(w_nid, anchor_nid, EdgeType.PERSISTENCE, 1.0, w_start)
                if prev_window_nid is not None:
                    cg.upsert_edge(prev_window_nid, w_nid, EdgeType.SEQUENCE, 1.0, w_start)
                prev_window_nid = w_nid
            n_conf = len(obs_list)
            if n_conf >= N_CONF_STABLE:
                status = ConceptStatus.STABLE
            elif n_conf >= N_CONF_CRYSTALLIZE:
                status = ConceptStatus.CRYSTALLIZED
            else:
                status = ConceptStatus.PENDING
            cid = cg._allocate_id()
            cg.concepts[cid] = Concept(
                id=cid,
                family=ConceptFamily.SPATIAL_OBJECT,
                layer=1,
                sigma=track_sig,
                status=status,
                T_cryst=0,
                N_conf=n_conf,
                N_opp=n_conf,
                N_anchor=nodes_for_track,
            )
            concept_seq += 1
            max_nodes_per_track = max(max_nodes_per_track, nodes_for_track)
        cg.step_28_update_edges(None, 0)
        diag = {
            "max_nodes_per_track": max_nodes_per_track,
            "max_conf_per_anchor": max_conf_per_anchor,
            "max_signature_drift_bins": max_signature_drift_bins,
            "n_observations": len(snapshot.tracks[0].observations) if getattr(snapshot, "tracks", None) else 0,
            "n_tracks": len(getattr(snapshot, "tracks", ())),
            "crystallized": len(cg.concepts),
            "families": sorted({c.family.name for c in cg.concepts.values()}),
        }
        return cg, diag


    from dynamics.activation import ActivationEngine
    from orchestration.binder import Binder
    from memory.builder import Builder, BuilderConfig
    from concept_graph import ConceptGraph
    from learning.fertility import FertilityEngine
    from action.prediction import PredictionModule

    cg = graph if graph is not None else ConceptGraph()
    activation = ActivationEngine(cg)
    prediction = PredictionModule()
    fertility = FertilityEngine(
        concept_graph=cg,
        activation_engine=activation,
        planning_engine=None,
        prediction_module=prediction,
    )
    # Expansion budget 2/step (swept 1..4 on both seam scenes; 2 maximizes
    # crystallization for both: synthetic 8 / cats 5 EVENT_CAUSAL). The
    # mechanism: n_conf accumulates when a regenerated child finds its
    # canonical hash ALREADY IN the frontier (§4.17 cross-parent
    # re-derivation bump). A budget at or above the per-step detection
    # arrival rate expands every child the step it is born — it never sits
    # in the frontier to be bumped (measured at budget 4 on the cat scene:
    # the stable {motion, motion, position} trio regenerated 133 times and
    # still died at n_conf<=2, zero emissions). A budget below the arrival
    # rate lets children wait one step and collect the bump.
    builder = Builder(
        concept_graph=cg,
        fertility_engine=fertility,
        prediction_module=prediction,
        config=BuilderConfig(
            builder_expansions_per_step=int(builder_expansions_per_step)
        ),
    )
    binder = Binder(concept_graph=cg, bootstrap_relax_orthogonality=False)

    track_nodes: Dict[int, set] = {}
    track_sigs: Dict[int, list] = {}
    # Running-max bookkeeping (per anchor): confirmation accumulation is a
    # PEAK property — a scaffold candidate that reaches n_conf=12 and is
    # later consumed by expansion/crystallization still proves the anchor
    # accumulated. A final-state snapshot would systematically under-report
    # exactly the success signal mode 5 asks for.
    anchor_conf_peak: Dict[Any, int] = {}

    def _record_conf_peaks() -> None:
        for cand in builder.get_frontier():
            n_conf = int(getattr(cand, "n_conf", 0))
            for anchor, _step in getattr(cand, "anchors", ()):
                if n_conf > anchor_conf_peak.get(anchor, 0):
                    anchor_conf_peak[anchor] = n_conf
        for cand in cg.candidates.values():
            for anchor in (cand.Anchors or set()):
                if int(cand.N_conf) > anchor_conf_peak.get(anchor, 0):
                    anchor_conf_peak[anchor] = int(cand.N_conf)
        for concept in cg.concepts.values():
            for anchor in (getattr(concept, "Anchors", None) or set()):
                if int(concept.N_conf) > anchor_conf_peak.get(anchor, 0):
                    anchor_conf_peak[anchor] = int(concept.N_conf)

    # [Phase 3] Pass 1: clip-global consensus motion signature per track.
    # The ensemble frame is computed over the clip-aggregated tracks
    # (snapshot.tracks), so the consensus — and therefore the node identity —
    # is independent of window boundaries and per-window velocity jitter.
    clip_tracks = tuple(sorted(getattr(snapshot, "tracks", ()), key=lambda tr: tr.track_id))
    identity_sigmas: Dict[int, TypedSignature] = {}
    consensus_sigmas: Dict[int, TypedSignature] = {}
    if clip_tracks:
        g_cos, g_sin, g_clip = observation_motion_frame(clip_tracks)
        for tr in clip_tracks:
            consensus = track_motion_signature(tr, g_cos, g_sin, g_clip)
            consensus_sigmas[tr.track_id] = consensus
            identity_sigmas[tr.track_id] = track_identity_signature(
                tr.track_id, consensus
            )

    for t, obs in enumerate(snapshot.observations):
        # 1. Pattern nodes + typed temporal edges (the bridge proper),
        #    with track-stable node identity where known.
        bridge_temporal_observation(
            obs,
            cg,
            id_base=0,
            t=t,
            identity_sigmas=identity_sigmas or None,
            node_signatures=consensus_sigmas or None,
        )

        tracks = sorted(obs.tracks, key=lambda tr: tr.track_id)
        if not tracks:
            continue
        # [mode-3 stability] Per-window signatures are evaluated in the
        # clip-global ensemble frame (g_cos/g_sin/g_clip from pass 1), not
        # in the per-window frame: a faster entity joining mid-clip would
        # otherwise re-normalize every speed bin window-to-window.
        frame_cos, frame_sin, clip_max = (
            (g_cos, g_sin, g_clip) if clip_tracks
            else observation_motion_frame(tuple(tracks))
        )

        # Ensemble position envelope for the Position-lens convention.
        last_pos: Dict[int, Tuple[float, float]] = {}
        for tr in tracks:
            last_pos[tr.track_id] = (
                tr.positions[-1] if tr.positions else (0.0, 0.0)
            )
        xs = [p[0] for p in last_pos.values()]
        ys = [p[1] for p in last_pos.values()]
        envelope = (
            sum(xs) / len(xs),
            max(xs) - min(xs),
            sum(ys) / len(ys),
            max(ys) - min(ys),
        )

        # 2. Detections: Motion + Position per track, anchored on track_id.
        detections: set = set()
        for tr in tracks:
            sigma = track_motion_signature(tr, frame_cos, frame_sin, clip_max)
            nid = cg.get_or_create_pattern_node(
                MOTION_LENS_TYPE,
                _node_signature_hash(
                    identity_sigmas.get(
                        tr.track_id,
                        track_identity_signature(tr.track_id, sigma),
                    )
                ),
                signature=consensus_sigmas.get(tr.track_id, sigma),
            )
            track_nodes.setdefault(tr.track_id, set()).add(nid)
            track_sigs.setdefault(tr.track_id, []).append(tuple(sigma.quantized))
            speed = math.hypot(float(tr.velocity[0]), float(tr.velocity[1]))
            z = min(1.0, 0.3 + 0.7 * min(1.0, speed / max(clip_max, P.EPSILON)))
            # [IDENTITY SYNC] The detection MUST carry the same (lens_type,
            # signature_hash) the bridge used for the node: k=MOTION (the
            # bridge constant, not the still-image lens name) and sigma =
            # the OPAQUE track-identity signature. A canonical per-window
            # sigma here would route the Binder/Builder to a PARALLEL node
            # that never sees the bridge's SEQUENCE /
            # CO_MOTION edges — Rule-1 causality would be unreachable and
            # anchor identity would fork per stream.
            identity_sigma = identity_sigmas.get(
                tr.track_id, track_identity_signature(tr.track_id, sigma)
            )
            detections.add(
                Detection(
                    k=MOTION_LENS_TYPE,
                    source_layer=0,
                    scale=0,
                    sigma=identity_sigma,
                    alpha=tr.track_id,
                    z=z,
                    t=t,
                    rho=(("entity", tr.track_id, 0, "+"),),
                )
            )
            # Position detections are emitted for EVERY track. Two measured
            # roles, both load-bearing:
            #   1. the static prop's position node anchors the stable
            #      causal lineage ({motion, motion, position_prop} — the
            #      trio whose member set, and therefore canonical hash,
            #      recurs across windows so n_conf can accumulate);
            #   2. the MOVERS' churning position nodes keep the frontier
            #      footprints changing, which is the §4.11.1 unlock signal
            #      that lets transposition-memoized candidates re-derive.
            #      A static-only position set removes the churn and the
            #      whole seam starves (measured: synthetic conf 12 -> 4,
            #      zero emissions).
            detections.add(
                Detection(
                    k="Position",
                    source_layer=0,
                    scale=0,
                    sigma=_position_signature(last_pos[tr.track_id], envelope),
                    alpha=tr.track_id,
                    z=0.8,
                    t=t,
                    rho=(("entity", tr.track_id, 0, "+"),),
                )
            )

        # 3. Canonical per-step ordering (pipeline.py phases 3-5).
        s_t = hash_64(
            serialize_tuple(
                [serialize_int(int(obs.window_start)), serialize_int(len(tracks))]
                + [serialize_int(int(tr.track_id)) for tr in tracks]
            )
        )
        cg.set_current_state_hash(s_t)
        binder.step(t, frozenset(detections))
        builder_output = builder.step(t, detections, s_t)
        cross_sections = list(builder_output.cross_sections)
        cg.step_21_update_candidate_stats(cross_sections, t)
        cg.step_23_candidate_forgetting_and_capacity()
        cg.step_24_tombstone_expiry()
        cg.step_25_evaluate_crystallization(
            t=t,
            activation_engine=activation,
            active_plans=[],
            s_t=s_t,
            action_t=None,
        )
        cg.step_28_update_edges(activation, t)
        cg.step_28_update_witness_and_status(t)
        _record_conf_peaks()

    # --- diag: the five-mode observables ---------------------------------
    max_nodes_per_track = max((len(v) for v in track_nodes.values()), default=0)

    drift = 0
    for sigs in track_sigs.values():
        for a, b in zip(sigs, sigs[1:]):
            drift = max(drift, int(sum(abs(x - y) for x, y in zip(a, b))))

    _record_conf_peaks()  # final sweep (post-last-step state)

    diag: Dict[str, Any] = {
        "max_nodes_per_track": max_nodes_per_track,
        "max_conf_per_anchor": (
            max(anchor_conf_peak.values()) if anchor_conf_peak else None
        ),
        "max_signature_drift_bins": drift,
        "n_observations": len(snapshot.observations),
        "n_tracks": len(snapshot.tracks),
        "crystallized": len(cg.concepts),
        "families": sorted({c.family.name for c in cg.concepts.values()}),
    }
    return cg, diag
