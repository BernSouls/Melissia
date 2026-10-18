"""perception/lenses/derived.py -- the derived-lens aggregator (Move 3.5).

    Tiers live in derived_base / v2 / v4 / s4; shared constants in
    derived_common. Their public names are re-exported here, so the
    legacy `from derived_lenses import X` surface is unchanged.
    """
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple
import parameters as P
from concept_graph import (  # noqa: F401
    ARITY_MAX,
    Concept,
    ConceptStatus,
    EPSILON,
    Edge,
    KAPPA_BIRTH,
    TAU_EDGE_INACTIVE,
    TAU_OP_COOLDOWN,
    TAU_REVERT,
    THETA_EDGE_ACTIVE,
    THETA_HYST_OP,
)
from core.hashing import (  # noqa: F401
    _detection_sort_key,
    clamp,
    hash_64,
    serialize_bytes,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
    serialize_value,
)
from core.types import (  # noqa: F401
    Detection,
    EdgeType,
    TypedSignature,
)
from learning.fertility import (  # noqa: F401
    KAPPA_EMA,
    RF_MAX,
)
from memory.cores import (  # noqa: F401
    _sig_similarity,
)
from parameters._text import (  # noqa: F401
    B_MAX,
    DELTA_CAT,
    DELTA_THETA_CAT_MAX,
    KAPPA_CAT,
    K_CATEGORIZE,
    RHO_CAT_TARGET,
    RHO_PROGRESS_MIN,
    TAU_PROGRESS,
    THETA_CATEGORIZE,
    THETA_CATEGORIZE_MAX,
    THETA_CATEGORIZE_MIN,
    THETA_SIM,
    W_CHUNK,
    W_PLAN_CHUNK,
)

class DerivedLenses:
    def __init__(self) -> None:
        # Recognition lens state.
        self.theta_categorize = THETA_CATEGORIZE
        self.r_bar_cat = 0.0
        self.M_rec_prev: Dict[int, int] = {}

        # Chunk emission lens state.
        self.theta_chunk_emit = THETA_CHUNK_EMIT
        self.M_chunk_prev: Dict[bytes, int] = {}

        # Operator emission lens state.
        self.operator_state: Dict[int, Dict[str, Any]] = {}
        self.birth_ema: Dict[int, float] = {}
        self.prev_fert_norm_op: Dict[int, float] = {}

        # Constitutional-exception monitoring for the recognition lens.
        self._ever_recognized: Set[int] = set()
        self._recognition_drift_count: int = 0

        # [PERFORMANCE] Per-concept recognition evidence index cache:
        # (concept id, registry length) → (exact-sig set, compat groups).
        # Evidence registries only grow, so the length is the exact
        # invalidation signal. Bounded (cleared past cap).
        self._recog_evidence_cache: Dict[Tuple[int, int], Tuple[Set, Dict]] = {}

        # [PERFORMANCE] Step-scoped active-edge pair cache (§1.12
        # predicate computed in one pass; see _active_pair_set).
        self._active_cache_key: Optional[Tuple] = None
        self._active_cache_pairs: Set[Tuple[int, int]] = set()

        # §2.8.1.2 Binding Layer Robustness Contract monitoring. Counts
        # cross-sections whose family-specific key constructor matched no
        # recognized role-name pattern and were rescued by the generic
        # value-based fallback key, plus those discarded even after the
        # fallback. Silent discard is architectural drift, so both are made
        # observable here instead of vanishing without a trace.
        self._binding_fallback_key_count: int = 0
        self._binding_discard_count: int = 0
        self._binding_discard_log: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Core evidence subset for operator promotion (§D.3.1)
    # -------------------------------------------------------------------------

    def _get_core_evidence(self, concept: Concept) -> List[Tuple]:
        if not concept.evidence_registry:
            return []
        layer_evid: Dict[int, List[Tuple]] = {}
        for e in concept.evidence_registry:
            layer_evid.setdefault(e[3], []).append(e)
        best_layer = None
        for m in sorted(layer_evid.keys()):
            if len(layer_evid[m]) >= 2:
                best_layer = m
                break
        if best_layer is None:
            return []
        evid = layer_evid[best_layer]
        r_P = min(len(evid), ARITY_MAX)
        if r_P < 2:
            return []
        return evid[:r_P]

    # -------------------------------------------------------------------------
    # Public pipeline-facing step methods
    # -------------------------------------------------------------------------

    def step_13_recognition(
        self, D_raw: List[Detection], concepts: Dict[int, Concept], t: int
    ) -> List[Detection]:
        return self._recognition(D_raw, concepts, t)

    def step_14_operator_emission(
        self,
        D_base: List[Detection],
        concepts: Dict[int, Concept],
        edges: Dict[Tuple[int, int, EdgeType], Edge],
        active_concepts: Set[int],
        t: int,
        ef_eff_proposal: Optional[Dict[int, float]] = None,
        rf: Optional[Dict[int, float]] = None,
    ) -> List[Detection]:
        return self._operators(
            D_base, concepts, edges, active_concepts, t, ef_eff_proposal, rf
        )

    def step_15_chunk_candidates(
        self,
        active_plans: Any,
        concepts: Dict[int, Concept],
        edges: Dict[Tuple[int, int, EdgeType], Edge],
        coact_ema: Dict[Tuple[int, int, EdgeType], float],
        time_since_coact: Dict[Tuple[int, int, EdgeType], int],
        active_concepts: Set[int],
        t: int,
    ) -> List[Tuple[Set[int], Set[Tuple[int, int]]]]:
        return self._chunk_candidates(
            active_plans,
            concepts,
            edges,
            coact_ema,
            time_since_coact,
            active_concepts,
            t,
        )

    def step_16_chunk_emission(
        self,
        candidates: List[Tuple[Set[int], Set[Tuple[int, int]]]],
        concepts: Dict[int, Concept],
        edges: Dict[Tuple[int, int, EdgeType], Edge],
        coact_ema: Dict[Tuple[int, int, EdgeType], float],
        t: int,
        context_sig: int = 0,
        active_plans: Any = None,
    ) -> List[Detection]:
        return self._chunk_emission(
            candidates, concepts, edges, coact_ema, t, context_sig, active_plans
        )

    # -------------------------------------------------------------------------
    # Active-edge predicate (§1.12)
    # -------------------------------------------------------------------------

    def _active_pair_set(self, edges: Dict, t: Optional[int]) -> Set[Tuple[int, int]]:
        """[PERFORMANCE — identical predicate] The set of (parent, child)
        pairs having ANY edge type above the §1.12 active predicate,
        computed in ONE pass over the edge store and cached per
        (t, edges-identity, size). The former per-call implementation ran
        12 EdgeType dict lookups per query; _chunk_candidates queries it
        once per EDGE (measured: ~100k dict gets + enum hashes per step
        on rich graphs — the dominant hot loop). Derived-lens steps all
        run in Phase 4, before the step's edge writes, so the intra-step
        cache can never observe a mutated store."""
        key = (t, id(edges), len(edges))
        if self._active_cache_key != key:
            pairs: Set[Tuple[int, int]] = set()
            for (p, c, _et), e in edges.items():
                if (
                    e.weight >= THETA_EDGE_ACTIVE
                    and e.inactive_time < TAU_EDGE_INACTIVE
                ):
                    pairs.add((p, c))
            self._active_cache_key = key
            self._active_cache_pairs = pairs
        return self._active_cache_pairs

    def _is_active_edge(self, p: int, c: int, edges: Dict, t: Optional[int]) -> bool:
        """True if any edge type between (p, c) is above threshold and not
        past the inactivity timeout (§1.12)."""
        return (p, c) in self._active_pair_set(edges, t)

    # -------------------------------------------------------------------------
    # Recognition lens (§2.31)
    # -------------------------------------------------------------------------

    def _recognition_evidence_index(self, c: Concept) -> Tuple[Set, Dict]:
        """(exact layer-0 signature set, compatibility-grouped signatures)
        for one concept, cached per (concept id, registry length) — the
        registry only grows, so its length is the exact invalidation
        signal. Pure read-side index; semantics unchanged."""
        key = (c.id, len(c.evidence_registry))
        cached = self._recog_evidence_cache.get(key)
        if cached is not None:
            return cached
        sig_set: Set[TypedSignature] = set()
        groups: Dict[Optional[Tuple], List[TypedSignature]] = {}
        for (_idx, sig, _fam, layer, _scale, _anchor) in c.evidence_registry:
            if layer != 0:
                continue
            sig_set.add(sig)
            gk = _compat_key(sig)
            if gk is not None:
                groups.setdefault(gk, []).append(sig)
        if len(self._recog_evidence_cache) > 4096:
            self._recog_evidence_cache.clear()
        self._recog_evidence_cache[key] = (sig_set, groups)
        return (sig_set, groups)

    def _recognition(
        self,
        D_raw: List[Detection],
        concepts: Dict[int, Concept],
        t: int,
    ) -> List[Detection]:
        # CONSTITUTIONAL EXCEPTION (documented deviation from §2.31.1):
        # §2.31.1 specifies EligCat(P,t) = 1 iff status(P) = STABLE. In
        # continuous environments this bootstrap-deadlocks: a CRYSTALLIZED
        # concept reaches STABLE only via witness evidence, which requires
        # exact-match binding, which signature bin drift prevents. Extended
        # eligibility includes CRYSTALLIZED. This does NOT relax binding
        # evidence (z_bind stays exact) and does NOT alter evidence-registry
        # semantics. Monitoring via _recognition_drift_count.
        #
        # [PERFORMANCE — semantics-preserving] best_evidence(d) depends
        # ONLY on d.sigma, and max_score = max_d z_d · best_evidence(σ_d)
        # with z ≥ 0 — so the scan collapses to DISTINCT detection
        # signatures carrying their max z, an exact-membership fast path
        # (score 1.0 is the maximum; the graded loop cannot beat it), and
        # per-concept evidence signatures pre-grouped by compatibility
        # key (kind, class, relation, arity): an OPAQUE or off-group
        # detection signature skips the graded scan entirely instead of
        # paying a TypedSignature comparison per evidence entry. The
        # former triple loop (concepts × detections × evidence) measured
        # ~11k signature comparisons/step and dominated the loop once
        # spatial channels revived the pattern population.
        if not D_raw:
            return []
        max_z_by_sigma: Dict[TypedSignature, float] = {}
        for d in D_raw:
            prev_z = max_z_by_sigma.get(d.sigma)
            if prev_z is None or d.z > prev_z:
                max_z_by_sigma[d.sigma] = d.z

        eligible: List[Tuple[int, float, Concept]] = []
        for pid, c in concepts.items():
            if c.status not in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE):
                continue
            if c.dormant or c.soft_dormant or c.provisional:
                continue
            sig_set, groups = self._recognition_evidence_index(c)
            if not sig_set:
                continue
            max_score = 0.0
            for dsig, z in max_z_by_sigma.items():
                if z <= 0.0:
                    continue
                if dsig in sig_set:
                    best_evidence = 1.0
                else:
                    best_evidence = 0.0
                    group = groups.get(_compat_key(dsig))
                    if group:
                        for sig in group:
                            score = _sim_memo(dsig, sig)
                            if score > best_evidence:
                                best_evidence = score
                score = z * best_evidence
                if score > max_score:
                    max_score = score
            eligible.append((pid, clamp(max_score, 0, 1), c))
        if not eligible:
            return []

        selected: List[Tuple[int, float, Concept]] = []
        for pid, score, c in eligible:
            prev = self.M_rec_prev.get(pid, 0)
            if t == 0:
                m = 1 if score >= self.theta_categorize else 0
            else:
                m = (
                    1
                    if score >= self.theta_categorize + DELTA_CAT * (1 - 2 * prev)
                    else 0
                )
            if m == 1:
                selected.append((pid, score, c))
        selected.sort(key=lambda x: (-x[1], hash_64(x[2].sigma.serialize())))
        topk = selected[:K_CATEGORIZE]
        for pid, _score, _c in topk:
            self._ever_recognized.add(pid)

        detections: List[Detection] = []
        for pid, score, c in topk:
            # [PERFORMANCE — identical semantics] Per-distinct-signature
            # best pairing via the cached evidence index, then a single
            # D_raw-ordered pass. The original double loop evaluated the
            # same (detection-signature × registry) pairing for every
            # detection sharing a signature; the tie key
            # (exact + sim, hash_64(serialize)) is a pure function of the
            # signature PAIR, and equal ties are order-insensitive (equal
            # hash ⇒ equal canonical bytes ⇒ the same signature), so the
            # winner — including the first-detection-wins rule on exact
            # tie values across D_raw order — is preserved.
            sig_set, groups = self._recognition_evidence_index(c)
            best_for_sigma: Dict[
                TypedSignature, Optional[Tuple[Tuple[float, bytes], TypedSignature]]
            ] = {}
            best_tie: Optional[Tuple[float, bytes]] = None
            best_d: Optional[Detection] = None
            best_ev: Optional[TypedSignature] = None
            for d in D_raw:
                dsig = d.sigma
                if dsig in best_for_sigma:
                    entry = best_for_sigma[dsig]
                else:
                    entry = None
                    sig_tie: Optional[Tuple[float, bytes]] = None
                    sig_ev: Optional[TypedSignature] = None
                    if dsig in sig_set:
                        # Exact match: similarity(dsig, dsig) = 1.0 and
                        # equal signatures serialize identically, so the
                        # tie key may be computed against dsig itself
                        # (exact=1.0 + sim=1.0 = 2.0).
                        sig_tie = (2.0, hash_64(dsig.serialize()))
                        sig_ev = dsig
                    gk = _compat_key(dsig)
                    if gk is not None:
                        for sig_obj in groups.get(gk, ()):
                            sim = _sim_memo(dsig, sig_obj)
                            tie_s = (sim, hash_64(sig_obj.serialize()))
                            if sig_tie is None or tie_s > sig_tie:
                                sig_tie = tie_s
                                sig_ev = sig_obj
                    if sig_ev is not None:
                        entry = (sig_tie, sig_ev)
                    best_for_sigma[dsig] = entry
                if entry is None:
                    continue
                tie, ev = entry
                if best_tie is None or tie > best_tie:
                    best_tie = tie
                    best_d = d
                    best_ev = ev
            if best_d is None or best_ev is None:
                continue
            rho = (("recognizedConcept", pid, 0, "+"), ("evidence", best_ev, 0, "+"))
            detections.append(
                Detection(
                    k="Recognition",
                    source_layer=0,
                    scale=0,
                    sigma=best_ev,
                    alpha=best_d.alpha,
                    z=score,
                    t=t,
                    rho=rho,
                )
            )

        new_M_rec_prev: Dict[int, int] = {}
        for pid, score, _c in eligible:
            prev = self.M_rec_prev.get(pid, 0)
            if t == 0:
                m = 1 if score >= self.theta_categorize else 0
            else:
                m = (
                    1
                    if score >= self.theta_categorize + DELTA_CAT * (1 - 2 * prev)
                    else 0
                )
            new_M_rec_prev[pid] = m
        self.M_rec_prev = new_M_rec_prev
        self._update_recognition_homeostasis(eligible, t)
        return detections

    def _update_recognition_homeostasis(
        self, eligible: List[Tuple[int, float, Concept]], t: int
    ) -> None:
        N_elig = len(eligible)
        if N_elig == 0:
            return
        emitted_count = sum(
            1 for _pid, score, _c in eligible if score >= self.theta_categorize
        )
        r_cat = emitted_count / (N_elig + EPSILON)
        self.r_bar_cat = (1 - KAPPA_EMA) * self.r_bar_cat + KAPPA_EMA * r_cat
        delta = clamp(
            KAPPA_CAT * (self.r_bar_cat - RHO_CAT_TARGET),
            -DELTA_THETA_CAT_MAX,
            DELTA_THETA_CAT_MAX,
        )
        self.theta_categorize = clamp(
            self.theta_categorize + delta,
            THETA_CATEGORIZE_MIN,
            THETA_CATEGORIZE_MAX,
        )

    def recognize_raw(
        self,
        D_raw: List[Detection],
        concepts: Dict[int, Concept],
        t: int,
        min_similarity: float = 0.9,
    ) -> Set[Detection]:
        """Read-only optimization: raw detections already covered by a
        STABLE concept's evidence at similarity >= min_similarity.
        Does not emit or mutate."""
        matched: Set[Detection] = set()
        for _pid, c in concepts.items():
            if c.status != ConceptStatus.STABLE:
                continue
            if c.dormant or c.soft_dormant or c.provisional:
                continue
            sigma_0 = [
                sig
                for (_idx, sig, _fam, layer, _scale, _anchor) in c.evidence_registry
                if layer == 0
            ]
            if not sigma_0:
                continue
            for d in D_raw:
                if d in matched:
                    continue
                best_sim = 0.0
                for sig in sigma_0:
                    if d.sigma == sig:
                        sim = 1.0
                    elif _sig_compatible(d.sigma, sig):
                        sim = _sig_similarity(d.sigma, sig)
                    else:
                        sim = 0.0
                    best_sim = max(best_sim, sim)
                    if best_sim >= 1.0:
                        break
                if best_sim >= min_similarity:
                    matched.add(d)
        return matched

    # -------------------------------------------------------------------------
    # Operator emission lens (§2.28)
    # -------------------------------------------------------------------------

    def _operators(
        self,
        D_base: List[Detection],
        concepts: Dict[int, Concept],
        edges: Dict[Tuple[int, int, EdgeType], Edge],
        active_concepts: Set[int],
        t: int,
        ef_eff_proposal: Optional[Dict[int, float]] = None,
        rf: Optional[Dict[int, float]] = None,
    ) -> List[Detection]:
        detections: List[Detection] = []
        # A demoted or rejected operator must not retain its emission
        # homeostasis indefinitely (nor recover stale state on promotion).
        active_ids = {
            pid for pid, c in concepts.items()
            if c.promoted and c.status not in
            (ConceptStatus.REJECTED, ConceptStatus.TOMBSTONED)
        }
        for stale_pid in sorted(self.operator_state.keys() - active_ids):
            del self.operator_state[stale_pid]
            self.prev_fert_norm_op.pop(stale_pid, None)
        promoted = [pid for pid in concepts if pid in active_ids]
        if not promoted:
            return detections

        for op_pid in promoted:
            op_concept = concepts[op_pid]
            state = self.operator_state.setdefault(
                op_pid,
                {"theta": THETA_HYST_OP, "F": 0.0, "C": 0.0, "M_prev": {}},
            )
            state["last_emissions"] = 0
            evid = self._get_core_evidence(op_concept)
            if not evid:
                continue
            m_op = evid[0][3]
            r_P = len(evid)
            D_base_m = [
                d
                for d in D_base
                if d.source_layer == m_op and d.sigma.type_tag != "OPERATOR_PATTERN"
            ]
            if len(D_base_m) < r_P:
                continue

            det_cache: Dict[Detection, Tuple[Tuple, int]] = {}
            for d in D_base_m:
                substrate_ids = sorted(
                    [role[1] for role in d.rho if isinstance(role[1], int)]
                )
                sub_hash = (
                    hash_64(serialize_set([serialize_int(i) for i in substrate_ids]))
                    if substrate_ids
                    else 0
                )
                info_src = (d.k, d.source_layer, d.scale, sub_hash)
                sig_h = (
                    d.sigma.hash_value
                    if d.sigma.kind == "OPAQUE"
                    else hash_64(d.sigma.serialize())
                )
                det_cache[d] = (info_src, sig_h)

            valid_instances = []
            cache_items = sorted(
                [(d, det_cache[d]) for d in D_base_m],
                key=lambda pair: _detection_sort_key(pair[0]),
            )
            for combo_t in combinations(cache_items, r_P):
                combo = tuple(d for d, _ in combo_t)
                sources = {v[0] for _, v in combo_t}
                hashes = {v[1] for _, v in combo_t}
                if len(sources) >= 2 and len(hashes) >= 2:
                    key, anchor = self._family_key_anchor(
                        op_concept.family, combo, concepts, m_op, op_concept, t
                    )
                    if key is not None and anchor is not None:
                        valid_instances.append((combo, key, anchor))
                        if len(valid_instances) >= P.OPERATOR_MAX_INSTANCES:
                            break

            unit_scores: Dict[Tuple, float] = {}
            unit_best_R: Dict[Tuple, Tuple[Detection, ...]] = {}
            for R, key, anchor in valid_instances:
                y = self._max_min_match(R, evid)
                if y < THETA_SIM:
                    y = 0.0
                if y == 0:
                    continue
                u = (key, anchor)
                if u not in unit_scores or y > unit_scores[u]:
                    unit_scores[u] = y
                    unit_best_R[u] = R
            if not unit_scores:
                continue

            Y_max = max(max(unit_scores.values()), Y_FLOOR_OP, EPSILON)
            theta_eff = clamp(
                state["theta"] - ETA_CTX_OP * state["C"] + ETA_FAT_OP * state["F"],
                THETA_FLOOR_OP,
                1 - EPSILON,
            )
            M_prev = state["M_prev"]
            selected_units: List[Tuple[Tuple, float, float]] = []
            for u, y in unit_scores.items():
                y_norm = y / Y_max
                z = clamp((y_norm - theta_eff) / (1 - theta_eff + EPSILON), 0, 1)
                prev = M_prev.get(u, 0)
                m = 1 if z >= THETA_HYST_OP + DELTA_OP * (1 - 2 * prev) else 0
                if m == 1:
                    selected_units.append((u, z, y_norm))
            selected_units.sort(
                key=lambda x: (
                    -x[1],
                    hash_64(
                        serialize_tuple(
                            [
                                self._serialize_alpha(x[0][0]),
                                self._serialize_alpha(x[0][1]),
                            ]
                        )
                    ),
                )
            )
            topk = selected_units[:K_OP]
            state["last_emissions"] = len(topk)

            for u, z, _y_norm in topk:
                _key, anchor = u
                R_best = unit_best_R[u]
                evid_op_serialized = serialize_set(
                    [
                        serialize_tuple(
                            [
                                serialize_int(e[0]),
                                e[1].serialize(),
                                serialize_bytes(e[2].encode()),
                                serialize_int(e[3]),
                                serialize_int(e[4]),
                                serialize_bytes(e[5].encode()),
                            ]
                        )
                        for e in evid
                    ]
                )
                O_P = hash_64(
                    serialize_tuple(
                        [
                            serialize_bytes(b"EMERGENT_OPERATOR"),
                            op_concept.sigma.serialize(),
                            serialize_int(op_concept.layer),
                            serialize_int(m_op),
                            evid_op_serialized,
                        ]
                    )
                )
                source_ids: Set[int] = set()
                for d in R_best:
                    for role in d.rho:
                        if isinstance(role[1], int):
                            source_ids.add(role[1])

                rho: List[Tuple] = [("operator", O_P, 0, "+")]
                for i, q in enumerate(
                    sorted(source_ids, key=lambda x: concepts[x].sigma.serialize())
                ):
                    rho.append(("substrateSource", q, i, "+"))
                if m_op == 0 and not source_ids:
                    rho.append(("entity", anchor, 0, "+"))

                sigma = TypedSignature.opaque(
                    "OPERATOR_PATTERN",
                    hash_64(
                        serialize_tuple(
                            [
                                serialize_int(O_P),
                                serialize_int(m_op),
                                serialize_int(0),
                            ]
                        )
                    ),
                )
                detections.append(
                    Detection(
                        k=f"L_{op_pid}",
                        source_layer=m_op,
                        scale=0,
                        sigma=sigma,
                        alpha=anchor,
                        z=z,
                        t=t,
                        rho=tuple(rho),
                    )
                )

            freeze = (
                op_concept.dormant
                or op_concept.soft_dormant
                or op_concept.status
                in (ConceptStatus.REJECTED, ConceptStatus.TOMBSTONED)
                or len(unit_scores) == 0
            )
            if not freeze:
                r_op = len(topk) / (min(K_OP, len(unit_scores)) + EPSILON)
                state["theta"] = clamp(
                    state["theta"] + KAPPA_OP * (r_op - RHO_OP_TARGET),
                    THETA_FLOOR_OP,
                    1 - EPSILON,
                )
            state["F"] = ETA_FAT_DECAY_OP * state["F"] + (1 - ETA_FAT_DECAY_OP) * (
                len(topk) / (len(unit_scores) + EPSILON)
            )
            if ef_eff_proposal is not None:
                prev_fert = self.prev_fert_norm_op.get(op_pid, 0.0)
                state["C"] = prev_fert
                current_fert = math.log(
                    1 + ef_eff_proposal.get(op_pid, 0.0)
                ) / math.log(1 + RF_MAX + EPSILON)
                self.prev_fert_norm_op[op_pid] = current_fert
            else:
                state["C"] = self.prev_fert_norm_op.get(op_pid, 0.0)

            M_new: Dict[Tuple, int] = {}
            for u in unit_scores:
                prev = M_prev.get(u, 0)
                y_norm = unit_scores[u] / Y_max
                z = clamp((y_norm - theta_eff) / (1 - theta_eff + EPSILON), 0, 1)
                m = 1 if z >= THETA_HYST_OP + DELTA_OP * (1 - 2 * prev) else 0
                M_new[u] = m
            state["M_prev"] = M_new
        return detections

    def _max_min_match(self, R: Tuple[Detection, ...], evid: List[Tuple]) -> float:
        scores: List[List[float]] = []
        for d in R:
            row = []
            for e in evid:
                if d.source_layer != e[3] or d.scale != e[4] or d.k != e[2]:
                    score = 0.0
                elif d.sigma.serialize() == e[1].serialize():
                    score = 1.0
                elif _sig_compatible(d.sigma, e[1]):
                    score = _sig_similarity(d.sigma, e[1])
                else:
                    score = 0.0
                row.append(score)
            scores.append(row)
        unique_scores = sorted(
            {s for row in scores for s in row if s > 0}, reverse=True
        )
        for threshold in unique_scores:
            if self._bipartite_match_possible(scores, threshold):
                return threshold
        return 0.0

    def _bipartite_match_possible(
        self, scores: List[List[float]], threshold: float
    ) -> bool:
        r = len(scores)
        if r == 0:
            return True
        c = len(scores[0])
        if c != r:
            raise ValueError(
                f"_bipartite_match_possible requires square matrix, got {r}x{c}"
            )
        match_to_evid = [-1] * r

        def dfs(det: int, seen: List[bool]) -> bool:
            for ev in range(r):
                if scores[det][ev] >= threshold and not seen[ev]:
                    seen[ev] = True
                    if match_to_evid[ev] == -1 or dfs(match_to_evid[ev], seen):
                        match_to_evid[ev] = det
                        return True
            return False

        for det in range(r):
            seen = [False] * r
            if not dfs(det, seen):
                return False
        return True

    def _family_key_anchor(
        self,
        family: Any,
        R: Tuple[Detection, ...],
        concepts: Dict[int, Concept],
        m_op: int,
        op_concept: Concept,
        t: int,
    ) -> Tuple[Optional[Any], Optional[int]]:
        """§2.8.1.1 family-specific key constructor with the §2.8.1.2
        generic fallback layered on top.

        The family-specific constructor keys on role NAMES, which the Role
        Grammar (§2.7.1) defines for the canonical lenses. But §2.8.1.2 is
        explicit: when no recognized role-name pattern matches, the key MUST
        fall back to one derived from role VALUES only (names, indices and
        polarity ignored), and a cross-section MUST NOT be silently
        discarded. The `_recursive_key()` incident — role names
        rejected for 70+ steps — is exactly the failure
        this wrapper prevents.

        Returns (key, anchor). Only a detection set with no integer role
        values AND no serializable anchors can fail to produce a key; that
        case is counted and logged, never dropped silently.
        """
        key, anchor = self._family_key_anchor_specific(
            family, R, concepts, m_op, op_concept, t
        )
        if key is not None and anchor is not None:
            return key, anchor

        fb_key, fb_anchor = self._generic_fallback_key_anchor(R)
        if fb_key is not None and fb_anchor is not None:
            self._binding_fallback_key_count += 1
            return fb_key, fb_anchor

        # Even the value-based fallback found nothing to key on. Record the
        # reason (§2.8.1.2: "discarded with a logged reason including the
        # fallback key and the set of lens types involved") instead of
        # vanishing.
        self._binding_discard_count += 1
        if len(self._binding_discard_log) < 64:
            self._binding_discard_log.append(
                {
                    "family": getattr(family, "name", family),
                    "lens_types": sorted({d.k for d in R}),
                    "arity": len(R),
                }
            )
        return None, None

    def _generic_fallback_key_anchor(
        self, R: Tuple[Detection, ...]
    ) -> Tuple[Optional[Any], Optional[int]]:
        """§2.8.1.2 K_fallback: a key from role VALUES only.

            K_fallback(C) = hash_64(serialize(
                sorted({ serialize(value)
                         for d in C
                         for (roleName, value, index, polarity) in rho_d })))

        roleName, index and polarity are ignored by construction, so two
        detections referencing the same underlying entities collide on the
        same key regardless of what names their source lenses assigned. The
        anchor reuses the family-agnostic anchor hash (sorted serialized
        detection anchors) that every specific branch already agrees on.
        """
        values: List[bytes] = []
        for d in R:
            for role in d.rho:
                if len(role) < 2:
                    continue
                values.append(serialize_value(role[1]))
        if not values:
            return None, None
        key = hash_64(serialize_set(sorted(set(values))))
        anchor = hash_64(
            serialize_set([self._serialize_alpha(d.alpha) for d in R])
        )
        return key, anchor

    def _family_key_anchor_specific(
        self,
        family: Any,
        R: Tuple[Detection, ...],
        concepts: Dict[int, Concept],
        m_op: int,
        op_concept: Concept,
        t: int,
    ) -> Tuple[Optional[Any], Optional[int]]:
        # op_concept and t are part of the call-site contract; the key is a
        # pure function of (family, R), so they are intentionally unread.
        fam = getattr(family, "name", family) if hasattr(family, "name") else family
        if fam == "SPATIAL_OBJECT":
            if len({d.alpha for d in R}) == 1:
                anchor = next(iter(R)).alpha
                return anchor, anchor
            return None, None
        elif fam in ("EVENT_CAUSAL", "TEMPORAL_CHAIN"):
            if m_op == 0:
                seq_name, struct_name = "Sequence", "Structure"
            else:
                seq_name, struct_name = "ConceptSequence", "ConceptStructure"
            seq = [d for d in R if d.k == seq_name]
            struct = [d for d in R if d.k == struct_name]
            if len(seq) == 1 and len(struct) == 1:
                key = (seq[0].sigma, struct[0].sigma)
                anchor = hash_64(
                    serialize_set([self._serialize_alpha(d.alpha) for d in R])
                )
                return key, anchor
            return None, None
        elif fam == "RECURSIVE_COMPOSITION":
            parent_ids: Set[int] = set()
            child_ids: Set[int] = set()
            for d in R:
                for role in d.rho:
                    if role[0] in ("parent", "antecedent"):
                        parent_ids.add(role[1])
                    elif role[0] in ("child", "consequent"):
                        child_ids.add(role[1])
            if parent_ids or child_ids:
                e = hash_64(serialize_set([self._serialize_alpha(d.alpha) for d in R]))
                if parent_ids:
                    key = (frozenset(parent_ids), e)
                else:
                    key = (frozenset(child_ids), e)
                anchor = hash_64(
                    serialize_set([self._serialize_alpha(d.alpha) for d in R])
                )
                return key, anchor
            return None, None
        elif fam == "CROSS_DOMAIN_PATTERN":
            k_set = {d.k for d in R}
            if len(k_set) == 1:
                k: Any = next(iter(k_set))
            else:
                k = tuple(sorted(k_set))
            substrate_hashes: List[int] = []
            for d in R:
                substrate_ids = sorted(
                    [role[1] for role in d.rho if isinstance(role[1], int)]
                )
                sub_hash = (
                    hash_64(serialize_set([serialize_int(i) for i in substrate_ids]))
                    if substrate_ids
                    else 0
                )
                substrate_hashes.append(sub_hash)
            substrate_hashes_t = tuple(sorted(set(substrate_hashes)))
            key = (k, substrate_hashes_t)
            anchor = hash_64(serialize_set([self._serialize_alpha(d.alpha) for d in R]))
            return key, anchor
        else:
            key = hash_64(serialize_set([d.sigma.serialize() for d in R]))
            anchor = hash_64(serialize_set([self._serialize_alpha(d.alpha) for d in R]))
            return key, anchor

    def _serialize_alpha(self, alpha: Any) -> bytes:
        if isinstance(alpha, int):
            return serialize_int(alpha)
        elif isinstance(alpha, float):
            return serialize_float(alpha)
        elif isinstance(alpha, str):
            return serialize_bytes(alpha.encode())
        elif isinstance(alpha, bytes):
            return serialize_bytes(alpha)
        elif isinstance(alpha, TypedSignature):
            return alpha.serialize()
        elif isinstance(alpha, tuple):
            return serialize_tuple([self._serialize_alpha(x) for x in alpha])
        elif isinstance(alpha, (frozenset, set)):
            return serialize_set(
                [self._serialize_alpha(x) for x in sorted(alpha, key=str)]
            )
        else:
            return serialize_bytes(str(alpha).encode())

    # -------------------------------------------------------------------------
    # Chunk emission lens (§5.10)
    # -------------------------------------------------------------------------

    def _chunk_candidates(
        self,
        active_plans: Any,
        concepts: Dict[int, Concept],
        edges: Dict[Tuple[int, int, EdgeType], Edge],
        coact_ema: Dict[Tuple[int, int, EdgeType], float],
        time_since_coact: Dict[Tuple[int, int, EdgeType], int],
        active_concepts: Set[int],
        t: int,
    ) -> List[Tuple[Set[int], Set[Tuple[int, int]]]]:
        # Co-activation is defined over CO_ACTIVATION edges (§11.4).
        coact_pair: Dict[Tuple[int, int], float] = {}
        for (p, c, et), v in coact_ema.items():
            if et == EdgeType.CO_ACTIVATION:
                coact_pair[(p, c)] = v
        tsc_pair: Dict[Tuple[int, int], int] = {}
        for (p, c, et), v in time_since_coact.items():
            if et == EdgeType.CO_ACTIVATION:
                tsc_pair[(p, c)] = v

        candidates: List[Tuple[float, Set[int], Set[Tuple[int, int]], int]] = []
        active_neighbors: Dict[int, Set[int]] = {}
        for u, v, _et in edges:
            if self._is_active_edge(u, v, edges, t):
                active_neighbors.setdefault(u, set()).add(v)
                active_neighbors.setdefault(v, set()).add(u)

        plan_candidates: List[Tuple[float, Set[int], Set[Tuple[int, int]], int]] = []
        for plan in (
            active_plans.values() if isinstance(active_plans, dict) else active_plans
        ):
            if t - getattr(plan, "last_active", t) > W_PLAN_CHUNK:
                continue
            B_pi = plan.basis_snapshot
            B_elig = {
                p
                for p in B_pi
                if isinstance(p, int)
                and p in concepts
                and (
                    p in active_concepts
                    or (p in active_neighbors and active_neighbors[p])
                )
            }
            if len(B_elig) >= 2:
                for size in (2, 3):
                    for subset in combinations(B_elig, size):
                        V_sub = set(subset)
                        if self._is_connected(V_sub, edges):
                            E_sub = self._induced_edges(V_sub, edges)
                            score = self._preliminary_score(V_sub, E_sub, coact_pair)
                            key = hash_64(
                                serialize_set(
                                    [concepts[p].sigma.serialize() for p in V_sub]
                                )
                            )
                            plan_candidates.append((score, V_sub, E_sub, key))
        plan_candidates.sort(key=lambda x: (-x[0], x[3]))
        plan_candidates = plan_candidates[:K_CHUNK_PLAN]
        candidates.extend(plan_candidates)

        star_candidates: List[Tuple[float, Set[int], Set[Tuple[int, int]], int]] = []
        for (p, c), coact in coact_pair.items():
            if coact < THETA_EDGE_ACTIVE:
                continue
            if tsc_pair.get((p, c), 999) > W_CHUNK:
                continue
            # Skip any pair whose endpoints are not concept ids. The Binder
            # also writes CO_ACTIVATION edges between pattern nodes; those
            # are not concepts.
            if p not in concepts or c not in concepts:
                continue

            neighbors = set(active_neighbors.get(p, set())) - {c}
            neighbor_list = [
                (q, coact_pair.get((p, q), 0)) for q in neighbors if q in concepts
            ]
            neighbor_list.sort(key=lambda x: -x[1])
            V_sub = {p, c}
            if neighbor_list:
                V_sub.add(neighbor_list[0][0])
            if len(V_sub) >= 2:
                E_sub = self._induced_edges(V_sub, edges)
                score = self._preliminary_score(V_sub, E_sub, coact_pair)
                key = hash_64(
                    serialize_set([concepts[q].sigma.serialize() for q in V_sub])
                )
                star_candidates.append((score, V_sub, E_sub, key))

            neighbors = set(active_neighbors.get(c, set())) - {p}
            neighbor_list = [
                (q, coact_pair.get((c, q), 0)) for q in neighbors if q in concepts
            ]
            neighbor_list.sort(key=lambda x: -x[1])
            V_sub = {c, p}
            if neighbor_list:
                V_sub.add(neighbor_list[0][0])
            if len(V_sub) >= 2:
                E_sub = self._induced_edges(V_sub, edges)
                score = self._preliminary_score(V_sub, E_sub, coact_pair)
                key = hash_64(
                    serialize_set([concepts[q].sigma.serialize() for q in V_sub])
                )
                star_candidates.append((score, V_sub, E_sub, key))

        star_candidates.sort(key=lambda x: (-x[0], x[3]))
        star_candidates = star_candidates[:K_CHUNK_STAR]
        candidates.extend(star_candidates)

        candidates.sort(key=lambda x: (-x[0], x[3]))
        # [v8] Dedup by canonical key so the plan-candidate loop and the
        # star-candidate loop cannot emit the identical V_sub twice,
        # wasting K_CHUNK_CAND slots.
        seen_keys: set = set()
        deduped: List[Tuple[float, Set[int], Set[Tuple[int, int]], int]] = []
        for entry in candidates:
            if entry[3] in seen_keys:
                continue
            seen_keys.add(entry[3])
            deduped.append(entry)
        candidates = deduped[:K_CHUNK_CAND]
        return [(V, E) for _score, V, E, _key in candidates]

    def _is_connected(
        self, V_sub: Set[int], edges: Dict[Tuple[int, int, EdgeType], Edge]
    ) -> bool:
        if len(V_sub) <= 1:
            return True
        start = next(iter(V_sub))
        visited = {start}
        queue = [start]
        while queue:
            p = queue.pop()
            for q in V_sub:
                if q in visited:
                    continue
                if self._is_active_edge(p, q, edges, None) or self._is_active_edge(
                    q, p, edges, None
                ):
                    visited.add(q)
                    queue.append(q)
        return len(visited) == len(V_sub)

    def _induced_edges(
        self, V_sub: Set[int], edges: Dict[Tuple[int, int, EdgeType], Edge]
    ) -> Set[Tuple[int, int]]:
        E: Set[Tuple[int, int]] = set()
        for p in V_sub:
            for q in V_sub:
                if p < q and self._is_active_edge(p, q, edges, None):
                    E.add((p, q))
        return E

    def _preliminary_score(
        self,
        V_sub: Set[int],
        E_sub: Set[Tuple[int, int]],
        coact_ema: Dict[Tuple[int, int], float],
    ) -> float:
        if not E_sub:
            return 0.0
        total = sum(coact_ema.get((p, q), 0) for p, q in E_sub)
        return total / len(E_sub)

    def _chunk_emission(
        self,
        candidates: List[Tuple[Set[int], Set[Tuple[int, int]]]],
        concepts: Dict[int, Concept],
        edges: Dict[Tuple[int, int, EdgeType], Edge],
        coact_ema: Dict[Tuple[int, int, EdgeType], float],
        t: int,
        C_t: int,
        active_plans: Any = None,
    ) -> List[Detection]:
        coact_pair = {
            (p, c): v
            for (p, c, et), v in coact_ema.items()
            if et == EdgeType.CO_ACTIVATION
        }
        detections: List[Detection] = []
        scored: List[Tuple[float, Set[int], Set[Tuple[int, int]], int]] = []
        for V_sub, E_sub in candidates:
            if not self._eligible_chunk(V_sub, E_sub, concepts, edges, t):
                continue
            score = self._chunk_score(V_sub, E_sub, coact_pair, t, active_plans)
            key = hash_64(serialize_set([concepts[p].sigma.serialize() for p in V_sub]))
            scored.append((score, V_sub, E_sub, key))
        if not scored:
            return []

        M_prev = self.M_chunk_prev
        selected: List[Tuple[float, Set[int], Set[Tuple[int, int]], int]] = []
        for score, V_sub, E_sub, key in scored:
            prev = M_prev.get(key, 0)
            if t == 0:
                m = 1 if score >= self.theta_chunk_emit else 0
            else:
                m = (
                    1
                    if score >= self.theta_chunk_emit + DELTA_CHUNK * (1 - 2 * prev)
                    else 0
                )
            if m == 1:
                selected.append((score, V_sub, E_sub, key))
        selected.sort(key=lambda x: (-x[0], x[3]))
        topk = selected[:K_CHUNK_CAND]
        for score, V_sub, _E_sub, key in topk:
            m_sub = max(concepts[p].layer for p in V_sub)
            sigma = TypedSignature.opaque(
                "CHUNK",
                hash_64(
                    serialize_tuple(
                        [
                            serialize_bytes(b"CHUNK"),
                            serialize_set(
                                [concepts[p].sigma.serialize() for p in V_sub]
                            ),
                            serialize_int(m_sub + 1),
                        ]
                    )
                ),
            )
            alpha = hash_64(serialize_tuple([serialize_int(key), serialize_int(C_t)]))
            rho: List[Tuple] = []
            sorted_members = sorted(V_sub, key=lambda p: concepts[p].sigma.serialize())
            for i, p in enumerate(sorted_members):
                rho.append(("member", p, i, "+"))
            rho.append(("chunkKey", key, 0, "+"))
            detections.append(
                Detection(
                    k="Chunk",
                    source_layer=m_sub,
                    scale=0,
                    sigma=sigma,
                    alpha=alpha,
                    z=score,
                    t=t,
                    rho=tuple(rho),
                )
            )

        M_new: Dict[int, int] = {}
        for score, _V_sub, _E_sub, key in scored:
            prev = M_prev.get(key, 0)
            m = (
                1
                if score >= self.theta_chunk_emit + DELTA_CHUNK * (1 - 2 * prev)
                else 0
            )
            M_new[key] = m
        self.M_chunk_prev = M_new

        N_active = len(scored)
        if N_active > 0:
            emitted = len(topk)
            r_chunk = emitted / min(K_CHUNK_CAND, N_active)
            delta = KAPPA_CHUNK_HOMEO * (r_chunk - RHO_CHUNK_TARGET)
            self.theta_chunk_emit = clamp(
                self.theta_chunk_emit + delta,
                THETA_CHUNK_EMIT_MIN,
                THETA_CHUNK_EMIT_MAX,
            )
        return detections

    def _eligible_chunk(
        self,
        V_sub: Set[int],
        E_sub: Set[Tuple[int, int]],
        concepts: Dict[int, Concept],
        edges: Dict[Tuple[int, int, EdgeType], Edge],
        t: int,
    ) -> bool:
        if len(V_sub) < 2 or len(V_sub) > B_MAX:
            return False
        if len(E_sub) < 1:
            return False
        if not self._is_connected(V_sub, edges):
            return False
        for p in V_sub:
            c = concepts[p]
            if c.status not in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
            ):
                return False
            if c.dormant or c.soft_dormant or c.provisional:
                return False
        for p, q in E_sub:
            if not self._is_active_edge(p, q, edges, t):
                return False
        max_chunk_depth = 0
        for p in V_sub:
            fam = getattr(concepts[p].family, "name", concepts[p].family)
            if fam == "CHUNK":
                if concepts[p].chunk_depth > max_chunk_depth:
                    max_chunk_depth = concepts[p].chunk_depth
        if 1 + max_chunk_depth > D_CHUNK_MAX:
            return False
        return True

    def _chunk_score(
        self,
        V_sub: Set[int],
        E_sub: Set[Tuple[int, int]],
        coact_ema: Dict[Tuple[int, int], float],
        t: int,
        active_plans: Any = None,
    ) -> float:
        if not E_sub:
            return 0.0
        coact = sum(coact_ema.get((p, q), 0) for p, q in E_sub) / len(E_sub)
        reuse = 0.0
        if active_plans:
            plans = (
                active_plans.values()
                if isinstance(active_plans, dict)
                else active_plans
            )
            for plan in plans:
                if hasattr(plan, "basis_snapshot"):
                    reuse += len(V_sub & plan.basis_snapshot) / len(V_sub)
        reuse /= W_PLAN_CHUNK
        reuse_norm = reuse / (1 + reuse)
        return coact * (1 + reuse_norm) / 2.0

    # -------------------------------------------------------------------------
    # Promotion gating (§2.28.3-4)
    # -------------------------------------------------------------------------

    def compute_promotion_eligibility(
        self, concept: Concept, concepts: Dict[int, Concept], t: int
    ) -> bool:
        if concept.status != ConceptStatus.STABLE:
            return False
        if concept.promoted:
            return False
        if concept.dormant or concept.soft_dormant or concept.provisional:
            return False
        if t - concept.T_cryst < TAU_PROMO_MIN:
            return False
        if not concept.evidence_registry:
            return False
        evid_op = self._get_core_evidence(concept)
        if not evid_op:
            return False
        if len(evid_op) > SIGMA_MAX_OP:
            return False
        if concept.op_demoted_at != -1 and t < concept.op_demoted_at + TAU_OP_COOLDOWN:
            return False
        return True

    def compute_promotion_score(
        self,
        concept: Concept,
        concepts: Dict[int, Concept],
        t: int,
        EF_eff_proposal: float,
        RF: float,
    ) -> float:
        if not concept.evidence_registry:
            return 0.0
        FertNorm = math.log(1 + EF_eff_proposal) / math.log(1 + RF_MAX + EPSILON)
        core_evid = self._get_core_evidence(concept)
        BreadthNorm = min(1, len(core_evid) / ARITY_MAX)
        had_birth = any(
            c in concept.child_ids and concepts[c].T_cryst == t for c in concept.child_ids
        )
        b_P = self.birth_ema.get(concept.id, 0.0)
        b_P = (1 - KAPPA_BIRTH) * b_P + KAPPA_BIRTH * (1.0 if had_birth else 0.0)
        self.birth_ema[concept.id] = b_P
        FertileEnough = max(
            clamp(b_P / (RHO_BIRTH + EPSILON), 0, 1),
            clamp(RF / (RF_MAX * RHO_PROMO_RF + EPSILON), 0, 1),
        )
        return FertNorm * BreadthNorm * FertileEnough

    def compute_promotion_candidates(
        self,
        concepts: Dict[int, Concept],
        t: int,
        ef_eff_proposal: Dict[int, float],
        rf: Dict[int, float],
    ) -> List[Tuple[int, bool, float]]:
        candidates: List[Tuple[int, bool, float]] = []
        for cid, concept in concepts.items():
            if concept.status != ConceptStatus.STABLE:
                continue
            eligible = self.compute_promotion_eligibility(concept, concepts, t)
            if not eligible:
                candidates.append((cid, False, 0.0))
                continue
            score = self.compute_promotion_score(
                concept,
                concepts,
                t,
                ef_eff_proposal.get(cid, 0.0),
                rf.get(cid, 0.0),
            )
            candidates.append((cid, True, score))
        return candidates

    def get_operator_emissions(self, t: int) -> Dict[int, int]:
        emissions: Dict[int, int] = {}
        for op_pid, state in self.operator_state.items():
            emissions[op_pid] = state.get("last_emissions", 0)
        return emissions

    # -------------------------------------------------------------------------
    # Progress / recognition drift
    # -------------------------------------------------------------------------

    def compute_progress_failure(
        self, concept: Concept, concepts: Dict[int, Concept], t: int, FertNorm: float
    ) -> bool:
        last_prog = concept.last_structural_progress
        if t - last_prog <= TAU_PROGRESS:
            return False
        if FertNorm >= RHO_PROGRESS_MIN:
            return False
        return True

    def compute_fail_streak_update(
        self, concept: Concept, progress_failure: bool, fail_streak: int
    ) -> int:
        if not progress_failure:
            return 0
        return min(TAU_REVERT, fail_streak + 1)

    def update_recognition_drift(self, concepts: Dict[int, Any]) -> None:
        for cid in list(self._ever_recognized):
            c = concepts.get(cid)
            if c is not None and c.status == ConceptStatus.REJECTED:
                self._recognition_drift_count += 1
                self._ever_recognized.discard(cid)

    def get_recognition_drift_count(self) -> int:
        return self._recognition_drift_count



# --- tier re-exports (legacy surface) ---
from perception.lenses.derived_base import (  # noqa: F401
    ContourGainState,
    EndStoppedLens,
    HigherLens,
)

from perception.lenses.v2 import (  # noqa: F401
    BorderOwnershipLens,
    ContourLens,
    JunctionLens,
    TextureBoundaryLens,
)

from perception.lenses.v4 import (  # noqa: F401
    CurveComplexLens,
    CurveComplexLens_B13,
    HolisticShapeLens,
    IllusoryContourLens,
    RegionShapeLens,
    SurfaceLens,
)

from perception.lenses.s4 import (  # noqa: F401
    ShockEnsembleLens,
)

from perception.lenses.chain_lens import (  # noqa: F401  [A5]
    ChainLens,
)

from perception.lenses.derived_common import (  # noqa: F401
    ARITY_MAX,
    BO_COLOR_WEIGHT,
    BO_CUE_SCALE,
    BO_CURVATURE_WEIGHT,
    BO_GRAVITY_SCALE,
    BO_GRAVITY_WEIGHT,
    BO_RICHNESS_FALLBACK,
    BO_TEXTURE_BIN_MAX,
    BO_TEXTURE_WEIGHT,
    B_MAX,
    CURVATURE_TEMPLATES,
    DELTA_CAT,
    DELTA_CHUNK,
    DELTA_OP,
    DELTA_THETA_CAT_MAX,
    D_CHUNK_MAX,
    EPSILON,
    ETA_CTX_OP,
    ETA_FAT_DECAY_OP,
    ETA_FAT_OP,
    HIGHER_LENS_NAMES,
    KAPPA_BIRTH,
    KAPPA_CAT,
    KAPPA_CHUNK_HOMEO,
    KAPPA_EMA,
    KAPPA_OP,
    K_CATEGORIZE,
    K_CHUNK_CAND,
    K_CHUNK_PLAN,
    K_CHUNK_STAR,
    K_OP,
    RF_MAX,
    RHO_BIRTH,
    RHO_CAT_TARGET,
    RHO_CHUNK_TARGET,
    RHO_OP_TARGET,
    RHO_PROGRESS_MIN,
    RHO_PROMO_RF,
    SIGMA_MAX_OP,
    TAU_EDGE_INACTIVE,
    TAU_OP_COOLDOWN,
    TAU_PROGRESS,
    TAU_PROMO_MIN,
    TAU_REVERT,
    THETA_CATEGORIZE,
    THETA_CATEGORIZE_MAX,
    THETA_CATEGORIZE_MIN,
    THETA_CHUNK_EMIT,
    THETA_CHUNK_EMIT_MAX,
    THETA_CHUNK_EMIT_MIN,
    THETA_EDGE_ACTIVE,
    THETA_FLOOR_OP,
    THETA_HYST_OP,
    THETA_PROMO_OP,
    THETA_SIM,
    W_CHUNK,
    W_PLAN_CHUNK,
    Y_FLOOR_OP,
    _CONTOUR_RANGES,
    _JUNCTION_CANONICAL_BINS,
    _SIM_MEMO,
    _SIM_MEMO_CAP,
    _THETA_FRAME_SIGN,
    _TIE_MIN_COS,
    _TINY,
    _arm_tangent_stats,
    _canonical_curvature_sequence,
    _chamfer,
    _collect_continuity_neighbours,
    _compat_key,
    _contour_slots,
    _contour_turns,
    _envelope_cov,
    _oriented_tangents,
    _point_in_poly,
    _resample_closed,
    _sig_compatible,
    _sig_similarity,
    _signed_log_level,
    _sim_memo,
    _splat,
    _turn_positions,
    circ_mean_bin,
    compute_endstopped_map,
    contour_self_symmetry,
    diagnostic_derived_lenses,
    match_templates,
    qi,
    qi_circ,
    reflection_symmetry,
    sample_curvature_profile,
    texture_richness,
)
