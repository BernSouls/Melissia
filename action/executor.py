"""
executor.py — Search Function 4: The Executor (v31 §7).

Given a target consequence subgraph (from an action-query transfer), find
the sequence of actions that realizes it from the current state.

This is the ONLY place where pathfinding survives. Reasoning functions
(Builder, Consolidator, Mapper) have no target. The Executor has one.

Algorithm: A* over the action-edge graph, with an optional pattern-count
heuristic when the environment declares monotonicity. Macro-edges (§7.5)
are added for action cores whose precondition matches a state.

Determinism: every comparison, tie-break, and ordering is by canonical
hash (bytes, lexicographic) or by canonical serialized bytes. Same input
sequence produces byte-identical output, run to run.

Dependencies:
  - primitives: serialize_int, serialize_tuple, hash_128
  - sensory_floor: TypedSignature
  - prediction: Subgraph
  - consolidator: CoreKind, CoreLibrary, CorePayload_Action,
                  SignatureProvider, mcs_score
  - pipeline_support: decode_action_sequence (B2 chunk decoding)
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import parameters as P

from core.hashing import (  # noqa: F401
    hash_128,
    serialize_int,
    serialize_tuple,
)

from sensory_floor import TypedSignature
from core.graph_types import Subgraph

from consolidator import (
    CoreEntry,
    CoreKind,
    CoreLibrary,
    CorePayload_Action,
    SignatureProvider,
    _subgraph_size,
    mcs_score,
)

# [B2] Chunk-action decoding goes through the Action-lens registry
# (exact action objects), never through the former node-ID topological
# order — node IDs are not actions, so every macro-edge simulation died
# on predict(s, node_id) = None and the Executor permanently fell back
# to primitive A*.
from orchestration.pipeline import decode_action_sequence


# =============================================================================
# PARAMETERS (§7.8)
# =============================================================================

N_EDGE: int = P.N_EDGE
EDGE_COST: float = P.EDGE_COST
MASS_TIEBREAKER: bool = True
MAX_EXPANSIONS: int = P.MAX_EXPANSIONS
THETA_TARGET_CONTAINMENT: float = P.THETA_TARGET_CONTAINMENT

_INF: float = float("inf")

# Internal edge-kind tags used in came_from payloads.
_EDGE_PRIMITIVE: int = 0
_EDGE_MACRO: int = 1


# =============================================================================
# DATA STRUCTURES (§7.7)
# =============================================================================


@dataclass
class ActionSequence:
    """§7.7: an ordered sequence of primitive actions achieving a target.

    Field semantics:
      - actions: primitive action descriptors, in order.
      - reported_cost: number of A* edges walked (primitive + macro).
      - primitive_cost: number of primitive actions executed.
      - chunks_used: canonical hashes of action cores expanded.
      - expansion_bound_hit: kept for §7.7 interface compliance. In v31,
        find_actions returns None on bound hit (per §7.6), so a returned
        ActionSequence always carries expansion_bound_hit = False. Do not
        rely on this field being True.
    """

    actions: List[Any] = field(default_factory=list)
    reported_cost: int = 0
    primitive_cost: int = 0
    chunks_used: List[bytes] = field(default_factory=list)
    expansion_bound_hit: bool = False


class _MacroEdgeInfo:
    """Internal record for one applicable macro-edge from a state.

    Caches the primitive action sequence and its accumulated mass so that
    (a) heuristic scaling can bound target-pattern overlap, and (b) path
    reconstruction never needs a library lookup.
    """

    __slots__ = (
        "chunk_hash",
        "chunk_actions",
        "s_pred",
        "macro_mass",
        "target_overlap",
    )

    def __init__(
        self,
        chunk_hash: bytes,
        chunk_actions: List[Any],
        s_pred: int,
        macro_mass: float,
        target_overlap: int,
    ) -> None:
        self.chunk_hash = chunk_hash
        self.chunk_actions = chunk_actions
        self.s_pred = s_pred
        self.macro_mass = macro_mass
        self.target_overlap = target_overlap


class _StateCache:
    """Per-state evaluation cache used within a single find_actions call."""

    __slots__ = ("patterns", "sigs", "target_ok", "applicable")

    def __init__(
        self,
        patterns: Optional[Subgraph],
        sigs: Dict[int, Tuple[str, TypedSignature]],
        target_ok: bool,
        applicable: List[_MacroEdgeInfo],
    ) -> None:
        self.patterns = patterns
        self.sigs = sigs
        self.target_ok = target_ok
        self.applicable = applicable


# =============================================================================
# EXECUTOR
# =============================================================================


class Executor:
    """A* over the action-edge graph (§7.1)."""

    def __init__(
        self,
        concept_graph: Any,
        signature_provider: SignatureProvider,
        state_pattern_provider: Callable[[int], Optional[Subgraph]],
        is_monotonic_fn: Optional[Callable[[], bool]] = None,
        action_extractor: Optional[
            Callable[[Subgraph, SignatureProvider], List[Any]]
        ] = None,
    ) -> None:
        self.cg = concept_graph
        self._sig_provider = signature_provider
        self._state_pattern_provider = state_pattern_provider
        # [SPEC COMPLIANCE §7.4] The environment must declare monotonicity
        # dynamically; we query it once per find_actions call so that a
        # mid-episode change to is_monotonic() is honored.
        self._is_monotonic_fn = is_monotonic_fn
        self._action_extractor = (
            action_extractor
            if action_extractor is not None
            else self._default_chunk_extractor
        )

        self._sig_cache: Dict[int, Optional[Tuple[str, TypedSignature]]] = {}

        self.diagnostics: Dict[str, Any] = {
            "find_actions_calls": 0,
            "paths_found": 0,
            "paths_not_found": 0,
            "expansion_bound_hits": 0,
            "macro_edges_added": 0,
            "states_expanded": 0,
            "mass_lookup_failures": 0,
            "reconstruction_failures": 0,
            # [MACRO-TRANSFER DIAGNOSTIC]
            "find_actions_none_returned": 0,
            "find_actions_primitive_only": 0,
            "find_actions_macro_edges_used": 0,
        }

    # -------------------------------------------------------------------------
    # Public API (§7.7)
    # -------------------------------------------------------------------------

    def find_actions(
        self,
        t: int,
        consequence_subgraph: Subgraph,
        action_edge_map: Any,
        current_state: int,
        core_library: Optional[CoreLibrary] = None,
    ) -> Optional[ActionSequence]:
        """§7.7: sequence actions to realize consequence_subgraph.

        Returns ActionSequence, or None if unreachable within
        MAX_EXPANSIONS. `t` is currently diagnostic; no state depends on it
        within the search.
        """
        _ = t  # reserved for future time-dependent behavior
        self.diagnostics["find_actions_calls"] += 1
        self._sig_cache.clear()

        # Empty consequence is trivially satisfied by no action.
        if len(consequence_subgraph.nodes) == 0:
            self.diagnostics["paths_found"] += 1
            return ActionSequence()

        monotonic = bool(
            self._is_monotonic_fn() if self._is_monotonic_fn is not None else False
        )

        # Build the outgoing-edge index once.
        confirmed = list(action_edge_map.get_confirmed_edges())
        outgoing: Dict[int, List[Tuple[Any, int]]] = defaultdict(list)
        for s, a, ns in confirmed:
            outgoing[s].append((a, ns))

        target_nodes: Set[int] = set(consequence_subgraph.nodes)
        target_size = len(target_nodes)
        cs_sigs = self._collect_signatures(consequence_subgraph)

        # Per-call state cache: patterns + sigs + target-ok + applicable macros.
        state_cache: Dict[int, _StateCache] = {}

        def get_state_cache(state: int) -> _StateCache:
            cached = state_cache.get(state)
            if cached is not None:
                return cached
            patterns = self._state_pattern_provider(state)
            if patterns is None:
                sigs: Dict[int, Tuple[str, TypedSignature]] = {}
                cache = _StateCache(
                    patterns=None,
                    sigs=sigs,
                    target_ok=False,
                    applicable=[],
                )
                state_cache[state] = cache
                return cache
            sigs = self._collect_signatures(patterns)
            # [SPEC COMPLIANCE §6.4.1] MCS normalizes by min(|Q|, |C|), so
            # when the state's pattern subgraph is a strict subset of the
            # target, every one of the state's nodes matches and the score
            # saturates at 1.0 — a false positive. Guard the direction
            # first: the target cannot possibly be contained in a state
            # whose structural budget (nodes + edges) is smaller.
            # [SPEC COMPLIANCE §6.4.1] target_ok uses the full node+edge
            # size for its guard: the target is what we want to *achieve*,
            # so the state must fully exhibit the target's structure
            # (nodes AND edges) to be considered satisfied. This is
            # stricter than the macro-precondition guard in
            # _compute_applicable_macros, which is node-only because
            # precondition edges are action-order metadata, not state
            # properties. The asymmetry is intentional.
            if _subgraph_size(consequence_subgraph) > _subgraph_size(patterns):
                tok = False
            else:
                mcs_tok, _ = mcs_score(
                    consequence_subgraph, patterns, cs_sigs, sigs
                )
                tok = mcs_tok >= THETA_TARGET_CONTAINMENT
            # Applicable macros, if the library is present.
            applicable: List[_MacroEdgeInfo] = []
            if core_library is not None and sigs:
                applicable = self._compute_applicable_macros(
                    core_library,
                    patterns,
                    sigs,
                    state,
                    action_edge_map,
                    target_nodes,
                )
            cache = _StateCache(
                patterns=patterns,
                sigs=sigs,
                target_ok=tok,
                applicable=applicable,
            )
            state_cache[state] = cache
            return cache

        # Compute the heuristic scaling bound once per call.
        # [SPEC COMPLIANCE §7.4 + §7.5] The pattern-count heuristic assumes
        # each edge adds at most 1 target pattern. A macro-edge can add up
        # to K target patterns in a single edge (K = max target-node overlap
        # across any library core's consequence_subgraph, if any). Dividing
        # h by max(1, K_max) restores admissibility while preserving signal.
        max_target_overlap = 1
        if core_library is not None:
            for entry in core_library.entries:
                if entry.kind != CoreKind.ACTION:
                    continue
                payload = entry.payload
                if not isinstance(payload, CorePayload_Action):
                    continue
                overlap = len(payload.consequence_subgraph.nodes & target_nodes)
                if overlap > max_target_overlap:
                    max_target_overlap = overlap
        inv_scale = 1.0 / float(max_target_overlap)

        def heuristic(state: int) -> float:
            if not monotonic:
                return 0.0
            cache = get_state_cache(state)
            if cache.patterns is None:
                h_raw = float(target_size)
            else:
                present = target_nodes & cache.patterns.nodes
                h_raw = float(max(0, target_size - len(present)))
            return h_raw * inv_scale

        def target_ok(state: int) -> bool:
            return get_state_cache(state).target_ok

        # Trivial case: current state already satisfies the target.
        if target_ok(current_state):
            self.diagnostics["paths_found"] += 1
            return ActionSequence()

        # A* initialization.
        g_score: Dict[int, float] = {current_state: 0.0}
        path_mass: Dict[int, float] = {current_state: 0.0}
        # came_from payloads:
        #   primitive: (parent, _EDGE_PRIMITIVE, action)
        #   macro:     (parent, _EDGE_MACRO, _MacroEdgeInfo)
        came_from: Dict[int, Tuple[int, int, Any]] = {}
        closed: Set[int] = set()
        # Heap tuple: (f, -path_mass, serialized_state, g, state).
        # The g field is used for lazy-deletion on pop. serialized_state
        # ensures canonical-byte tie-breaking per §9.
        h0 = heuristic(current_state)
        open_heap: List[Tuple[float, float, bytes, float, int]] = [
            (h0, 0.0, serialize_int(current_state), 0.0, current_state)
        ]
        expansions = 0

        while open_heap:
            if expansions >= MAX_EXPANSIONS:
                self.diagnostics["expansion_bound_hits"] += 1
                self.diagnostics["paths_not_found"] += 1
                self.diagnostics["find_actions_none_returned"] += 1
                return None

            f, neg_mass, _ser, popped_g, state = heapq.heappop(open_heap)

            # [SPEC COMPLIANCE §7.6 / standard A*] Lazy-deletion guard:
            # heapq lacks decrease-key, so stale entries may be present.
            # Skip if a better g for this state has been recorded.
            if state in closed:
                continue
            if popped_g > g_score.get(state, _INF):
                continue
            closed.add(state)
            expansions += 1
            self.diagnostics["states_expanded"] += 1

            if target_ok(state):
                seq = self._reconstruct_path(came_from, state, core_library)
                if seq is None:
                    self.diagnostics["reconstruction_failures"] += 1
                    self.diagnostics["paths_not_found"] += 1
                    self.diagnostics["find_actions_none_returned"] += 1
                    return None
                self.diagnostics["paths_found"] += 1
                if seq.chunks_used:
                    self.diagnostics["find_actions_macro_edges_used"] += 1
                else:
                    self.diagnostics["find_actions_primitive_only"] += 1
                return seq

            # Primitive edges.
            for action, next_state in outgoing.get(state, []):
                if next_state in closed:
                    continue
                new_g = g_score[state] + EDGE_COST
                if new_g < g_score.get(next_state, _INF):
                    edge_mass = self._edge_mass(action_edge_map, state, action)
                    g_score[next_state] = new_g
                    path_mass[next_state] = path_mass[state] + edge_mass
                    came_from[next_state] = (state, _EDGE_PRIMITIVE, action)
                    new_f = new_g + heuristic(next_state)
                    heapq.heappush(
                        open_heap,
                        (
                            new_f,
                            -path_mass[next_state],
                            serialize_int(next_state),
                            new_g,
                            next_state,
                        ),
                    )

            # Macro-edges (from the state cache).
            macros = get_state_cache(state).applicable
            for macro in macros:
                s_pred = macro.s_pred
                if s_pred == state or s_pred in closed:
                    continue
                new_g = g_score[state] + EDGE_COST
                if new_g < g_score.get(s_pred, _INF):
                    g_score[s_pred] = new_g
                    # [SPEC COMPLIANCE §7.3 + §7.5] Macro-edge mass
                    # accumulates the primitive masses it replaces. A
                    # macro-edge composed of well-confirmed primitives
                    # carries higher mass, matching §7.5's intent that
                    # consolidated chunks be preferred.
                    path_mass[s_pred] = path_mass[state] + macro.macro_mass
                    came_from[s_pred] = (state, _EDGE_MACRO, macro)
                    new_f = new_g + heuristic(s_pred)
                    heapq.heappush(
                        open_heap,
                        (
                            new_f,
                            -path_mass[s_pred],
                            serialize_int(s_pred),
                            new_g,
                            s_pred,
                        ),
                    )
                    self.diagnostics["macro_edges_added"] += 1

        self.diagnostics["paths_not_found"] += 1
        self.diagnostics["find_actions_none_returned"] += 1
        return None

    def get_path_cost(
        self,
        consequence_subgraph: Subgraph,
        action_edge_map: Any,
        current_state: int,
        core_library: Optional[CoreLibrary] = None,
    ) -> float:
        """§7.7: return the reported_cost of the optimal path, or inf.

        [SPEC COMPLIANCE §7.7] The signature matches §7.7 and does not
        include `t`. `find_actions`'s `t` is currently unused by the
        search (diagnostic only), so we pass a sentinel. If a future
        revision introduces time-dependent search state, this method
        must gain a `t` parameter.
        """
        result = self.find_actions(
            -1,
            consequence_subgraph,
            action_edge_map,
            current_state,
            core_library,
        )
        if result is None:
            return _INF
        return float(result.reported_cost)

    # -------------------------------------------------------------------------
    # Macro-edge computation (§7.5)
    # -------------------------------------------------------------------------

    def _compute_applicable_macros(
        self,
        core_library: CoreLibrary,
        patterns: Subgraph,
        patterns_sigs: Dict[int, Tuple[str, TypedSignature]],
        state: int,
        action_edge_map: Any,
        target_nodes: Set[int],
    ) -> List[_MacroEdgeInfo]:
        """Return the applicable macro-edges from `state`.

        For each ACTION core whose core_subgraph matches the state's
        patterns above θ_target_containment, simulate the chunk's primitive
        actions to obtain s_pred and the accumulated primitive mass.
        """
        if not patterns_sigs:
            return []
        state_lens = {info[0] for info in patterns_sigs.values()}
        candidate_idxs = core_library.get_candidates_for_query(
            state_lens, CoreKind.ACTION
        )
        applicable: List[_MacroEdgeInfo] = []
        for idx in candidate_idxs:
            entry = core_library.entries[idx]
            # [SPEC COMPLIANCE §6.4.1 + §7.5] Guard against the
            # min()-normalization pitfall: when the state's patterns have
            # fewer *nodes* than the precondition, MCS saturates at 1.0
            # even though the state cannot contain the precondition. The
            # guard is deliberately node-count only, not node-plus-edge:
            # the precondition's SEQUENCE edges encode action-ordering
            # metadata for the chunk, not claims about the state's
            # edge structure. A state whose patterns carry the required
            # nodes is a valid macro trigger regardless of whether its
            # induced edge set matches the chunk's action order.
            if len(entry.core_subgraph.nodes) > len(patterns.nodes):
                continue
            mcs, _ = mcs_score(
                entry.core_subgraph,
                patterns,
                entry.core_node_signatures,
                patterns_sigs,
            )
            if mcs < THETA_TARGET_CONTAINMENT:
                continue

            chunk_actions = self._action_extractor(
                entry.core_subgraph, self._sig_lookup
            )
            if not chunk_actions:
                continue

            s_pred, macro_mass = self._simulate_chunk(
                chunk_actions, state, action_edge_map
            )
            if s_pred is None or s_pred == state:
                continue

            payload = entry.payload
            if isinstance(payload, CorePayload_Action):
                overlap = len(payload.consequence_subgraph.nodes & target_nodes)
            else:
                overlap = 0

            applicable.append(
                _MacroEdgeInfo(
                    chunk_hash=entry.canonical_hash,
                    chunk_actions=list(chunk_actions),
                    s_pred=s_pred,
                    macro_mass=macro_mass,
                    target_overlap=overlap,
                )
            )
        return applicable

    def _simulate_chunk(
        self,
        chunk_actions: List[Any],
        start_state: int,
        action_edge_map: Any,
    ) -> Tuple[Optional[int], float]:
        """Simulate the chunk's primitive actions from start_state.

        Returns (s_pred, accumulated_mass) on success, (None, 0.0) if any
        transition is undefined.
        """
        s = start_state
        mass = 0.0
        for a in chunk_actions:
            s_next = self._predict_single(action_edge_map, s, a)
            if s_next is None:
                return None, 0.0
            mass += self._edge_mass(action_edge_map, s, a)
            s = s_next
        return s, mass

    # -------------------------------------------------------------------------
    # Path reconstruction (§7.7)
    # -------------------------------------------------------------------------

    def _reconstruct_path(
        self,
        came_from: Dict[int, Tuple[int, int, Any]],
        goal_state: int,
        core_library: Optional[CoreLibrary],
    ) -> Optional[ActionSequence]:
        """Reconstruct the action sequence from the A* parent tree.

        Macro-edge actions are cached in the came_from payload as
        `_MacroEdgeInfo`, so this method never performs a library lookup.
        If that invariant is ever violated (payload is not `_MacroEdgeInfo`
        or its `chunk_actions` is empty), the reconstruction fails loudly
        rather than silently producing a partial sequence.
        """
        _ = core_library  # retained in signature for caller convenience
        path: List[Tuple[int, int, Any, int]] = []
        state = goal_state
        visited: Set[int] = set()
        while state in came_from:
            if state in visited:
                # Defensive: A* parent trees are acyclic; this is unreachable.
                return None
            visited.add(state)
            parent, kind, payload = came_from[state]
            path.append((parent, kind, payload, state))
            state = parent
        path.reverse()

        actions: List[Any] = []
        chunks_used: List[bytes] = []
        reported_cost = len(path)
        primitive_cost = 0

        for _parent, kind, payload, _child in path:
            if kind == _EDGE_PRIMITIVE:
                actions.append(payload)
                primitive_cost += 1
                continue

            # Macro-edge.
            if not isinstance(payload, _MacroEdgeInfo):
                return None
            if not payload.chunk_actions:
                return None
            chunks_used.append(payload.chunk_hash)
            actions.extend(payload.chunk_actions)
            primitive_cost += len(payload.chunk_actions)

        return ActionSequence(
            actions=actions,
            reported_cost=reported_cost,
            primitive_cost=primitive_cost,
            chunks_used=chunks_used,
            expansion_bound_hit=False,
        )

    # -------------------------------------------------------------------------
    # Transition prediction (§5.3.4)
    # -------------------------------------------------------------------------

    def _predict_single(
        self,
        action_edge_map: Any,
        state: int,
        action: Any,
    ) -> Optional[int]:
        """Predict the successor state of (state, action).

        Prefers ActionEdgeMap.predict (§5.3.4). Falls back to get_outcomes
        (deterministic argmax by count, tie-broken by serialized state).
        Last-resort linear scan aggregates counts across all confirmed
        edges matching (state, action) before argmax, so the result does
        not depend on get_confirmed_edges iteration order.
        """
        predict_fn = getattr(action_edge_map, "predict", None)
        if predict_fn is not None:
            return predict_fn(state, action)

        outcomes_fn = getattr(action_edge_map, "get_outcomes", None)
        if outcomes_fn is not None:
            outcomes = outcomes_fn(state, action)
            if not outcomes:
                return None
            return min(
                outcomes.items(),
                key=lambda kv: (-kv[1], serialize_int(kv[0])),
            )[0]

        # Last resort: linear scan with aggregated counts.
        counts: Dict[int, int] = defaultdict(int)
        for s, a, ns in action_edge_map.get_confirmed_edges():
            if s == state and a == action:
                counts[ns] += 1
        if not counts:
            return None
        return min(
            counts.items(),
            key=lambda kv: (-kv[1], serialize_int(kv[0])),
        )[0]

    def _edge_mass(
        self,
        action_edge_map: Any,
        state: int,
        action: Any,
    ) -> float:
        """Return the mass of (state, action), or 0.0 on failure.

        Failures are counted in diagnostics so a degraded mass tiebreaker
        is detectable rather than silent.
        """
        get_mass = getattr(action_edge_map, "get_mass", None)
        if get_mass is not None:
            try:
                return float(get_mass(state, action))
            except Exception:
                self.diagnostics["mass_lookup_failures"] += 1
                return 0.0
        get_outcomes = getattr(action_edge_map, "get_outcomes", None)
        if get_outcomes is not None:
            try:
                outcomes = get_outcomes(state, action)
                if outcomes:
                    return float(sum(outcomes.values()))
            except Exception:
                self.diagnostics["mass_lookup_failures"] += 1
                return 0.0
        return 0.0

    # -------------------------------------------------------------------------
    # Signature helpers
    # -------------------------------------------------------------------------

    def _default_chunk_extractor(
        self, G: Subgraph, sig_provider: SignatureProvider
    ) -> List[Any]:
        """[B2] Default chunk-action decoder: Action-lens registry lookup
        in strict SEQUENCE-topological order (pipeline_support.
        decode_action_sequence, bound to this Executor's concept graph).

        Returns [] when the core is undecodable (cycle, missing registry
        entry, or a legacy core with no Action-lens members) — an empty
        chunk is skipped by _compute_applicable_macros, which is the
        honest outcome. The former default returned node IDs in
        topological order; no AEM is keyed by node IDs, so every macro
        simulation returned None and macro-edges were dead by
        construction (blockers.md B2).
        """
        decoded = decode_action_sequence(G, self.cg)
        return decoded if decoded is not None else []

    def _sig_lookup(self, nid: int) -> Optional[Tuple[str, TypedSignature]]:
        if nid in self._sig_cache:
            return self._sig_cache[nid]
        info = self._sig_provider(nid)
        self._sig_cache[nid] = info
        return info

    def _collect_signatures(self, G: Subgraph) -> Dict[int, Tuple[str, TypedSignature]]:
        result: Dict[int, Tuple[str, TypedSignature]] = {}
        for nid in G.nodes:
            info = self._sig_lookup(nid)
            if info is not None:
                result[nid] = info
        return result

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        return dict(self.diagnostics)

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear episode-local state. Cumulative diagnostics are preserved
        for pipeline inspection."""
        self._sig_cache.clear()
