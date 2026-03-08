"""Intelligent qubit subset selection based on live fidelity data.

Given a circuit's requirements (qubit count + connectivity), finds the optimal
physical qubit subset on the chip by solving a constrained subgraph matching problem.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum

from qpilot.characterization.noise_profile import ChipNoiseProfile

logger = logging.getLogger(__name__)


class ScoringMetric(StrEnum):
    """Available scoring metrics for qubit selection."""

    COMPOSITE_FIDELITY = "composite_fidelity"
    SINGLE_GATE = "single_gate"
    TWO_GATE_MIN = "two_gate_min"


@dataclass
class QubitMapping:
    """Mapping from logical circuit qubits to physical chip qubits."""

    mapping: dict[int, str] = field(default_factory=dict)  # logical → physical
    score: float = 0.0
    metric: str = ""

    @property
    def physical_qubits(self) -> list[str]:
        """List of selected physical qubit IDs."""
        return list(self.mapping.values())

    def physical(self, logical: int) -> str:
        """Get the physical qubit for a logical qubit index."""
        return self.mapping[logical]

    def __repr__(self) -> str:
        pairs = ", ".join(f"L{k}→{v}" for k, v in sorted(self.mapping.items()))
        return f"QubitMapping({pairs}, score={self.score:.4f})"


class QubitSelector:
    """Select optimal physical qubits for a circuit.

    Algorithm:
    1. Build chip topology graph from ChipNoiseProfile.
    2. If no connectivity required: return the N highest-fidelity qubits.
    3. If connectivity required: enumerate candidate subgraphs that satisfy
       the connectivity constraints, score each, return the best.

    For small circuits (<=10 qubits) on typical chip sizes (<100 qubits),
    brute-force enumeration is feasible. For larger circuits, a greedy
    heuristic is used.
    """

    BRUTE_FORCE_LIMIT = 10  # Max circuit qubits for exhaustive search

    def __init__(self, noise_profile: ChipNoiseProfile) -> None:
        self.profile = noise_profile

    def select(
        self,
        num_qubits: int,
        connectivity: list[tuple[int, int]] | None = None,
        metric: str = ScoringMetric.COMPOSITE_FIDELITY,
    ) -> QubitMapping:
        """Find optimal physical qubit subset.

        Args:
            num_qubits: Number of qubits the circuit needs.
            connectivity: List of (logical_i, logical_j) pairs that require
                         a two-qubit gate (must be physically connected).
                         If None, no connectivity constraints.
            metric: Scoring metric to use.

        Returns:
            QubitMapping with the best scoring assignment.

        Raises:
            ValueError: If no valid mapping exists.
        """
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if num_qubits > len(self.profile.profiles):
            raise ValueError(
                f"Requested {num_qubits} qubits but chip only has "
                f"{len(self.profile.profiles)} profiled"
            )

        if connectivity is None:
            return self._select_unconstrained(num_qubits, metric)
        return self._select_constrained(num_qubits, connectivity, metric)

    def _select_unconstrained(self, n: int, metric: str) -> QubitMapping:
        """Select the N best qubits without connectivity constraints."""
        scored = [(qid, self._qubit_score(qid, metric)) for qid in self.profile.profiles]
        scored.sort(key=lambda x: x[1], reverse=True)

        mapping = {i: qid for i, (qid, _) in enumerate(scored[:n])}
        total_score = sum(s for _, s in scored[:n])
        return QubitMapping(mapping=mapping, score=total_score, metric=metric)

    def _select_constrained(
        self,
        n: int,
        connectivity: list[tuple[int, int]],
        metric: str,
    ) -> QubitMapping:
        """Select qubits that satisfy connectivity constraints."""
        if n <= self.BRUTE_FORCE_LIMIT:
            return self._brute_force_search(n, connectivity, metric)
        return self._greedy_search(n, connectivity, metric)

    def _brute_force_search(
        self,
        n: int,
        connectivity: list[tuple[int, int]],
        metric: str,
    ) -> QubitMapping:
        """Exhaustively search all possible mappings for small circuits."""
        physical_qubits = list(self.profile.profiles.keys())
        best_mapping: QubitMapping | None = None
        best_score = -math.inf

        # Try all permutations of n physical qubits from the chip
        for combo in itertools.permutations(physical_qubits, n):
            # Check connectivity: each (i, j) in connectivity must be
            # physically connected
            valid = True
            for li, lj in connectivity:
                if li >= n or lj >= n:
                    continue
                pq_i = combo[li]
                pq_j = combo[lj]
                neighbors = self.profile.topology.get(pq_i, [])
                if pq_j not in neighbors:
                    valid = False
                    break

            if not valid:
                continue

            # Score this mapping
            score = self._score_mapping(list(combo), connectivity, metric)
            if score > best_score:
                best_score = score
                best_mapping = QubitMapping(
                    mapping={i: combo[i] for i in range(n)},
                    score=score,
                    metric=metric,
                )

        if best_mapping is None:
            raise ValueError("No valid qubit mapping found satisfying connectivity constraints")
        return best_mapping

    def _greedy_search(
        self,
        n: int,
        connectivity: list[tuple[int, int]],
        metric: str,
    ) -> QubitMapping:
        """Greedy heuristic for large circuits.

        Strategy: start with the highest-fidelity edge from the connectivity
        requirements, then greedily expand by adding the best-scoring neighbor
        that satisfies the next constraint.
        """
        # Build adjacency for logical circuit
        logical_adj: dict[int, set[int]] = {}
        for li, lj in connectivity:
            logical_adj.setdefault(li, set()).add(lj)
            logical_adj.setdefault(lj, set()).add(li)

        # Find the best physical edge to seed with
        best_seed_score = -math.inf
        best_seed: tuple[str, str] | None = None

        for q1, neighbors in self.profile.topology.items():
            for q2 in neighbors:
                pair_score = (
                    self._qubit_score(q1, metric)
                    + self._qubit_score(q2, metric)
                    + self._pair_score(q1, q2)
                )
                if pair_score > best_seed_score:
                    best_seed_score = pair_score
                    best_seed = (q1, q2)

        if best_seed is None:
            raise ValueError("No connected pairs found in chip topology")

        # Pick the first connectivity edge to seed with
        if connectivity:
            seed_li, seed_lj = connectivity[0]
        else:
            seed_li, seed_lj = 0, 1

        assigned: dict[int, str] = {seed_li: best_seed[0], seed_lj: best_seed[1]}
        used_physical: set[str] = {best_seed[0], best_seed[1]}

        # Greedily assign remaining logical qubits
        unassigned = set(range(n)) - set(assigned)
        while unassigned:
            best_lq = None
            best_pq = None
            best_lq_score = -math.inf

            for lq in unassigned:
                # Find which physical qubits could work for this logical qubit
                required_neighbors = logical_adj.get(lq, set()) & set(assigned)
                for pq in self.profile.profiles:
                    if pq in used_physical:
                        continue
                    # Check all required connections are satisfied
                    ok = True
                    for req_lq in required_neighbors:
                        req_pq = assigned[req_lq]
                        if pq not in self.profile.topology.get(req_pq, []):
                            ok = False
                            break
                    if not ok:
                        continue
                    score = self._qubit_score(pq, metric)
                    if score > best_lq_score:
                        best_lq_score = score
                        best_lq = lq
                        best_pq = pq

            if best_lq is None or best_pq is None:
                # Fall back: assign remaining without connectivity
                for lq in unassigned:
                    for pq in self.profile.profiles:
                        if pq not in used_physical:
                            assigned[lq] = pq
                            used_physical.add(pq)
                            break
                break

            assigned[best_lq] = best_pq
            used_physical.add(best_pq)
            unassigned.discard(best_lq)

        total_score = self._score_mapping(
            [assigned.get(i, "") for i in range(n)], connectivity, metric
        )
        return QubitMapping(mapping=assigned, score=total_score, metric=metric)

    def _qubit_score(self, qid: str, metric: str) -> float:
        """Score a single physical qubit."""
        p = self.profile.profiles.get(qid)
        if p is None:
            return 0.0
        if metric == ScoringMetric.SINGLE_GATE:
            return p.single_gate_fidelity
        if metric == ScoringMetric.TWO_GATE_MIN:
            if not p.two_gate_fidelities:
                return p.single_gate_fidelity
            return min(p.two_gate_fidelities.values())
        # Default: composite
        return p.composite_fidelity

    def _pair_score(self, q1: str, q2: str) -> float:
        """Score a physical qubit pair by their two-gate fidelity."""
        fid = self.profile.pair_fidelity(q1, q2)
        return fid if fid is not None else 0.0

    def _score_mapping(
        self,
        physical: list[str],
        connectivity: list[tuple[int, int]],
        metric: str,
    ) -> float:
        """Score a complete mapping as sum of qubit + pair scores."""
        score = sum(self._qubit_score(pq, metric) for pq in physical if pq)
        for li, lj in connectivity:
            if li < len(physical) and lj < len(physical):
                score += self._pair_score(physical[li], physical[lj])
        return score
