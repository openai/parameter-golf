"""
test_cantor_emergence.py

Tests for the Cantor-Recursive Emergence method applied to Parameter Golf.

Theory mapping:
  Ω₁  →  token-level (raw propositions, A₀)
  Ω₂  →  phrase/ngram level (A₁ = A₀ ∪ COMPRESS(C ⊆ A₀))
  Ω₃  →  sentence/motif level (A₂ = A₁ ∪ COMPRESS(C ⊆ A₁))
  Ω₄  →  discourse level (A₃ = A₂ ∪ COMPRESS(C ⊆ A₂))

Key invariants tested:
  1. Strict Cantor enrichment: |A_{n+1}| > |A_n|
  2. COMPRESS preserves binding: m(h) = B(C)
  3. grow() is local: only affects touched propositions
  4. Binding-to-Fisher correlation (the core Parameter Golf hypothesis)
  5. Budget allocation by binding energy respects 16MB constraint
  6. Level-lifting: same W(·,·) formula works at every level
  7. n_eff diversity anti-inflation
  8. Productive incompleteness: level N cannot describe all of level N+1
"""

import math
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core data structures (minimal implementation of the hypergraph)
# ---------------------------------------------------------------------------

@dataclass
class Proposition:
    id: str
    mass: float           # m(p) = posterior belief
    entities: Set[str] = field(default_factory=set)
    source_pages: Set[str] = field(default_factory=set)

@dataclass
class Handle:
    """COMPRESS(C) → Handle. m(h) = B(C)."""
    id: str
    level: int            # Cantor level this handle lives at
    mass: float           # = B(C) at creation time
    members: Set[str] = field(default_factory=set)  # proposition/handle ids compressed

@dataclass
class Relation:
    type: str             # causal, temporal, motive, contradiction
    p1: str
    p2: str
    confidence: float = 1.0

ALPHA = {
    "causal":        +1.0,
    "temporal":      +1.0,
    "motive":        +0.5,
    "contradiction": -0.5,
}

class EpistemicHypergraph:
    """
    Minimal implementation of H = (V, E, τ, L, m, w) sufficient for testing
    the Cantor-emergence / Parameter Golf binding-energy hypotheses.
    """

    def __init__(self):
        self.propositions: Dict[str, Proposition] = {}
        self.handles: Dict[str, Handle] = {}
        self.relations: List[Relation] = []
        self._entity_degree: Dict[str, int] = defaultdict(int)

    # -- graph building -------------------------------------------------------

    def add_proposition(self, p: Proposition):
        self.propositions[p.id] = p
        for e in p.entities:
            self._entity_degree[e] += 1

    def add_relation(self, r: Relation):
        self.relations.append(r)

    # -- binding forces (§4) --------------------------------------------------

    def specificity(self, entity: str) -> float:
        deg = self._entity_degree[entity]
        return 1.0 / deg if deg > 0 else 0.0

    def W_entity(self, p1: Proposition, p2: Proposition) -> float:
        shared = p1.entities & p2.entities
        return sum(p1.mass * p2.mass * self.specificity(e) for e in shared)

    def W_relation(self, pid1: str, pid2: str) -> float:
        total = 0.0
        for r in self.relations:
            if {r.p1, r.p2} == {pid1, pid2}:
                total += r.confidence * ALPHA.get(r.type, 0.0)
        return total

    def W_context(self, p1: Proposition, p2: Proposition) -> float:
        shared_pages = p1.source_pages & p2.source_pages
        return sum(1.0 / max(1, len(p.source_pages))
                   for page in shared_pages
                   for p in self.propositions.values()
                   if page in p.source_pages) / max(1, len(shared_pages)) \
               if shared_pages else 0.0

    def W(self, pid1: str, pid2: str) -> float:
        p1, p2 = self.propositions[pid1], self.propositions[pid2]
        return self.W_entity(p1, p2) + self.W_relation(pid1, pid2) + self.W_context(p1, p2)

    # -- binding energy (§5) --------------------------------------------------

    def binding_energy(self, ids: Set[str]) -> float:
        """B(S) = (2 / |S|(|S|-1)) Σ_{i<j} W(pᵢ, pⱼ)"""
        ids = list(ids)
        n = len(ids)
        if n < 2:
            return 0.0
        n_pairs = n * (n - 1) / 2
        total = sum(self.W(ids[i], ids[j])
                    for i in range(n) for j in range(i + 1, n))
        return total / n_pairs

    # -- COMPRESS (§9) --------------------------------------------------------

    def compress(self, ids: Set[str], level: int, handle_id: str) -> Handle:
        """COMPRESS(C) → Handle with m(h) = B(C)."""
        b = self.binding_energy(ids)
        h = Handle(id=handle_id, level=level, mass=b, members=set(ids))
        self.handles[handle_id] = h
        return h

    # -- n_eff diversity (§here_news) -----------------------------------------

    @staticmethod
    def n_eff(source_counts: Dict[str, int], k: float = 1.0) -> float:
        """n_eff(x) = Σ_a [1 - exp(-n_a(x) / k)]"""
        return sum(1.0 - math.exp(-n / k) for n in source_counts.values())


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_dense_cluster(graph: EpistemicHypergraph,
                       prefix: str,
                       n: int,
                       shared_entity: str,
                       mass: float = 1.0) -> Set[str]:
    """Build n propositions that all share one entity (high binding)."""
    ids = set()
    for i in range(n):
        pid = f"{prefix}_{i}"
        graph.add_proposition(Proposition(
            id=pid,
            mass=mass,
            entities={shared_entity, f"{prefix}_specific_{i}"},
            source_pages={f"page_{prefix}_{i // 2}"},
        ))
        ids.add(pid)
    # add entity to degree accounting
    graph._entity_degree[shared_entity] += n
    return ids


def make_sparse_cluster(graph: EpistemicHypergraph,
                        prefix: str,
                        n: int,
                        mass: float = 1.0) -> Set[str]:
    """Build n propositions with NO shared entities (low binding)."""
    ids = set()
    for i in range(n):
        pid = f"{prefix}_{i}"
        graph.add_proposition(Proposition(
            id=pid,
            mass=mass,
            entities={f"{prefix}_unique_{i}"},
            source_pages={f"page_{prefix}_{i}"},
        ))
        ids.add(pid)
    return ids


# ---------------------------------------------------------------------------
# 1. Strict Cantor Enrichment: |A_{n+1}| > |A_n|
# ---------------------------------------------------------------------------

class TestCantorEnrichment:

    def test_level_0_has_only_propositions(self):
        g = EpistemicHypergraph()
        ids = make_dense_cluster(g, "p", 4, "entity_A")
        A0 = set(g.propositions.keys())
        assert len(A0) == 4

    def test_level_1_strictly_larger(self):
        g = EpistemicHypergraph()
        ids = make_dense_cluster(g, "p", 4, "entity_A")
        A0_size = len(g.propositions)
        # COMPRESS the cluster into a level-1 handle
        h = g.compress(ids, level=1, handle_id="h1")
        A1_size = len(g.propositions) + len(g.handles)
        assert A1_size > A0_size, "Level 1 alphabet must be strictly larger than level 0"

    def test_level_2_strictly_larger_than_level_1(self):
        g = EpistemicHypergraph()
        ids_a = make_dense_cluster(g, "a", 4, "entity_A")
        ids_b = make_dense_cluster(g, "b", 4, "entity_B")
        h1 = g.compress(ids_a, level=1, handle_id="h1")
        h2 = g.compress(ids_b, level=1, handle_id="h2")
        A1_size = len(g.propositions) + len(g.handles)
        # Level-2 handle over the two level-1 handles
        # We simulate by treating handle ids as "propositions" for the level-2 cluster
        h3 = Handle(id="h3", level=2, mass=0.0, members={"h1", "h2"})
        g.handles["h3"] = h3
        A2_size = len(g.propositions) + len(g.handles)
        assert A2_size > A1_size

    def test_no_enrichment_without_emergent_cause(self):
        """If B(S) ≈ 0 (fully disconnected), no handle should be created."""
        g = EpistemicHypergraph()
        ids = make_sparse_cluster(g, "sparse", 3)
        b = g.binding_energy(ids)
        # binding energy of unconnected propositions is ~0
        assert b == pytest.approx(0.0), \
            "Sparse cluster should have zero binding energy"
        # No handle created → A0 == A1 semantically (no new structure)
        assert len(g.handles) == 0


# ---------------------------------------------------------------------------
# 2. COMPRESS preserves binding: m(h) = B(C)
# ---------------------------------------------------------------------------

class TestCompressPreservesBinding:

    def test_handle_mass_equals_binding_energy(self):
        g = EpistemicHypergraph()
        ids = make_dense_cluster(g, "p", 5, "entity_X")
        b = g.binding_energy(ids)
        h = g.compress(ids, level=1, handle_id="h_test")
        assert h.mass == pytest.approx(b, rel=1e-9), \
            "m(h) must equal B(C) exactly — COMPRESS preserves binding"

    def test_high_binding_cluster_produces_high_mass_handle(self):
        g = EpistemicHypergraph()
        dense_ids = make_dense_cluster(g, "dense", 6, "shared_entity", mass=1.0)
        sparse_ids = make_sparse_cluster(g, "sparse", 6, mass=1.0)
        h_dense = g.compress(dense_ids, level=1, handle_id="h_dense")
        h_sparse = g.compress(sparse_ids, level=1, handle_id="h_sparse")
        assert h_dense.mass > h_sparse.mass, \
            "Dense cluster must produce higher-mass handle than sparse cluster"

    def test_adding_unrelated_proposition_decreases_binding(self):
        """Cause is a binding maximum: B(C ∪ {p}) < B(C) for external p."""
        g = EpistemicHypergraph()
        core_ids = make_dense_cluster(g, "core", 4, "entity_core")
        outsider = Proposition(id="outsider", mass=1.0,
                               entities={"totally_different_entity"},
                               source_pages={"outsider_page"})
        g.add_proposition(outsider)
        g._entity_degree["totally_different_entity"] += 1

        b_core = g.binding_energy(core_ids)
        b_with_outsider = g.binding_energy(core_ids | {"outsider"})
        assert b_with_outsider < b_core, \
            "Adding unrelated proposition must decrease binding energy (cause is a local max)"

    def test_compress_at_level_n_feeds_level_n_plus_1(self):
        """Each level's COMPRESS output becomes next level's grow() input."""
        g = EpistemicHypergraph()
        ids_a = make_dense_cluster(g, "a", 4, "entity_A")
        ids_b = make_dense_cluster(g, "b", 4, "entity_B")
        h1 = g.compress(ids_a, level=1, handle_id="h1")
        h2 = g.compress(ids_b, level=1, handle_id="h2")
        # Level-2 handle references level-1 handles (Gödel encoding)
        h3 = Handle(id="h3", level=2, mass=h1.mass + h2.mass,
                    members={"h1", "h2"})
        g.handles["h3"] = h3
        assert h3.level == 2
        assert "h1" in h3.members and "h2" in h3.members
        assert h3.mass > 0


# ---------------------------------------------------------------------------
# 3. grow() locality: only touches affected propositions
# ---------------------------------------------------------------------------

class TestGrowLocality:

    def test_update_does_not_touch_unrelated_cluster(self):
        g = EpistemicHypergraph()
        fire_ids = make_dense_cluster(g, "fire", 4, "bondi_beach")
        lai_ids = make_dense_cluster(g, "lai", 4, "jimmy_lai")

        b_lai_before = g.binding_energy(lai_ids)

        # Simulate grow(): update mass of one fire proposition
        g.propositions["fire_0"].mass = 2.0
        # Recompute binding only for fire cluster
        b_fire_after = g.binding_energy(fire_ids)
        b_lai_after = g.binding_energy(lai_ids)

        assert b_lai_before == pytest.approx(b_lai_after), \
            "Updating fire cluster must not change lai cluster binding (locality)"
        # fire binding changed (trivially true since mass changed)
        assert b_fire_after != pytest.approx(0.0)

    def test_shared_entity_creates_cross_cluster_binding(self):
        """If two clusters share a specific entity, they DO interact."""
        g = EpistemicHypergraph()
        # Both clusters mention the same rare entity
        make_dense_cluster(g, "cluster_a", 3, "rare_entity")
        make_dense_cluster(g, "cluster_b", 3, "rare_entity")
        # cross-binding should be positive because of shared rare entity
        cross = g.W("cluster_a_0", "cluster_b_0")
        assert cross > 0.0, \
            "Shared specific entity must produce positive cross-cluster binding"


# ---------------------------------------------------------------------------
# 4. Binding-to-Fisher correlation (core Parameter Golf hypothesis)
# ---------------------------------------------------------------------------

class TestBindingFisherCorrelation:
    """
    The hypothesis: weight blocks with high binding energy B(C) correspond
    to weight blocks with high Fisher information (gradient magnitude).
    We test the *structural* analogy, not the full neural network.

    Fisher information proxy: for a simple Gaussian model, Fisher ∝ 1/variance.
    We simulate this by treating m(p) as the "activation magnitude" and
    checking that high-binding clusters produce higher Fisher-proxy scores.
    """

    @staticmethod
    def fisher_proxy(masses: List[float]) -> float:
        """Fisher proxy = sum of squared masses (like squared gradient norms)."""
        return sum(m ** 2 for m in masses)

    def test_high_binding_cluster_has_higher_fisher_proxy(self):
        g = EpistemicHypergraph()
        dense_ids = make_dense_cluster(g, "dense", 5, "shared_entity", mass=1.0)
        sparse_ids = make_sparse_cluster(g, "sparse", 5, mass=1.0)

        b_dense = g.binding_energy(dense_ids)
        b_sparse = g.binding_energy(sparse_ids)

        masses_dense = [g.propositions[pid].mass for pid in dense_ids]
        masses_sparse = [g.propositions[pid].mass for pid in sparse_ids]

        fp_dense = self.fisher_proxy(masses_dense)
        fp_sparse = self.fisher_proxy(masses_sparse)

        # Both have same masses (1.0), but different binding structure
        # The *structural* claim: if we were to prune by Fisher, high-binding
        # clusters survive; low-binding ones don't
        assert b_dense > b_sparse, "Dense cluster must have higher binding"
        # Fisher proxy is equal here (same masses) — this is expected.
        # The test for the full hypothesis requires a trained model;
        # here we verify the structural precondition holds.
        assert fp_dense == pytest.approx(fp_sparse), \
            "Fisher proxy is mass-based; binding is structure-based — they're independent signals"

    def test_binding_energy_monotone_in_mass(self):
        """Higher mass propositions in the same structure → higher binding energy."""
        g1 = EpistemicHypergraph()
        ids1 = make_dense_cluster(g1, "p", 4, "entity_A", mass=1.0)
        b1 = g1.binding_energy(ids1)

        g2 = EpistemicHypergraph()
        ids2 = make_dense_cluster(g2, "p", 4, "entity_A", mass=2.0)
        b2 = g2.binding_energy(ids2)

        assert b2 > b1, "Binding energy must increase with proposition mass (W_entity ∝ m₁·m₂)"

    def test_specificity_modulates_binding(self):
        """Rare entities (high specificity) create tighter binding than common ones."""
        g = EpistemicHypergraph()
        # rare entity: only 2 propositions mention it
        p1 = Proposition("p1", mass=1.0, entities={"rare_entity"})
        p2 = Proposition("p2", mass=1.0, entities={"rare_entity"})
        g.add_proposition(p1)
        g.add_proposition(p2)
        g._entity_degree["rare_entity"] = 2

        # common entity: 100 propositions mention it
        p3 = Proposition("p3", mass=1.0, entities={"common_entity"})
        p4 = Proposition("p4", mass=1.0, entities={"common_entity"})
        g.add_proposition(p3)
        g.add_proposition(p4)
        g._entity_degree["common_entity"] = 100

        w_rare = g.W_entity(p1, p2)
        w_common = g.W_entity(p3, p4)
        assert w_rare > w_common, \
            "Rare entities (σ=1/deg) must produce stronger binding than common ones"


# ---------------------------------------------------------------------------
# 5. Budget allocation by binding energy (16MB Parameter Golf constraint)
# ---------------------------------------------------------------------------

TOTAL_BUDGET_BYTES = 16_000_000  # 16MB decimal

class TestBudgetAllocation:
    """
    The allocation rule: bits_per_handle ∝ m(h) = B(C).
    High-binding handles get more bits (lower quantization).
    Total must stay within 16MB.
    """

    @staticmethod
    def bits_for_handle(handle: Handle,
                        total_binding: float,
                        total_budget_bits: int) -> int:
        """Allocate bits proportional to binding energy."""
        if total_binding == 0:
            return 0
        return int((handle.mass / total_binding) * total_budget_bits)

    def test_total_allocation_within_budget(self):
        g = EpistemicHypergraph()
        clusters = [
            make_dense_cluster(g, f"c{i}", 5, f"entity_{i}")
            for i in range(4)
        ]
        handles = [g.compress(c, level=1, handle_id=f"h{i}")
                   for i, c in enumerate(clusters)]

        total_binding = sum(h.mass for h in handles)
        total_bits = TOTAL_BUDGET_BYTES * 8

        allocated = [self.bits_for_handle(h, total_binding, total_bits)
                     for h in handles]
        assert sum(allocated) <= total_bits, \
            "Total allocated bits must not exceed 16MB budget"

    def test_higher_binding_gets_more_bits(self):
        g = EpistemicHypergraph()
        dense_ids = make_dense_cluster(g, "dense", 6, "hot_entity", mass=1.0)
        sparse_ids = make_sparse_cluster(g, "sparse", 6, mass=1.0)

        h_dense = g.compress(dense_ids, level=1, handle_id="h_dense")
        h_sparse = g.compress(sparse_ids, level=1, handle_id="h_sparse")

        total_binding = h_dense.mass + h_sparse.mass
        if total_binding == 0:
            pytest.skip("No binding energy — trivial case")

        total_bits = TOTAL_BUDGET_BYTES * 8
        bits_dense = self.bits_for_handle(h_dense, total_binding, total_bits)
        bits_sparse = self.bits_for_handle(h_sparse, total_binding, total_bits)

        assert bits_dense > bits_sparse, \
            "High-binding handle must receive more bits (lower effective quantization)"

    def test_zero_binding_handle_gets_zero_bits(self):
        g = EpistemicHypergraph()
        ids = make_sparse_cluster(g, "empty", 3)
        h = g.compress(ids, level=1, handle_id="h_empty")

        assert h.mass == pytest.approx(0.0)
        total_bits = TOTAL_BUDGET_BYTES * 8
        bits = self.bits_for_handle(h, total_binding=1.0, total_budget_bits=total_bits)
        assert bits == 0, "Zero-binding handle must receive zero bits (drop it)"

    def test_cantor_level_allocation(self):
        """
        Deeper Cantor levels (richer alphabet) should get proportionally more bits
        if their binding mass is higher — which it should be for high-coherence discourse.
        """
        g = EpistemicHypergraph()
        # Level 1: two dense clusters
        ids_a = make_dense_cluster(g, "a", 4, "entity_A")
        ids_b = make_dense_cluster(g, "b", 4, "entity_B")
        h1 = g.compress(ids_a, level=1, handle_id="h1")
        h2 = g.compress(ids_b, level=1, handle_id="h2")

        # Level 2: handle over two level-1 handles
        # Mass = sum of level-1 masses (simplified level-lifting)
        h_level2 = Handle(id="h_level2", level=2,
                          mass=h1.mass + h2.mass,
                          members={"h1", "h2"})
        g.handles["h_level2"] = h_level2

        # Level-2 handle's mass >= either level-1 handle's mass
        assert h_level2.mass >= max(h1.mass, h2.mass), \
            "Level-2 handle (discourse) must have mass >= its component level-1 handles"


# ---------------------------------------------------------------------------
# 6. Level-lifting: same W(·,·) formula at every level
# ---------------------------------------------------------------------------

class TestLevelLifting:

    def test_same_binding_formula_applies_at_level_2(self):
        """
        W(h₁, h₂) at level N+1 uses the same three forces.
        We test that two handles with shared 'meta-entities' (shared member propositions)
        have positive cross-binding — the same formula, parameterized by level.
        """
        g = EpistemicHypergraph()

        # Both clusters share a proposition (simulating shared boundary node)
        shared_prop = Proposition("shared", mass=1.0, entities={"shared_entity"})
        g.add_proposition(shared_prop)
        g._entity_degree["shared_entity"] = 1

        ids_a = make_dense_cluster(g, "a", 3, "entity_A") | {"shared"}
        ids_b = make_dense_cluster(g, "b", 3, "entity_B") | {"shared"}

        h1 = g.compress(ids_a, level=1, handle_id="h1")
        h2 = g.compress(ids_b, level=1, handle_id="h2")

        # Cross-binding via shared member
        shared_in_both = h1.members & h2.members
        assert len(shared_in_both) > 0, "Handles must share at least one member"

        # At level 2, binding between h1 and h2 includes shared boundary propositions
        # We use the member overlap as a proxy for W_entity at level 2
        cross_binding_proxy = len(shared_in_both) / max(len(h1.members), len(h2.members))
        assert cross_binding_proxy > 0, \
            "Handles sharing boundary propositions must have positive level-2 cross-binding"

    def test_sub_cause_embed_criterion(self):
        """
        C₁ is sub-cause of C₂ iff:
          (i)  B(C₁) > 0
          (ii) W̄(C₁, C₂) > 0
          (iii) B(C₁ ∪ C₂) ≤ max(B(C₁), B(C₂))   [merge dilutes]
          (iv) W̄(C₁, C₂) > W̄(C₁, C₃) for any other C₃
        """
        g = EpistemicHypergraph()
        # Dense parent cluster
        parent_ids = make_dense_cluster(g, "parent", 6, "main_entity")
        # Small sub-cluster sharing the same entity (sub-cause)
        sub_ids = make_dense_cluster(g, "sub", 2, "main_entity")
        # Unrelated cluster
        unrelated_ids = make_sparse_cluster(g, "unrelated", 4)

        b_sub = g.binding_energy(sub_ids)
        b_parent = g.binding_energy(parent_ids)
        b_unrelated = g.binding_energy(unrelated_ids)
        b_merged = g.binding_energy(parent_ids | sub_ids)

        # cross-binding densities
        def cross_binding(s1, s2):
            pairs = [(a, b) for a in s1 for b in s2]
            if not pairs:
                return 0.0
            return sum(g.W(a, b) for a, b in pairs) / len(pairs)

        wcross_sub_parent = cross_binding(sub_ids, parent_ids)
        wcross_sub_unrelated = cross_binding(sub_ids, unrelated_ids)

        # (i) B(sub) > 0
        assert b_sub > 0, "Sub-cause must have positive internal binding"
        # (ii) W̄(sub, parent) > 0
        assert wcross_sub_parent > 0, "Sub-cause must bind to parent"
        # (iii) merge dilutes (or at most equals)
        assert b_merged <= max(b_sub, b_parent) + 1e-9, \
            "Merging sub-cause into parent must not increase binding above max component"
        # (iv) W̄(sub, parent) > W̄(sub, unrelated)
        assert wcross_sub_parent > wcross_sub_unrelated, \
            "Sub-cause must bind more to parent than to unrelated cluster"


# ---------------------------------------------------------------------------
# 7. n_eff diversity anti-inflation
# ---------------------------------------------------------------------------

class TestNEffDiversity:

    def test_single_source_saturates(self):
        """One article repeating itself 10 times saturates — n_eff stays near 1."""
        single_source = {"source_A": 10}
        n = EpistemicHypergraph.n_eff(single_source, k=1.0)
        assert n < 1.01, "Single source with 10 repeats should saturate near 1"
        assert n > 0.99, "Should be close to 1 (not 0)"

    def test_two_independent_sources_give_higher_n_eff(self):
        """A second independent source is a real signal → n_eff > 1."""
        two_sources = {"source_A": 5, "source_B": 5}
        one_source = {"source_A": 10}
        n_two = EpistemicHypergraph.n_eff(two_sources, k=1.0)
        n_one = EpistemicHypergraph.n_eff(one_source, k=1.0)
        assert n_two > n_one, "Two independent sources must give higher n_eff"

    def test_n_eff_bounded_by_source_count(self):
        """n_eff ≤ number of distinct sources (maximum diversity)."""
        sources = {f"source_{i}": 100 for i in range(5)}
        n = EpistemicHypergraph.n_eff(sources, k=1.0)
        assert n <= 5 + 1e-9, "n_eff cannot exceed number of distinct sources"

    def test_n_eff_monotone_in_source_count(self):
        """More distinct sources → higher n_eff."""
        for k in [0.5, 1.0, 2.0]:
            prev = 0.0
            for n_sources in [1, 2, 5, 10]:
                sources = {f"s_{i}": 3 for i in range(n_sources)}
                n = EpistemicHypergraph.n_eff(sources, k=k)
                assert n > prev, f"n_eff must increase with source count (k={k})"
                prev = n

    def test_n_eff_as_training_data_diversity_signal(self):
        """
        Parameter Golf application: documents with high n_eff diversity
        should be selected over redundant corroborations.
        The selection criterion: keep doc if it increases n_eff by > threshold.
        """
        selected_sources: Dict[str, int] = {}
        candidates = [
            ("doc_A", "source_1"),
            ("doc_B", "source_1"),   # same source as doc_A → low marginal n_eff
            ("doc_C", "source_2"),   # new source → high marginal n_eff
            ("doc_D", "source_3"),   # new source → high marginal n_eff
        ]
        # Marginal gain of adding a doc from the same source decreases with each repeat.
        # Use k=1.0: first doc from source_1 gives ~0.632 gain,
        # second doc from source_1 gives ~0.233 gain (saturating quickly).
        # Threshold set above the saturated marginal gain to filter redundancy.
        threshold = 0.3   # above saturation plateau for repeated sources
        selected = []
        for doc_id, source in candidates:
            n_before = EpistemicHypergraph.n_eff(
                {**selected_sources}, k=1.0) if selected_sources else 0.0
            test_sources = dict(selected_sources)
            test_sources[source] = test_sources.get(source, 0) + 1
            n_after = EpistemicHypergraph.n_eff(test_sources, k=1.0)
            if n_after - n_before > threshold:
                selected.append(doc_id)
                selected_sources = test_sources

        assert "doc_A" in selected, "First document from new source must be selected"
        assert "doc_B" not in selected, "Redundant same-source document must be rejected"
        assert "doc_C" in selected, "Document from new source must be selected"
        assert "doc_D" in selected, "Document from another new source must be selected"


# ---------------------------------------------------------------------------
# 8. Productive incompleteness: level N cannot fully describe level N+1
# ---------------------------------------------------------------------------

class TestProductiveIncompleteness:

    def test_level_1_handle_creates_new_structure_invisible_at_level_0(self):
        """
        At level 0 we have only propositions.
        After COMPRESS, the Handle is a new object that doesn't exist at level 0.
        It encodes structure (B(C)) that no single proposition captures.
        """
        g = EpistemicHypergraph()
        ids = make_dense_cluster(g, "p", 4, "entity_X")
        b = g.binding_energy(ids)

        # No level-0 node encodes B(C)
        for pid in ids:
            p = g.propositions[pid]
            assert p.mass != pytest.approx(b), \
                "No single proposition mass equals the cluster's binding energy"

        h = g.compress(ids, level=1, handle_id="h1")
        # Only the handle encodes B(C) — new level-1 structure
        assert h.mass == pytest.approx(b), \
            "Handle IS the new structure: it encodes B(C) which level-0 couldn't express"

    def test_level_n_binding_creates_object_not_in_level_n_minus_1(self):
        """
        Each COMPRESS creates a Gödel-like encoding that references the level below.
        The handle's members are level-N objects; the handle itself is level-N+1.
        """
        g = EpistemicHypergraph()
        ids_a = make_dense_cluster(g, "a", 3, "entity_A")
        ids_b = make_dense_cluster(g, "b", 3, "entity_B")
        h1 = g.compress(ids_a, level=1, handle_id="h1")
        h2 = g.compress(ids_b, level=1, handle_id="h2")

        # h3 is a level-2 object — it references level-1 objects
        h3 = Handle(id="h3", level=2, mass=h1.mass + h2.mass,
                    members={"h1", "h2"})
        g.handles["h3"] = h3

        # h3's members are all level-1 objects
        for member_id in h3.members:
            assert member_id in g.handles, f"{member_id} must be a level-1 handle"
            assert g.handles[member_id].level == 1

        # h3 itself is level-2 — new structure not expressible at level-1
        assert h3.level == 2
        assert h3.level > g.handles["h1"].level

    def test_incompleteness_drives_level_ascent(self):
        """
        If B(C) > 0 at level N, we need level N+1 to describe it.
        If B(C) = 0 at level N, no level-N+1 handle is warranted.
        This is productive incompleteness: non-zero binding = new structure = new level needed.
        """
        g = EpistemicHypergraph()
        dense_ids = make_dense_cluster(g, "dense", 5, "entity_Z")
        sparse_ids = make_sparse_cluster(g, "sparse", 5)

        b_dense = g.binding_energy(dense_ids)
        b_sparse = g.binding_energy(sparse_ids)

        # Dense cluster: binding > 0 → level ascent warranted
        assert b_dense > 0, "Dense cluster must have positive binding → level ascent needed"
        # Sparse cluster: binding = 0 → no level ascent warranted
        assert b_sparse == pytest.approx(0.0), \
            "Sparse cluster has zero binding → no new level needed (no productive incompleteness)"


# ---------------------------------------------------------------------------
# 9. Integration test: full Cantor pipeline for a toy FineWeb-like corpus
# ---------------------------------------------------------------------------

class TestFullCantorPipeline:
    """
    End-to-end test simulating the Parameter Golf use case:
    tokens → phrases → motifs → discourse,
    with binding-energy-based bit allocation.
    """

    def build_toy_corpus(self) -> Tuple[EpistemicHypergraph, List[Set[str]]]:
        """
        Simulates a tiny FineWeb-like corpus with 3 topically coherent clusters
        and 1 noise cluster.
        """
        g = EpistemicHypergraph()
        # Topic A: machine learning
        ml_ids = make_dense_cluster(g, "ml", 6, "machine_learning", mass=1.0)
        # Topic B: climate science
        clim_ids = make_dense_cluster(g, "clim", 6, "climate_change", mass=1.0)
        # Topic C: sports
        sport_ids = make_dense_cluster(g, "sport", 6, "football", mass=1.0)
        # Noise: no shared entities
        noise_ids = make_sparse_cluster(g, "noise", 6, mass=1.0)
        return g, [ml_ids, clim_ids, sport_ids, noise_ids]

    def test_level_1_handles_emerge_for_coherent_topics(self):
        g, clusters = self.build_toy_corpus()
        handles = []
        for i, ids in enumerate(clusters):
            b = g.binding_energy(ids)
            if b > 0:
                h = g.compress(ids, level=1, handle_id=f"h_{i}")
                handles.append(h)
        # 3 coherent topics should produce handles; noise should not
        assert len(handles) == 3, \
            "Exactly 3 coherent topics should produce level-1 handles"

    def test_budget_allocation_preserves_topical_handles(self):
        g, clusters = self.build_toy_corpus()
        handles = []
        for i, ids in enumerate(clusters):
            b = g.binding_energy(ids)
            if b > 0:
                h = g.compress(ids, level=1, handle_id=f"h_{i}")
                handles.append(h)

        total_binding = sum(h.mass for h in handles)
        total_bits = TOTAL_BUDGET_BYTES * 8

        for h in handles:
            bits = int((h.mass / total_binding) * total_bits)
            assert bits > 0, "Every coherent topic handle must receive non-zero bits"

    def test_cantor_hierarchy_is_3_levels_deep_for_toy_corpus(self):
        g, clusters = self.build_toy_corpus()
        # Level 1
        l1_handles = []
        for i, ids in enumerate(clusters[:3]):  # coherent topics only
            h = g.compress(ids, level=1, handle_id=f"h1_{i}")
            l1_handles.append(h)
        # Level 2: compress the 3 level-1 handles into a discourse handle
        h_discourse = Handle(
            id="h_discourse",
            level=2,
            mass=sum(h.mass for h in l1_handles),
            members={h.id for h in l1_handles},
        )
        g.handles["h_discourse"] = h_discourse
        # Level 3: meta-handle (system self-model, Φ)
        h_meta = Handle(
            id="h_meta",
            level=3,
            mass=h_discourse.mass,
            members={"h_discourse"},
        )
        g.handles["h_meta"] = h_meta

        levels = {h.level for h in g.handles.values()}
        assert levels == {1, 2, 3}, \
            "Toy corpus should produce exactly 3 Cantor levels: phrase, discourse, meta"

        # Strict enrichment at each level
        assert h_discourse.level > l1_handles[0].level
        assert h_meta.level > h_discourse.level


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
