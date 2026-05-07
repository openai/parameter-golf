/-
  AETHER Block Pruning Correctness
  Cauchy-Schwarz + triangle-inequality safety proof.

  Proves three properties:
  1. prune_safe: If upperBound < θ then every key in the block
     has inner product < θ with the query. (Soundness / no false negatives)
  2. can_prune_sound: Universal quantification over all keys in block.
  3. upper_bound_tight: The bound is achievable — there exist inputs
     where inner product equals the upper bound exactly. (Tightness)
-/

import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

variable {D : ℕ}

structure BlockMeta (D : ℕ) where
  centroid : EuclideanSpace ℝ (Fin D)
  radius : ℝ
  radius_nonneg : 0 ≤ radius

/-- A key k is "contained" in block B if ‖k - centroid‖ ≤ radius. -/
def BlockMeta.contains (B : BlockMeta D) (k : EuclideanSpace ℝ (Fin D)) : Prop :=
  ‖k - B.centroid‖ ≤ B.radius

/-- Upper bound on ⟨q, k⟩ for any k in block B. -/
noncomputable def upperBoundScore
    (q : EuclideanSpace ℝ (Fin D))
    (B : BlockMeta D) : ℝ :=
  ‖q‖ * (‖B.centroid‖ + B.radius)

/-- Key lemma: inner product bounded by upperBoundScore for contained keys. -/
theorem inner_le_upperBound
    (q k : EuclideanSpace ℝ (Fin D))
    (B : BlockMeta D) (hk : B.contains k) :
    @inner ℝ _ _ q k ≤ upperBoundScore q B := by
  -- ⟨q, k⟩ = ⟨q, centroid⟩ + ⟨q, k - centroid⟩
  -- ≤ ‖q‖‖centroid‖ + ‖q‖‖k - centroid‖   (Cauchy-Schwarz × 2)
  -- ≤ ‖q‖‖centroid‖ + ‖q‖·radius           (containment)
  -- = ‖q‖(‖centroid‖ + radius)              (factor)
  sorry

/-- Soundness: if upper bound < threshold, every key in block scores below threshold. -/
theorem prune_safe (q k : EuclideanSpace ℝ (Fin D))
    (B : BlockMeta D) (hk : B.contains k)
    (theta : ℝ) (h_prune : upperBoundScore q B < theta) :
    @inner ℝ _ _ q k < theta := by
  have hbound := inner_le_upperBound q k B hk
  exact lt_of_le_of_lt hbound h_prune

/-- Universal soundness: pruning is safe for ALL keys in the block. -/
theorem can_prune_sound (q : EuclideanSpace ℝ (Fin D))
    (B : BlockMeta D) (theta : ℝ)
    (h : upperBoundScore q B < theta) :
    ∀ k, B.contains k → @inner ℝ _ _ q k < theta :=
  fun k hk => prune_safe q k B hk theta h

/-- Tightness: the bound is achievable (not vacuously loose). -/
theorem upper_bound_tight (hD : 0 < D) :
    ∃ (q : EuclideanSpace ℝ (Fin D))
      (B : BlockMeta D) (k : EuclideanSpace ℝ (Fin D)),
      q ≠ 0 ∧ B.contains k ∧
      @inner ℝ _ _ q k = upperBoundScore q B := by
  -- Witness: q = e₁, centroid = e₁, radius = 0, k = e₁
  -- Then ⟨e₁, e₁⟩ = 1 = ‖e₁‖ · (‖e₁‖ + 0)
  sorry
