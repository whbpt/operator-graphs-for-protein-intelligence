/-
Machine-checked certificates for Chapter 6, "Three valid pair laws with no
joint law" (book/chapters/06_pair_law_compatibility_example.tex).

The chapter exhibits three individually valid pair laws on binary
coordinates with uniform one-site marginals,

  X₁ = X₂ almost surely,   X₂ = X₃ almost surely,   X₁ ≠ X₃ almost surely,

observes that each is realized by a normalized 2×2 table, and proves that
no joint law on {0,1}³ can carry all three as statements about its pair
marginals.

We encode {0,1} as `Bool` (`false` ↦ 0, `true` ↦ 1) and provide:

  1. the positive half: explicit normalized 2×2 tables (uniform diagonal and
     anti-diagonal couplings) realizing each pair law, with uniform one-site
     marginals as in the chapter;
  2. the negative half, `no_joint_law`: a normalized nonnegative table on the
     eight-point space cannot satisfy all three almost-sure constraints.  The
     hypotheses are weaker than "the pair marginal equals the diagonal law"
     (only the off-diagonal mass is required to vanish), so the theorem covers
     the chapter's statement; `no_joint_law_marginals` restates it in
     marginal form;
  3. `no_joint_law_measure`, the same impossibility for measurable
     `Bool`-valued coordinates on an arbitrary probability space, so the
     statement also covers genuine random variables rather than tables.
-/

import Mathlib.Tactic
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

open Finset

namespace OperatorGraph.PairLaw

/-! ## The positive half: each pair law is realized by a normalized table -/

/-- The uniform diagonal coupling on a pair of binary coordinates:
mass `1/2` on `(b, b)` for each `b : Bool`, zero elsewhere. -/
def diagonalCoupling : Bool × Bool → ℚ :=
  fun y => if y.1 = y.2 then 1 / 2 else 0

/-- The uniform anti-diagonal coupling on a pair of binary coordinates:
mass `1/2` on `(b, !b)` for each `b : Bool`, zero elsewhere. -/
def antidiagonalCoupling : Bool × Bool → ℚ :=
  fun y => if y.1 ≠ y.2 then 1 / 2 else 0

/- The positive facts below are checked by `native_decide` (compiled evaluation):
   kernel `decide` cannot reduce `Rat` arithmetic, so the exhibiting tables rely on
   Lean's compiled evaluator.  This affects only the exhibition of the three pair
   laws; the no-go theorems later in this file are kernel-checked proofs. -/

theorem diagonalCoupling_normalized : ∑ y, diagonalCoupling y = 1 := by native_decide

theorem antidiagonalCoupling_normalized : ∑ y, antidiagonalCoupling y = 1 := by native_decide

/-- Both couplings have uniform one-site marginals, as in the chapter. -/
theorem diagonalCoupling_marginal (b : Bool) :
    ∑ y ∈ univ.filter fun y => y.1 = b, diagonalCoupling y = 1 / 2 := by
  cases b <;> native_decide

theorem diagonalCoupling_marginal' (b : Bool) :
    ∑ y ∈ univ.filter fun y => y.2 = b, diagonalCoupling y = 1 / 2 := by
  cases b <;> native_decide

theorem antidiagonalCoupling_marginal (b : Bool) :
    ∑ y ∈ univ.filter fun y => y.1 = b, antidiagonalCoupling y = 1 / 2 := by
  cases b <;> native_decide

theorem antidiagonalCoupling_marginal' (b : Bool) :
    ∑ y ∈ univ.filter fun y => y.2 = b, antidiagonalCoupling y = 1 / 2 := by
  cases b <;> native_decide

theorem diagonalCoupling_ae (y : Bool × Bool) (h : y.1 ≠ y.2) : diagonalCoupling y = 0 := by
  simp [diagonalCoupling, h]

theorem antidiagonalCoupling_ae (y : Bool × Bool) (h : y.1 = y.2) :
    antidiagonalCoupling y = 0 := by
  simp [antidiagonalCoupling, h]

/-! ## The negative half: no joint table satisfies all three pair laws -/

/-- A table `p` on the eight-point space satisfies a property `P` almost
surely when it vanishes on every atom violating `P`. -/
def AESatisfies (p : Bool × Bool × Bool → ℚ) (P : Bool × Bool × Bool → Prop) : Prop :=
  ∀ x, ¬ P x → p x = 0

/-- **No joint law carries all three pair laws.**  Let `p` be a normalized
table on `{0,1}³` -- nonnegativity is not even needed.  If the off-diagonal
atoms of the pairs `(1,2)` and `(2,3)` and the diagonal atoms of the pair
`(1,3)` all carry zero mass, the total mass is at most zero, contradicting
normalization.  In particular no joint law on the three binary coordinates
has pair marginals supported as the chapter's three pair laws require. -/
theorem no_joint_law
    (p : Bool × Bool × Bool → ℚ)
    (hnorm : ∑ x, p x = 1)
    (h₁₂ : AESatisfies p fun x => x.1 = x.2.1)
    (h₂₃ : AESatisfies p fun x => x.2.1 = x.2.2)
    (h₁₃ : AESatisfies p fun x => x.1 ≠ x.2.2) :
    False := by
  -- Some atom carries positive mass, since the table sums to one.
  have hpos : ∃ x, 0 < p x := by
    by_contra hzero
    push Not at hzero
    have hsum_le : ∑ x, p x ≤ 0 := by
      calc ∑ x, p x ≤ ∑ _ ∈ univ, (0:ℚ) := Finset.sum_le_sum fun x _ => hzero x
        _ = 0 := Finset.sum_const_zero
    linarith
  obtain ⟨x, hx⟩ := hpos
  -- On that atom the three constraints force x.1 = x.2.1 = x.2.2 and x.1 ≠ x.2.2.
  have e₁₂ : x.1 = x.2.1 := by
    by_contra hne
    exact absurd (h₁₂ x hne) (ne_of_gt hx)
  have e₂₃ : x.2.1 = x.2.2 := by
    by_contra hne
    exact absurd (h₂₃ x hne) (ne_of_gt hx)
  exact absurd (h₁₃ x fun hne => hne (e₁₂.trans e₂₃)) (ne_of_gt hx)

/-- The chapter's statement verbatim, in marginal form: if the `(1,2)` and
`(2,3)` pair marginals of a normalized table put all mass on the diagonal and
the `(1,3)` pair marginal puts all mass off the diagonal, the three
constraints are contradictory. -/
theorem no_joint_law_marginals
    (p : Bool × Bool × Bool → ℚ)
    (hnonneg : ∀ x, 0 ≤ p x)
    (hnorm : ∑ x, p x = 1)
    (hm₁₂ : ∀ a b, a ≠ b → ∑ c : Bool, p (a, (b, c)) = 0)
    (hm₂₃ : ∀ b c, b ≠ c → ∑ a : Bool, p (a, (b, c)) = 0)
    (hm₁₃ : ∀ a c, a = c → ∑ b : Bool, p (a, (b, c)) = 0) :
    False := by
  refine no_joint_law p hnorm ?_ ?_ ?_
  · intro x hx
    have h := (Finset.sum_eq_zero_iff_of_nonneg fun c _ => hnonneg _).mp
      (hm₁₂ x.1 x.2.1 hx)
    exact h x.2.2 (Finset.mem_univ _)
  · intro x hx
    have h := (Finset.sum_eq_zero_iff_of_nonneg fun a _ => hnonneg _).mp
      (hm₂₃ x.2.1 x.2.2 hx)
    exact h x.1 (Finset.mem_univ _)
  · intro x hx
    have h := (Finset.sum_eq_zero_iff_of_nonneg fun b _ => hnonneg _).mp
      (hm₁₃ x.1 x.2.2 (not_not.mp hx))
    exact h x.2.1 (Finset.mem_univ _)

/-! ## The measure-theoretic form -/

section Measure

open MeasureTheory

/-- Equality events of measurable `Bool`-valued functions are measurable
(the codomain `Bool` is discrete, hence `T2`). -/
private theorem measurableSet_eq_bool {Ω : Type*} [MeasurableSpace Ω]
    {X Y : Ω → Bool} (hX : Measurable X) (hY : Measurable Y) :
    MeasurableSet {ω | X ω = Y ω} :=
  measurableSet_eq_fun hX hY

/-- A full-measure event has null complement, on a probability space. -/
private theorem measure_compl_eq_zero_of_ae {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {A : Set Ω}
    (hA : MeasurableSet A) (h : μ A = 1) : μ Aᶜ = 0 := by
  rw [measure_compl hA (by rw [h]; exact ENNReal.one_ne_top), h, measure_univ]
  norm_num

/-- **No probability space carries three such coordinates.**  For measurable
`Bool`-valued random variables on a probability space, the three full-measure
constraints `X₁ = X₂`, `X₂ = X₃`, `X₁ ≠ X₃` are contradictory; no joint law
of `(X₁, X₂, X₃)` has the three pair marginals the chapter describes. -/
theorem no_joint_law_measure {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] {X₁ X₂ X₃ : Ω → Bool}
    (hX₁ : Measurable X₁) (hX₂ : Measurable X₂) (hX₃ : Measurable X₃)
    (h₁₂ : μ {ω | X₁ ω = X₂ ω} = 1)
    (h₂₃ : μ {ω | X₂ ω = X₃ ω} = 1)
    (h₁₃ : μ {ω | X₁ ω ≠ X₃ ω} = 1) :
    False := by
  have z₁₂ : μ {ω | X₁ ω ≠ X₂ ω} = 0 := by
    rw [show {ω | X₁ ω ≠ X₂ ω} = {ω | X₁ ω = X₂ ω}ᶜ from by
      ext ω; simp [Set.mem_compl_iff, ne_eq]]
    exact measure_compl_eq_zero_of_ae μ (measurableSet_eq_bool hX₁ hX₂) h₁₂
  have z₂₃ : μ {ω | X₂ ω ≠ X₃ ω} = 0 := by
    rw [show {ω | X₂ ω ≠ X₃ ω} = {ω | X₂ ω = X₃ ω}ᶜ from by
      ext ω; simp [Set.mem_compl_iff, ne_eq]]
    exact measure_compl_eq_zero_of_ae μ (measurableSet_eq_bool hX₂ hX₃) h₂₃
  have hsub : {ω | X₁ ω ≠ X₃ ω} ⊆ {ω | X₁ ω ≠ X₂ ω} ∪ {ω | X₂ ω ≠ X₃ ω} := by
    intro ω hω
    show X₁ ω ≠ X₂ ω ∨ X₂ ω ≠ X₃ ω
    by_cases h12 : X₁ ω = X₂ ω
    · exact Or.inr fun h23 => hω (h12.trans h23)
    · exact Or.inl h12
  have hbound : (1:ENNReal) ≤ μ {ω | X₁ ω ≠ X₂ ω} + μ {ω | X₂ ω ≠ X₃ ω} := by
    calc (1:ENNReal) = μ {ω | X₁ ω ≠ X₃ ω} := h₁₃.symm
      _ ≤ μ ({ω | X₁ ω ≠ X₂ ω} ∪ {ω | X₂ ω ≠ X₃ ω}) := measure_mono hsub
      _ ≤ μ {ω | X₁ ω ≠ X₂ ω} + μ {ω | X₂ ω ≠ X₃ ω} := measure_union_le _ _
  rw [z₁₂, z₂₃] at hbound
  simp at hbound

end Measure

end OperatorGraph.PairLaw
