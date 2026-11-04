/-
Machine-checked certificate for the monograph's Proposition
"Refinement reorganizes; it does not recover"
(prop:book-opendde-no-recovery, book/chapters/08_three_alphafolds.tex).

The proposition: if the refined structural states are computed from the
deposited condition alone — deposited = f(evidence), refined = g(deposited) —
then evidence → deposited → refined is a Markov chain and

  I(evidence; refined) ≤ I(evidence; deposited).

mathlib v4.35 carries no Shannon-entropy layer, so this file certifies the
proposition's operative core, self-contained over finite types:

  * `entropy_processed_le`: for any nonnegative mass `P` on a finite type and
    any function `g`, the entropy of the pushed-forward mass satisfies
    `H(g#P) ≤ H(P)` — processing cannot increase entropy.  The proof is a
    termwise log-sum argument: `P a ≤ (g#P)(g a)` fiberwise, so
    `Σ a, P a log P a ≤ Σ a, P a log ((g#P)(g a)) = Σ b, (g#P) b log (g#P) b`,
    the last step by regrouping over fibers.

  * `no_recovery`: in the book's deterministic-composition setting both
    mutual informations reduce to entropies of the deposited and refined
    images (standard: `I(E; h(E)) = H(h(E))` for deterministic `h`), so the
    displayed inequality is exactly `H((g∘f)#P) ≤ H(f#P)` — a role tier that
    reads only the deposited state can reorganize relational memory but
    cannot recover information removed by the residue-level compression.

Normalization is not needed: the inequality is homogeneous, so masses are
only assumed nonnegative (a probability mass is the special case summing to
one).  All theorems are kernel-checked.
-/

import Mathlib.Tactic

open Finset

namespace OperatorGraph.NoRecovery

variable {α β γ : Type*} [Fintype α] [Fintype β] [Fintype γ]
  [DecidableEq β] [DecidableEq γ]

/-- Pushforward of a mass along a function: total mass of each fiber. -/
def push (g : α → β) (P : α → ℝ) : β → ℝ :=
  fun b => ∑ a ∈ univ.filter fun a => g a = b, P a

/-- Shannon entropy of a nonnegative mass, with the `0 log 0 = 0` convention
(which `Real.log` implements via `log_nonpos`). -/
noncomputable def entropy (P : α → ℝ) : ℝ := -∑ a, P a * Real.log (P a)

theorem push_nonneg {g : α → β} {P : α → ℝ} (hP : ∀ a, 0 ≤ P a) (b : β) :
    0 ≤ push g P b :=
  Finset.sum_nonneg fun a _ => hP a

/-- Each atom's mass is bounded by the mass of its fiber. -/
theorem le_push_self {g : α → β} {P : α → ℝ} (hP : ∀ a, 0 ≤ P a) (a : α) :
    P a ≤ push g P (g a) :=
  Finset.single_le_sum (fun a' _ => hP a')
    (Finset.mem_filter.2 ⟨Finset.mem_univ a, rfl⟩)

/-- **Processing cannot increase entropy.** -/
theorem entropy_processed_le (g : α → β) (P : α → ℝ) (hP : ∀ a, 0 ≤ P a) :
    entropy (push g P) ≤ entropy P := by
  -- Termwise monotonicity inside fibers, then regrouping over fibers.
  have hmono : ∑ a, P a * Real.log (P a) ≤ ∑ a, P a * Real.log (push g P (g a)) := by
    refine Finset.sum_le_sum fun a _ => ?_
    by_cases h0 : P a = 0
    · simp [h0]
    · have hpos : 0 < P a := by
        rcases eq_or_lt_of_le (hP a) with h | h
        · exact absurd h.symm h0
        · exact h
      have h1 : P a ≤ push g P (g a) := le_push_self hP a
      exact mul_le_mul_of_nonneg_left (Real.log_le_log hpos h1) (hP a)
  have hregroup : ∑ a, P a * Real.log (push g P (g a))
      = ∑ b, push g P b * Real.log (push g P b) := by
    have e1 : ∑ b, ∑ a ∈ univ.filter fun a => g a = b,
          P a * Real.log (push g P (g a))
        = ∑ a, P a * Real.log (push g P (g a)) :=
      Finset.sum_fiberwise univ g fun a => P a * Real.log (push g P (g a))
    rw [← e1]
    refine Finset.sum_congr rfl fun b _ => ?_
    have hfiber : ∀ a ∈ univ.filter fun a => g a = b,
        P a * Real.log (push g P (g a)) = P a * Real.log (push g P b) := by
      intro a ha
      rw [(Finset.mem_filter.1 ha).2]
    rw [Finset.sum_congr rfl hfiber, ← Finset.sum_mul, ← push]
  unfold entropy
  linarith [hmono.trans_eq hregroup]

/-- Pushforward along a composition is the composition of pushforwards. -/
theorem push_comp (g : β → γ) (f : α → β) (P : α → ℝ) :
    push (g ∘ f) P = push g (push f P) := by
  funext c
  simp only [push]
  rw [Finset.sum_fiberwise_eq_sum_filter univ
    (univ.filter fun b => g b = c) f P]
  refine Finset.sum_congr ?_ fun _ _ => rfl
  ext a
  simp [Finset.mem_filter, Function.comp_apply]

/-- The book's no-recovery inequality in its deterministic-composition form:
the refined image of the evidence mass is a further pushforward of the
deposited image, so its entropy — the mutual information the refined tier
carries about the evidence — cannot exceed that of the deposited state. -/
theorem no_recovery (g : β → γ) (f : α → β) (P : α → ℝ) (hP : ∀ a, 0 ≤ P a) :
    entropy (push (g ∘ f) P) ≤ entropy (push f P) := by
  rw [push_comp]
  exact entropy_processed_le g (push f P) (push_nonneg hP)

end OperatorGraph.NoRecovery
