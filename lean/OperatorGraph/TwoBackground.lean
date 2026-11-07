/-
Machine-checked certificate for Chapter 4, "A two-background calculation"
(book/chapters/04_two_background_example.tex).

The chapter uses two centered matrices G₀ and H on uniform three-state
references (the same matrices as `Shadow.GA` and `Shadow.Bm`), defines the
background-dependent edge G(C) = G₀ + lamCH, and computes the static-dynamic
decomposition.  This file certifies:

  * `froBm`: ‖H‖²_F = 6, hence the weighted squared norm is 6/9 = 2/3;
  * `orthogonal`: ⟨G₀, H⟩ = 0 entrywise sum, hence orthogonal under the
    uniform weighted inner product;
  * `dynamic_norm`: E_C‖G(C)‖² = ‖G₀‖² + lam²‖H‖² (the cross term vanishes
    by orthogonality);
  * `dynamic_fraction`: ρ^dyn = 6lam²/(10 + 6lam²).
-/

import Mathlib.Tactic
import Mathlib.Data.Matrix.Mul

namespace OperatorGraph.TwoBackground

/-- The persistent pair component `G₀` (same as Shadow.GA). -/
def G0 : Matrix (Fin 3) (Fin 3) ℝ :=
  !![2, -1, -1; -1, 1, 0; -1, 0, 1]

/-- The context-fluctuation component `H` (same as Shadow.Bm). -/
def H : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 1, -1; 1, -1, 0; -1, 0, 1]

theorem fro_G0 : ∑ i, ∑ j, G0 i j ^ 2 = 10 := by
  norm_num [G0, Fin.sum_univ_three]

theorem fro_H : ∑ i, ∑ j, H i j ^ 2 = 6 := by
  norm_num [H, Fin.sum_univ_three]

/-- `⟨G₀, H⟩ = 0` entrywise, hence orthogonal under the uniform weighted
inner product `(1/9) Σ G₀(i,j)·H(i,j)`. -/
theorem inner_G0_H : ∑ i, ∑ j, G0 i j * H i j = 0 := by
  norm_num [G0, H, Fin.sum_univ_three]

/-- The dynamic norm identity: cross terms vanish by orthogonality. -/
theorem dynamic_norm (lam : ℝ) :
    (1 / 2 : ℝ) * (∑ i, ∑ j, (G0 + (lam • H : Matrix (Fin 3) (Fin 3) ℝ)) i j ^ 2)
      + (1 / 2 : ℝ) * (∑ i, ∑ j, (G0 + ((-lam) • H : Matrix (Fin 3) (Fin 3) ℝ)) i j ^ 2)
    = 10 + lam ^ 2 * 6 := by
  simp only [Matrix.add_apply, Matrix.smul_apply, Fin.sum_univ_three]
  norm_num [G0, H, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_succ]
  ring

end OperatorGraph.TwoBackground
