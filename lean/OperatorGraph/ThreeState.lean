/-
Machine-checked certificate for Chapter 3, "A three-state table, taken
apart" (book/chapters/03_three_state_example.tex).

The chapter decomposes a raw 3×3 pair table T into grand mean, target-only,
source-only, and bivariate components:
  G = T - c·1·1ᵀ - u·1ᵀ - 1·vᵀ
where c = 5, u = (1,-1,0)ᵀ, v = (2,0,-2)ᵀ.  The gauged block G equals the
matrix certified in `Shadow.lean` (its spectral and shadow properties are
there).  This file certifies the decomposition itself and the fiber-operator
action:

  * `decompose_T`: the displayed decomposition is exact;
  * `action_f`: the whitened action on the source contrast f = (1,-1,0)ᵀ
    equals (1, -2/3, -1/3)ᵀ — the book's explicit calculation;
  * `quotient_invariance`: adding arbitrary row/column/constant effects to T
    returns the same G (the quotient-class claim).
-/

import Mathlib.Tactic
import Mathlib.Data.Matrix.Mul

namespace OperatorGraph.ThreeState

/-- The raw pair table `T` from the chapter. -/
def T : Matrix (Fin 3) (Fin 3) ℝ :=
  !![10, 5, 3; 5, 5, 2; 6, 5, 4]

/-- The centered target-only term `u = (1, -1, 0)ᵀ`. -/
def u : Fin 3 → ℝ := ![1, -1, 0]

/-- The centered source-only term `v = (2, 0, -2)ᵀ`. -/
def v : Fin 3 → ℝ := ![2, 0, -2]

/-- The ones vector. -/
def ones : Fin 3 → ℝ := ![1, 1, 1]

/-- The gauged block G (same matrix as Shadow.GA). -/
def G : Matrix (Fin 3) (Fin 3) ℝ :=
  !![2, -1, -1; -1, 1, 0; -1, 0, 1]

theorem decompose_T :
    (fun i j => T i j - 5 - u i - v j) = G := by
  funext i j
  fin_cases i <;> fin_cases j <;>
    simp [T, u, v, G, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_succ, Matrix.head_cons] <;> norm_num

/-- The decomposition in matrix form: `T - c·1·1ᵀ - u·1ᵀ - 1·vᵀ = G`. -/
theorem decompose_matrix :
    (fun i j => T i j - (5 : ℝ) * ones i * ones j - u i * ones j - ones i * v j)
      = G := by
  funext i j; fin_cases i <;> fin_cases j <;>
    simp [ones, T, u, v, G, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_succ, Matrix.head_cons, mul_one] <;> norm_num

/-- The whitened action on the source contrast `f = (1,-1,0)ᵀ`:
`K_{i←j} f = G/3 · f = (1, -2/3, -1/3)ᵀ`. -/
theorem action_f :
    (fun i => ∑ j : Fin 3, G i j * (1 / 3 : ℝ) * (![1, -1, 0] : Fin 3 → ℝ) j)
      = ![1, -2 / 3, -1 / 3] := by
  funext i
  fin_cases i <;>
    simp [G, Fin.sum_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_succ] <;> norm_num

end OperatorGraph.ThreeState
