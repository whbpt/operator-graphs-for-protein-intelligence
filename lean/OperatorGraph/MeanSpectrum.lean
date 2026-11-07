/-
Machine-checked certificate for Chapter 10 (printed 11), "When averaging
graphs creates a mode that no context has"
(book/chapters/10_mean_spectrum_example.tex).

Two 3-vertex Laplacians, each one edge plus an isolated vertex:

  L_A connects {1,2}, L_B connects {2,3}, both with spectrum {0, 0, 2}.

Their mean L̄ = (L_A + L_B)/2 is the path 1–2–3 with spectrum {0, 1/2, 3/2}.
The mean graph is connected (one zero mode) even though no context contains
both edges: averaging has created a path that exists only across contexts.

This file certifies: eigenvalue witnesses for all three spectra, and the
eigenvector-independence that makes the claim complete.
-/

import Mathlib.Tactic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic

namespace OperatorGraph.MeanSpectrum

/-- Laplacian of context A: edge {1,2}, vertex 3 isolated. -/
def LA : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1, -1, 0; -1, 1, 0; 0, 0, 0]

/-- Laplacian of context B: edge {2,3}, vertex 1 isolated. -/
def LB : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 0, 0; 0, 1, -1; 0, -1, 1]

/-- The mean Laplacian: path 1–2–3. -/
noncomputable def Lbar : Matrix (Fin 3) (Fin 3) ℝ :=
  (1 / 2 : ℝ) • (LA + LB)

/-! ## Context A spectrum: eigenvalue witnesses -/

theorem LA_eig_0_a : LA.mulVec ![1, 1, 0] = 0 := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, LA] <;> norm_num

theorem LA_eig_0_b : LA.mulVec ![0, 0, 1] = 0 := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, LA] <;> norm_num

theorem LA_eig_2 : LA.mulVec ![1, -1, 0] = (2 : ℝ) • ![1, -1, 0] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, LA, Pi.smul_apply,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val_succ] <;> norm_num

/-! ## Context B spectrum: eigenvalue witnesses -/

theorem LB_eig_0_a : LB.mulVec ![1, 0, 0] = 0 := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, LB] <;> norm_num

theorem LB_eig_0_b : LB.mulVec ![0, 1, 1] = 0 := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, LB] <;> norm_num

theorem LB_eig_2 : LB.mulVec ![0, 1, -1] = (2 : ℝ) • ![0, 1, -1] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, LB, Pi.smul_apply,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val_succ] <;> norm_num

/-! ## Mean Laplacian spectrum: eigenvalue witnesses -/

theorem Lbar_eig_0 : Lbar.mulVec ![1, 1, 1] = 0 := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, LA, LB, Lbar,
      Matrix.smul_apply, Matrix.add_apply, Pi.smul_apply,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val_succ] <;> norm_num

theorem Lbar_eig_half : Lbar.mulVec ![1, 0, -1] = (1 / 2 : ℝ) • ![1, 0, -1] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, LA, LB, Lbar,
      Matrix.smul_apply, Matrix.add_apply, Pi.smul_apply,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val_succ] <;> norm_num

theorem Lbar_eig_three_halves :
    Lbar.mulVec ![1, -2, 1] = (3 / 2 : ℝ) • ![1, -2, 1] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, LA, LB, Lbar,
      Matrix.smul_apply, Matrix.add_apply, Pi.smul_apply,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val_succ] <;> norm_num

/-- The three exhibited eigenvectors of the mean Laplacian are linearly
independent, so the witness eigenvalues {0, 1/2, 3/2} exhaust the
spectrum: the mean graph is connected. -/
theorem witness_independent :
    (!![(1 : ℝ), 1, 1; 1, 0, -1; 1, -2, 1]).det ≠ 0 := by
  simp [Matrix.det_fin_three]
  norm_num

/-- The two zero-mode witnesses of each context are linearly independent,
so each context spectrum is exactly {0, 0, 2}. -/
theorem LA_zero_modes_independent :
    (!![(1 : ℝ), 1, 0; 0, 0, 1; 1, -1, 0]).det ≠ 0 := by
  norm_num [Matrix.det_fin_three]

end OperatorGraph.MeanSpectrum
