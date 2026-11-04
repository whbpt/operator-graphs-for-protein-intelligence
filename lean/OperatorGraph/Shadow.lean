/-
Machine-checked certificate for the Chapter 5 section "Two operators with the
same scalar shadow" (book/chapters/05_same_shadow_example.tex).

The book exhibits two gauged blocks on the uniform three-state reference,

  G_A = [[2,-1,-1],[-1,1,0],[-1,0,1]],
  G_B = √(5/3) · [[0,1,-1],[1,-1,0],[-1,0,1]],

with zero row and column means, ordinary squared Frobenius norm 10 each,
hence identical weighted scalar shadows M = √10/3 — yet different singular
structure and different actions.  This file certifies:

  * `fro_GA`, `fro_GB`: both squared Frobenius norms are 10;
  * `shadow_eq`: the shadows coincide, and `shadow_GA` fixes the value √10/3
    (whitened squared norm 10/9);
  * `row_means_GA`, `col_means_GA`, `row_means_GB`, `col_means_GB`: both
    blocks are doubly centered;
  * spectral witnesses: G_A^⊤ G_A has eigenvalues 9 and 1 on
    (2,-1,-1) and (0,1,-1) and vanishes on (1,1,1); B^⊤ B acts as 3 on the
    two contrast directions (1,-1,0), (0,1,-1) and vanishes on (1,1,1)
    (`spec_GA_1`, `spec_GA_2`, `spec_GA_ker`, `spec_Bm_1`, `spec_Bm_2`,
    `spec_Bm_ker`).  Whitening by 3 gives singular values {1, 1/3, 0} for
    G_A and {√5/3, √5/3, 0} for G_B.  `witness_det`: the three exhibited
    eigenvectors are independent (nonzero determinant), so the witness
    eigenvalues exhaust the spectrum;
  * `action_GA`, `action_GB`, `action_diff`: the two whitened actions on the
    source contrast f = (1,-1,0)ᵀ differ — the book's explicit display.

All theorems are kernel-checked.
-/

import Mathlib.Tactic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic

namespace OperatorGraph.Shadow

/-- The first gauged block `G_A`. -/
def GA : Matrix (Fin 3) (Fin 3) ℝ :=
  !![2, -1, -1; -1, 1, 0; -1, 0, 1]

/-- The unnormalized second block `B` (`G_B = √(5/3) · B`). -/
def Bm : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 1, -1; 1, -1, 0; -1, 0, 1]

/-- The second gauged block `G_B`. -/
noncomputable def GB : Matrix (Fin 3) (Fin 3) ℝ := Real.sqrt (5 / 3) • Bm

/-- Squared Frobenius norm. -/
def fro (M : Matrix (Fin 3) (Fin 3) ℝ) : ℝ := ∑ i, ∑ j, M i j ^ 2

/-- Weighted scalar shadow: the norm divided by 9 (uniform three-state
reference on both axes) under the square root. -/
noncomputable def shadow (M : Matrix (Fin 3) (Fin 3) ℝ) : ℝ := Real.sqrt (fro M / 9)

theorem fro_GA : fro GA = 10 := by norm_num [fro, GA, Fin.sum_univ_three]

theorem hsqrt : Real.sqrt 5 ^ 2 / Real.sqrt 3 ^ 2 = 5 / 3 := by
  rw [Real.sq_sqrt (by norm_num), Real.sq_sqrt (by norm_num)]

theorem hsqrt' : (Real.sqrt 5 / Real.sqrt 3) ^ 2 = 5 / 3 := by
  rw [div_pow, hsqrt]

theorem fro_GB : fro GB = 10 := by
  have hd : Real.sqrt (5 / 3) = Real.sqrt 5 / Real.sqrt 3 :=
    Real.sqrt_div (by norm_num : (0 : ℝ) ≤ 5) 3
  simp only [fro, GB, Matrix.smul_apply, hd, Fin.sum_univ_three]
  norm_num [Bm, hsqrt']

theorem shadow_eq : shadow GA = shadow GB := by
  unfold shadow
  rw [fro_GA, fro_GB]

theorem shadow_GA : shadow GA = Real.sqrt 10 / 3 := by
  rw [shadow, fro_GA, Real.sqrt_div (by norm_num : (0 : ℝ) ≤ 10) 9]
  norm_num

theorem row_means_GA : GA.mulVec ![1, 1, 1] = 0 := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, GA] <;> norm_num

theorem col_means_GA : GA.vecMul ![1, 1, 1] = 0 := by
  funext j; fin_cases j <;>
    simp [Matrix.vecMul, dotProduct, Fin.sum_univ_three, GA] <;> norm_num

theorem row_means_Bm : Bm.mulVec ![1, 1, 1] = 0 := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, Bm] <;> norm_num

theorem col_means_Bm : Bm.vecMul ![1, 1, 1] = 0 := by
  funext j; fin_cases j <;>
    simp [Matrix.vecMul, dotProduct, Fin.sum_univ_three, Bm] <;> norm_num

theorem mulVec_smul (c : ℝ) (M : Matrix (Fin 3) (Fin 3) ℝ) (v : Fin 3 → ℝ) :
    (c • M).mulVec v = c • M.mulVec v := by
  funext i
  simp only [Matrix.mulVec, dotProduct, Matrix.smul_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by ring

theorem smul_vecMul (c : ℝ) (M : Matrix (Fin 3) (Fin 3) ℝ) (v : Fin 3 → ℝ) :
    (c • M).vecMul v = c • M.vecMul v := by
  funext j
  simp only [Matrix.vecMul, dotProduct, Matrix.smul_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => by ring

theorem row_means_GB : GB.mulVec ![1, 1, 1] = 0 := by
  rw [GB, mulVec_smul, row_means_Bm, smul_zero]

theorem col_means_GB : GB.vecMul ![1, 1, 1] = 0 := by
  rw [GB, smul_vecMul, col_means_Bm, smul_zero]

/-! ## Spectral witnesses -/

theorem spec_GA_1 : (Matrix.transpose GA * GA).mulVec ![2, -1, -1] = (9 : ℝ) • ![2, -1, -1] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Matrix.mul_apply,
      Matrix.transpose_apply, Fin.sum_univ_three, GA] <;> norm_num

theorem spec_GA_2 : (Matrix.transpose GA * GA).mulVec ![0, 1, -1] = (1 : ℝ) • ![0, 1, -1] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Matrix.mul_apply,
      Matrix.transpose_apply, Fin.sum_univ_three, GA] <;> norm_num

theorem spec_GA_ker : (Matrix.transpose GA * GA).mulVec ![1, 1, 1] = 0 := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Matrix.mul_apply,
      Matrix.transpose_apply, Fin.sum_univ_three, GA] <;> norm_num

theorem spec_Bm_1 : (Matrix.transpose Bm * Bm).mulVec ![1, -1, 0] = (3 : ℝ) • ![1, -1, 0] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Matrix.mul_apply,
      Matrix.transpose_apply, Fin.sum_univ_three, Bm] <;> norm_num

theorem spec_Bm_2 : (Matrix.transpose Bm * Bm).mulVec ![0, 1, -1] = (3 : ℝ) • ![0, 1, -1] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Matrix.mul_apply,
      Matrix.transpose_apply, Fin.sum_univ_three, Bm] <;> norm_num

theorem spec_Bm_ker : (Matrix.transpose Bm * Bm).mulVec ![1, 1, 1] = 0 := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Matrix.mul_apply,
      Matrix.transpose_apply, Fin.sum_univ_three, Bm] <;> norm_num

/-- The three exhibited eigenvectors of `G_Aᵀ G_A` are linearly independent,
so the witness eigenvalues {9, 1, 0} exhaust the spectrum. -/
theorem witness_det :
    (!![(2 : ℝ), -1, -1; 0, 1, -1; 1, 1, 1]).det ≠ 0 := by
  simp [Matrix.det_fin_three]
  norm_num

/-! ## Actions on the same source contrast -/

theorem action_GA : GA.mulVec ![1, -1, 0] = ![3, -2, -1] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, GA] <;> norm_num

theorem action_Bm : Bm.mulVec ![1, -1, 0] = ![-1, 2, -1] := by
  funext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_three, Bm] <;> norm_num

theorem action_GB : GB.mulVec ![1, -1, 0]
    = Real.sqrt (5 / 3) • ![-1, 2, -1] := by
  rw [GB, mulVec_smul, action_Bm]

/-- The whitened actions differ: `K_A f ≠ K_B f` for `f = (1,-1,0)ᵀ`. -/
theorem action_diff :
    (1 / 3 : ℝ) • GA.mulVec ![1, -1, 0] ≠ (1 / 3 : ℝ) • GB.mulVec ![1, -1, 0] := by
  intro h
  have hs : (0 : ℝ) < Real.sqrt (5 / 3) := Real.sqrt_pos.2 (by norm_num)
  have h0 := congrFun h 0
  have hL : (1 / 3 : ℝ) * 3 = 1 := by norm_num
  have hR : (1 / 3 : ℝ) * (Real.sqrt (5 / 3) * (-1 : ℝ)) < 0 := by
    nlinarith [hs]
  have hcomp : ((1 / 3 : ℝ) • GA.mulVec ![1, -1, 0]) 0
      = ((1 / 3 : ℝ) • GB.mulVec ![1, -1, 0]) 0 := h0
  rw [action_GA] at hcomp
  rw [action_GB] at hcomp
  simp only [Pi.smul_apply, smul_smul, Pi.smul_apply, Matrix.cons_val_zero,
    smul_eq_mul] at hcomp
  linarith [hL, hR]

end OperatorGraph.Shadow
