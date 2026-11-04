/-
Machine-checked certificates for Chapter 5, "A three-variable ANOVA
calculation" (book/chapters/05_three_variable_anova_primer.tex).

The primer audits the target-centered score

  Φ(x_j, x_k) = φ₀ + A x_j + B x_k + C x_j x_k,

over two binary source coordinates x_j, x_k ∈ {-1, +1}, with coefficients in
a real inner product space `H` (the target contrast space H_i^0 with its
p_i-inner product is exactly such an `H`).  This file certifies:

  1. `walsh_orthonormal`: under the uniform product reference, the four
     characters 1, x_j, x_k, x_j x_k are orthonormal;
  2. `walsh_coefficients`: expectation against each character extracts the
     corresponding coefficient of Φ (Parseval for the four-term expansion);
  3. the response audit G(x_k) = A + C x_k: the background mean is `A`
     (`meanH_G`), the residual is C x_k (`residual_G`), and the dynamic
     variance is exactly ‖C‖² (`dynamic_variance`) — the primer's
     mean-residual and dynamic-variance equations;
  4. `correlated_cross`: under a correlated law μ with normalized total mass,
     E_μ‖A x_j + B x_k‖² = ‖A‖² + ‖B‖² + 2ρ⟨A, B⟩ with ρ = E_μ[x_j x_k] —
     the primer's correlated cross-term equation: orthogonal attribution of
     the two pair terms requires the product reference.

All facts are kernel-checked (no `native_decide` in this file).
-/

import Mathlib.Tactic
import Mathlib.Analysis.InnerProductSpace.Basic

namespace OperatorGraph.Anova

/-! ## Binary coordinates and the uniform product reference -/

/-- Value of a binary coordinate: `false ↦ -1`, `true ↦ +1`. -/
def s (b : Bool) : ℝ := if b then 1 else -1

@[simp] theorem s_false : s false = -1 := by simp [s]

@[simp] theorem s_true : s true = 1 := by simp [s]

/-- Uniform product expectation over the four atoms of `{-1, +1}²`. -/
noncomputable def E (f : Bool → Bool → ℝ) : ℝ :=
  (f false false + f false true + f true false + f true true) / 4

/-- The four characters `1, x_j, x_k, x_j x_k` are orthonormal under `E`. -/
theorem walsh_orthonormal :
    E (fun _ _ => (1:ℝ)) = 1
      ∧ E (fun a _ => s a) = 0
      ∧ E (fun _ b => s b) = 0
      ∧ E (fun a b => s a * s b) = 0
      ∧ E (fun a _ => s a * s a) = 1
      ∧ E (fun _ b => s b * s b) = 1
      ∧ E (fun a b => s a * s b * (s a * s b)) = 1
      ∧ E (fun a b => s a * (s a * s b)) = 0
      ∧ E (fun a b => s b * (s a * s b)) = 0 := by
  norm_num [E, s]

/-- Expectation against each character extracts the corresponding coefficient
of the four-term score `Φ = φ₀ + A x_j + B x_k + C x_j x_k`. -/
theorem walsh_coefficients (φ₀ A B C : ℝ) :
    ((E fun a b => φ₀ + A * s a + B * s b + C * s a * s b) = φ₀)
      ∧ ((E fun a b => (φ₀ + A * s a + B * s b + C * s a * s b) * s a) = A)
      ∧ ((E fun a b => (φ₀ + A * s a + B * s b + C * s a * s b) * s b) = B)
      ∧ ((E fun a b => (φ₀ + A * s a + B * s b + C * s a * s b) * (s a * s b)) = C) := by
  norm_num [E, s]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> ring

/-! ## The response audit `G(x_k) = A + C x_k` in a target inner product space -/

section HilbertSpace

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- Uniform mean over one binary coordinate, `H`-valued. -/
noncomputable def meanH (f : Bool → H) : H := (1 / 2 : ℝ) • (f false + f true)

/-- Uniform mean over one binary coordinate, real-valued. -/
noncomputable def mean2 (f : Bool → ℝ) : ℝ := (f false + f true) / 2

/-- The response to source `j` while holding `x_k` fixed:
`G_{i←j}(x_k) = A + C x_k`. -/
def G (A C : H) : Bool → H := fun k => A + s k • C

/-- The background-averaged response is the persistent direction `A`. -/
theorem meanH_G (A C : H) : meanH (G A C) = A := by
  simp only [meanH, G, s_false, s_true, neg_one_smul, one_smul]
  have h2 : (A + -C) + (A + C) = (2 : ℝ) • A := by
    calc (A + -C) + (A + C) = (A + A) + (C + -C) := by abel
      _ = A + A := by simp
      _ = (2 : ℝ) • A := (two_smul ℝ A).symm
  have h3 : (1 / 2 : ℝ) * 2 = 1 := by norm_num
  rw [h2, smul_smul, h3, one_smul]

/-- The response residual is the context-fluctuation direction `C x_k`. -/
theorem residual_G (A C : H) (k : Bool) : G A C k - meanH (G A C) = s k • C := by
  rw [meanH_G]
  simp [G]

theorem norm_smul_sq_one {c : ℝ} {x : H} (hc : c * c = 1) : ‖c • x‖ ^ 2 = ‖x‖ ^ 2 := by
  have h2 : (c - 1) * (c + 1) = 0 := by linear_combination hc
  rcases mul_eq_zero.mp h2 with h | h
  · rw [show c = 1 by linarith]
    simp
  · rw [show c = -1 by linarith]
    simp

/-- The dynamic variance is exactly the squared mass of the interaction
component containing the audited source `j` and the additional source `k`. -/
theorem dynamic_variance (A C : H) :
    mean2 (fun k => ‖G A C k - meanH (G A C)‖ ^ 2) = ‖C‖ ^ 2 := by
  have hr : ∀ k, G A C k - meanH (G A C) = s k • C := residual_G A C
  simp only [mean2, hr, s_false, s_true]
  have n1 : ‖(1 : ℝ) • C‖ ^ 2 = ‖C‖ ^ 2 := norm_smul_sq_one (by norm_num)
  have n2 : ‖(-1 : ℝ) • C‖ ^ 2 = ‖C‖ ^ 2 := norm_smul_sq_one (by norm_num)
  rw [n2, n1]
  ring

end HilbertSpace

/-! ## Correlated reference: the cross terms survive -/

/-- Expectation under a general law `μ` on the four atoms of `{-1, +1}²`. -/
noncomputable def Eμ (μ : Bool → Bool → ℝ) (f : Bool → Bool → ℝ) : ℝ :=
  μ false false * f false false + μ false true * f false true
    + μ true false * f true false + μ true true * f true true

/-- Both inner products `⟨x_j, x_k⟩_{L²(μ)}` and `⟨1, x_j x_k⟩_{L²(μ)}}`
reduce to the same covariance `E_μ[x_j x_k]`. -/
theorem inner_one_sjk (μ : Bool → Bool → ℝ) :
    Eμ μ (fun a b => (1:ℝ) * (s a * s b)) = Eμ μ (fun a b => s a * s b) := by
  simp [Eμ]

section HilbertSpace

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

private theorem atom_inner (a b : Bool) (A B : H) :
    ‖s a • A + s b • B‖ ^ 2
      = s a * s a * ‖A‖ ^ 2 + s b * s b * ‖B‖ ^ 2
        + (s a * s b + s b * s a) * (inner ℝ A B) := by
  simp only [← real_inner_self_eq_norm_sq]
  simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right]
  rw [real_inner_comm B A]
  ring

/-- The correlated cross-term equation: under a law `μ` with normalized total
mass (and any correlation ρ = E_μ[x_j x_k], nonzero in general), the expected
response norm keeps the cross term `2 ρ ⟨A, B⟩`.  Attribution of the two pair
terms to mutually orthogonal interaction orders requires the product
reference. -/
theorem correlated_cross (μ : Bool → Bool → ℝ) (A B : H)
    (h1 : Eμ μ (fun _ _ => (1:ℝ)) = 1) :
    Eμ μ (fun a b => ‖s a • A + s b • B‖ ^ 2)
      = ‖A‖ ^ 2 + ‖B‖ ^ 2 + 2 * Eμ μ (fun a b => s a * s b) * (inner ℝ A B) := by
  have hsa : ∀ a : Bool, s a * s a = 1 := fun a => by cases a <;> norm_num
  simp only [Eμ] at h1 ⊢
  simp only [atom_inner, hsa, one_mul]
  linear_combination (‖A‖ ^ 2 + ‖B‖ ^ 2) * h1

end HilbertSpace

end OperatorGraph.Anova
