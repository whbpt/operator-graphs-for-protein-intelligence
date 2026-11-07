/-
Machine-checked certificate for Chapter 14, "A four-genotype grammar for
experimental interaction" (book/chapters/14_epistasis_primer.tex).

The primer defines experimental epistasis in a declared scalar coordinate
as the mixed finite difference over the four-genotype table, and notes that
the log-coordinate tests a multiplicative endpoint model.  This file
certifies:

  * `additive_null`: an additive endpoint has zero epistasis in the
    identity coordinate (ring identity);
  * `log_coordinate_id`: in the log coordinate, epistasis equals the log of
    the double ratio Y(a,b)·Y(a₀,b₀)/(Y(a,b₀)·Y(a₀,b));
  * `multiplicative_null`: a multiplicative endpoint has zero epistasis in
    the log coordinate.
-/

import Mathlib.Tactic

namespace OperatorGraph.Epistasis

/-- Experimental epistasis: the mixed finite difference. -/
def epsilon (F : ℝ → ℝ → ℝ) (a b a₀ b₀ : ℝ) : ℝ :=
  F a b - F a b₀ - F a₀ b + F a₀ b₀

/-- **Additive endpoints have zero epistasis** in the identity coordinate. -/
theorem additive_null (g h c : ℝ) (a b a₀ b₀ : ℝ) :
    epsilon (fun x y => g * x + h * y + c) a b a₀ b₀ = 0 := by
  unfold epsilon; ring

/-- **The log-coordinate identity**: epistasis in the log coordinate is the
log of the double ratio. -/
theorem log_coordinate_id (w x y z : ℝ) (hw : 0 < w) (hx : 0 < x)
    (hy : 0 < y) (hz : 0 < z) :
    Real.log w - Real.log x - Real.log y + Real.log z
      = Real.log (w * z / (x * y)) := by
  have hwz : 0 < w * z := mul_pos hw hz
  have hxy : 0 < x * y := mul_pos hx hy
  have e1 : Real.log (w * z / (x * y)) = Real.log (w * z) - Real.log (x * y) :=
    Real.log_div (ne_of_gt hwz) (ne_of_gt hxy)
  have e2 : Real.log (w * z) = Real.log w + Real.log z := Real.log_mul (ne_of_gt hw) (ne_of_gt hz)
  have e3 : Real.log (x * y) = Real.log x + Real.log y := Real.log_mul (ne_of_gt hx) (ne_of_gt hy)
  rw [e1, e2, e3]
  ring

/-- **Multiplicative endpoints have zero epistasis in the log coordinate.**
For a multiplicative endpoint Y(a,b) = f(a)·g(b) with all four values
positive, the double ratio is 1 and the log-epistasis vanishes. -/
theorem multiplicative_null (f g : ℝ → ℝ) (a b a₀ b₀ : ℝ)
    (h1 : 0 < f a * g b) (h2 : 0 < f a * g b₀)
    (h3 : 0 < f a₀ * g b) (h4 : 0 < f a₀ * g b₀) :
    Real.log (f a * g b) - Real.log (f a * g b₀)
      - Real.log (f a₀ * g b) + Real.log (f a₀ * g b₀) = 0 := by
  have h := log_coordinate_id (f a * g b) (f a * g b₀) (f a₀ * g b) (f a₀ * g b₀)
    h1 h2 h3 h4
  rw [h]
  have hnum : f a * g b * (f a₀ * g b₀) = f a * g b₀ * (f a₀ * g b) := by ring
  have hratio : f a * g b * (f a₀ * g b₀) / (f a * g b₀ * (f a₀ * g b)) = 1 := by
    rw [hnum, div_self (by nlinarith)]
  rw [hratio, Real.log_one]

end OperatorGraph.Epistasis
