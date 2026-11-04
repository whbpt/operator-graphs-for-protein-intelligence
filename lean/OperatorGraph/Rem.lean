import Mathlib.Tactic

open Finset

namespace OperatorGraph.Rem

/-- **The order-profile summation identity**: the binomial theorem at `b=1`
gives `Σ_k C(L,k)(q-1)^k = q^L`. -/
theorem order_profile_sum (L : ℕ) (q : ℕ) (hq : 1 ≤ q) :
    ∑ k ∈ range (L + 1), L.choose k * (q - 1) ^ k = q ^ L := by
  have h := (add_pow (q - 1 : ℕ) 1 L).symm
  simp only [one_pow, one_mul, Nat.sub_add_cancel hq] at h
  simpa [mul_comm] using h

end OperatorGraph.Rem
