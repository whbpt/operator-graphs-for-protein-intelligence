import Mathlib.Tactic

namespace OperatorGraph.Bayes

/-- **No deterministic decision beats the pointwise minimum. -/
theorem bayes_floor_bound (η : ℝ) (d : Bool) :
    min (1 - η) η ≤ if d then 1 - η else η := by
  match d with
  | true => exact min_le_left _ _
  | false => exact min_le_right _ _

/-- **Randomization cannot improve the bound.**  A mixed strategy's
conditional error is a convex combination of the two deterministic errors,
hence at least the min. -/
theorem randomized_convex (η ρ : ℝ) (hρ : 0 ≤ ρ) (hρ' : ρ ≤ 1) :
    min (1 - η) η ≤ ρ * (1 - η) + (1 - ρ) * η := by
  have h1 : min (1 - η) η ≤ ρ * (1 - η) + (1 - ρ) * min (1 - η) η := by
    have t1 : ρ * min (1 - η) η ≤ ρ * (1 - η) :=
      mul_le_mul_of_nonneg_left (min_le_left _ _) hρ
    have t2 : (1 - ρ) * min (1 - η) η ≤ (1 - ρ) * min (1 - η) η := le_refl _
    linarith
  have h2 : min (1 - η) η ≤ ρ * min (1 - η) η + (1 - ρ) * η := by
    have t1 : ρ * min (1 - η) η ≤ ρ * min (1 - η) η := le_refl _
    have t2 : (1 - ρ) * min (1 - η) η ≤ (1 - ρ) * η :=
      mul_le_mul_of_nonneg_left (min_le_right _ _) (by linarith)
    linarith
  have hsplit : ρ * (1 - η) + (1 - ρ) * η
      = (ρ * (1 - η) + (1 - ρ) * min (1 - η) η) + (ρ * min (1 - η) η + (1 - ρ) * η
          - (ρ * min (1 - η) η + (1 - ρ) * min (1 - η) η)) := by
    ring
  have hbase : ρ * min (1 - η) η + (1 - ρ) * min (1 - η) η = min (1 - η) η := by
    have hρsum : ρ + (1 - ρ) = 1 := by ring
    have hfac : ρ * min (1 - η) η + (1 - ρ) * min (1 - η) η
        = (ρ + (1 - ρ)) * min (1 - η) η := by ring
    rw [hfac, hρsum, one_mul]
  rw [hsplit, hbase]
  linarith

end OperatorGraph.Bayes
