# Theory Review Round 8: Compact Recurrence Capacity

Date: 2026-07-12

Scope: the section `Compact context is not a fixed intrinsic rank` in `paper/theory_main.tex` and
the corresponding section of `docs/removable_context_theorems.md`.

## Verdict

No P0 remains after revision. The section rigorously supports rejecting a universal fixed low
intrinsic rank. It does not prove that adaptive rank is necessary or superior.

## Corrections required and applied

1. The tensor-train cut-rank theorem applies to matrix-product representations with a linear
   terminal readout. An arbitrary nonlinear decoder does not inherit the bound.
2. The equality between unfolding ranks and minimal bond dimensions concerns unconstrained TT
   cores. It does not characterize minimal affine-recurrence width because augmented affine cores
   have additional structural constraints.
3. The singular-tail lower bound is stated for fixed length and target tensor, separately at each
   cut and with the corresponding cut width.
4. The squared-loss projection identity assumes a measurable summary and a square-integrable
   target. The log-loss identity assumes regular conditional laws and finite Bayes log risks.
5. Uniform contraction proves sensitivity decay. Effective forgetting additionally needs finite
   precision, state noise, or a bounded-gain decoder; nonlinear recurrences use Jacobian products.
6. The shared-core statement is limited to a linear simultaneous-realization class with shared
   token transitions and target-specific boundary/readout vectors. A sharp simultaneous lower
   bound is not claimed.
7. Variable-width blocks remain an optional compute strategy. The mathematics does not require
   one approximation rank for every context, but it also does not establish adaptive-width
   superiority.

## Supported design conclusion

- Categorical matrix rank and recurrence state width are independent capacity axes.
- Full `19 x 19` categorical tangent-product capacity remains the protein default on selected
  arcs.
- A fixed small linear context width is justified only when the relevant cutwise spectral tails
  are acceptably small.
- General nonlinear context compression should be judged by target-relative Bayes-risk gaps.
- A fixed `D_max` is an implementation ceiling, not an assertion of universal intrinsic rank.
