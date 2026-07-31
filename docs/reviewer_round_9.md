# Theory Review Round 9: Pairwise Completeness and Higher-Order Residuals

Date: 2026-07-12

Scope: `paper/theory_main.tex`, section `Pairwise sparsity is not interaction completeness`, and
`docs/hierarchical_cavity_interactions.md`.

## Verdict

No P0 mathematical error was found. The product-reference ANOVA projectors, pair-truncation
identity, strictly positive pure third-order counterexample, and nonlinear non-closure argument
are valid. The review rejected treating the higher-order residual as an already implemented base
component.

## Corrections required and applied

1. The base strict operator is the explicit pairwise truncation
   `R_i^(>=2),E = 0`. This is a modeling assumption, not evidence that the population residual is
   zero.
2. A higher-order branch is an optional extension. Membership in the intended subspace requires
   an exact ANOVA projector or an equivalent structural parameterization; an arbitrary residual MLP
   is insufficient.
3. The raw tensor generator for the optional residual must use the same whole-neighborhood cavity
   and cannot observe active endpoint categories while generating its parameters.
4. The nonlinear example now assumes a nonzero target-centered contrast `P_i u`; otherwise the
   remaining term may be independent of the target category.
5. The MSA limitation is scoped to teacher pipelines retaining only one- and two-site marginal
   statistics. Full aligned rows contain higher-order information in principle.
6. In the fixed-marginal model, the aggregate one-body adjustment and potential are unique after
   gauge fixing; individual adjusting functions retain additive-constant freedom.
7. Probability-exact means preservation of specified one-site marginals within one local
   neighborhood model. It does not imply global consistency across targetwise neighborhoods.

## Design consequence

The higher-order result strengthens the audit boundary without forcing an exponentially expensive
architecture. The current trainable proposal remains pairwise. A higher-order branch should be
promoted only after an exact projection or structural parameterization, independent supervision,
and a tractable compute scheme are specified.
