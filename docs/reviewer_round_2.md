# Independent Theory Review, Round 2

Date: 2026-07-12

Scope: `paper/theory_main.tex`, `docs/mathematical_foundations_and_prior_art.md`, and the proposed
fixed-marginal coupling architecture. Experiments remained paused.

## Verdict before revision

The weighted double-centering algebra and single-edge transport coupling are sound, but the draft
did not yet prove training-time separation of entropy from sparse signal in a dynamic neural
network. The missing object was the conditioning contract: the reference distribution, pair
score, and router could all observe the endpoint residues that the ANOVA statement purported to
vary. In that setting, double-centering proves only a local table gauge.

## P0 findings

1. **Conditional context was undefined.** A strict functional-ANOVA theorem requires a context
   `C_ij` excluding `X_i,X_j`, with marginals, scores, and edge selection measurable with respect
   to that context. Otherwise the network can route one-body information through dynamic context.
2. **Teacher and student used different gauges.** The MSA teacher used `p_theta`, while the
   student and weighted loss used detached EMA `bar p`. Both must be projected directly with the
   identical `bar p` during a training step.
3. **Gauge identifiability was conflated with sparsity.** The projector separates one-body and
   pair categorical function spaces under the stated reference. It neither makes the pair field
   sparse nor identifies the true edge support. Support sparsity is an additional model assumption.

## P1 findings

1. A multi-neighbor convex pool is a valid distribution and is marginal-calibrated in expectation
   only under fixed context-independent weights. It is not generally a posterior of a consistent
   joint model and is not pointwise marginal-preserving.
2. Sinkhorn/IPF requires positive marginals and kernel. Its first-order perturbation around
   independence is the weighted double-centered tangent field, but the exact layer is established
   generalized-KL/I-projection machinery.
3. The mutual-information quadratic expansion requires full support and a vanishing maximum
   relative perturbation. Its remainder deteriorates as the smallest marginal probability tends
   to zero.
4. The fixed-rank rejection was too strong. Fixed full rank `q-1` is valid, fixed low rank is a
   testable approximation, and adaptive rank has nontrivial selection cost. The zero-field case
   also needs an explicit rank-zero definition.

## Prior-art risk

The exact-coupling route is close to a combination of marginal log-linear models, IPF,
correspondence analysis, discrete dependence/copula models, differentiable Sinkhorn, and convex
conditional mixtures. Sei and Yano, *Minimum Information Dependence Modeling*, Bernoulli 2024,
is particularly close because it explicitly separates orthogonal marginal and dependence
parameters.

The plausible research contribution is therefore narrower:

> an efficient endpoint-excluding conditional neural operator with identifiable one-body/pair
> function spaces, sparse execution over sequence-position edges, and optional leave-query-out MSA
> supervision expressed in the identical model-defined reference.

This remains a research hypothesis, not a novelty claim.

## Solid claims after revision

- Weighted double-centering is a conditional pair-ANOVA projector for fixed full-support
  marginals and endpoint-excluding context.
- A single transport coupling exactly preserves its two prescribed marginals.
- Standardized categorical residual rank is at most `q-1`.
- Mutual information is exact dependence strength; the Frobenius expression is only a local
  second-order approximation.
- Pair-support sparsity and categorical approximation rank are independent modeling axes.
- MSA can be a training-only teacher while the forward contract remains single-sequence.
- Pairwise fixed-marginal couplings do not automatically define a globally consistent joint law.

## Required next theorem

Formalize the conditional functional-ANOVA result with an explicit context sigma-algebra and then
analyze which efficient endpoint-excluding context constructions can satisfy it without one
masked encoder pass per pair. This is now the central architecture problem.

## Post-revision verification

The revised theory now:

- distinguishes a deployable full-context local table gauge from a strict cavity model;
- conditions strict functional-ANOVA semantics on fixed, or jointly cavity-measurable, support;
- prevents endpoint information from re-entering the strict relation update through ordinary node
  aggregates;
- defines the shared cavity marginal using the complement of the active neighborhood;
- uses `bar p^E` consistently for strict student, MSA teacher, weighted norm, and spectrum;
- treats full rank `q-1` as capacity and low rank as an optional compression hypothesis.

Final narrow-review verdict: no remaining P0 or P1 inconsistency was identified within the stated
scope. The unresolved issue is constructive rather than algebraic: whether an efficient
endpoint-removable ordered summary can retain enough sequence information to make the strict
cavity model useful.
