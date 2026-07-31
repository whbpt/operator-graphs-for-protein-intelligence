# Independent Theory Review, Round 7

Date: 2026-07-12

Scope:

- `docs/support_recoverability_and_envelope_theory.md`
- the single-sequence recoverability section of `paper/theory_main.tex`
- related training-reward and conclusion text
- `docs/mathematical_foundations_and_prior_art.md`
- `docs/prior_art_matrix.md`

Experiments remained paused. The review tested edgewise measurability, Bayes and information
bounds, the envelope decision rule, scout acquisition, and optional-MSA semantics.

## Verdict

No P0 issue was found. On the full product domain, strict targetwise invariance implies that the
membership of candidate `j` is measurable with respect to `C_ij=X_-{i,j}`. The binary cavity Bayes
floor, Hamming lower bound, leafwise posterior decision, and Bellman opportunity-cost state are
correct after the qualifications recorded below.

The section now directly answers an important project boundary: optional MSA training labels do
not make an exact family interaction graph recoverable from one sequence.

## Randomized-router condition

For each fixed seed, pathwise measurability holds. The teacher-relative Bayes bound additionally
requires

\[
\omega\perp(F,X,T,W),
\]

or at least conditional independence from the teacher given the endpoint-deleted observation.
Independence from `X` alone is insufficient because a correlated external seed could encode
teacher or latent-family information. The revised theorem uses the stronger assumption.

## Bayes-risk bound

For

\[
\eta_{ij}(c)=P(T_{ij}=1\mid C_{ij}=c),
\]

every strict membership predictor satisfies

\[
P(Z_{ij}\ne T_{ij})
\ge
E\min\{\eta_{ij}(C_{ij}),1-\eta_{ij}(C_{ij})\}.
\]

Summing gives the Hamming lower bound. Independent randomization cannot improve conditional
binary 0-1 risk under the seed-independence condition.

The difference between endpoint-deleted and full-sequence Bayes error is teacher-relative and
edgewise. It does not include bounded-transcript compression, joint certificate restrictions,
degree constraints, model approximation, or optimization error. Those can only add risk.

## Conditional information and exact recovery

The quantity

\[
I(T_{ij};X_i,X_j\mid C_{ij})
\]

measures additional teacher-label information in the endpoints. It is not causal interaction and
positive conditional mutual information need not increase binary classification error if the
posterior does not cross the Bayes decision boundary.

Zero-error strict recovery is impossible whenever

\[
0<P(T_{ij}=1\mid C_{ij})<1
\]

on a positive-probability set of endpoint-deleted contexts.

The Fano formula is correct, but the support alphabet size must be the number of support sets with
positive probability. `choose(L-1,k)` is only the maximal size when all exact size-`k` sets are
possible. A large alphabet alone does not imply difficulty; the operative term is conditional
entropy.

## Envelope theorem

Under

\[
\mathcal L_{env}(N,W_i)
=\sum_{j\ne i,\,j\notin N}W_{ij}+\lambda|N|,
\]

the Bayes-optimal certificate leaf chooses up to `k` unqueried positions with the largest positive
values of

\[
E[W_{ij}\mid H]-\lambda.
\]

For a fixed size `k`, it chooses the top-`k` posterior expected weights. This is an established
Bayes decision under additive loss. It does not prove set recall, stability, calibration, or
biological correctness. The accepted name is **budgeted posterior teacher-mass envelope**.
Stability comes separately from the certificate theorem.

## Bellman correction

Let

\[
A(H,Q)=[L]\setminus(\{i\}\cup Q).
\]

The stop value sorts only candidates in `A(H,Q)`:

\[
V_{stop}(H,Q)
=\sum_{j\ne i}m_j(H)
-\sum_{r=1}^{k}[m_{(r)}(H)-\lambda]_+.
\]

Queried coordinates remain in the first missed-mass term but cannot enter the captured-mass term,
which records their source-opportunity cost. With `b` queries remaining,

\[
V_0=V_{stop},
\]

and `V_b` recurses to `V_{b-1}` after each query. Query cost and a fixed query budget may coexist.

Conditional mutual information is a generic information-gain heuristic, not the exact objective:
it is generally different from reduction in envelope Bayes loss, and this problem additionally
forfeits the queried coordinate as a source.

## MSA route/value separation

The route weight and categorical value teacher are different objects:

- `W_ij` quantifies the training-relative cost of omitting an arc;
- `G_{i<-j}^{T,E}` supervises the signed categorical field after the envelope and cavity are fixed.

If `W_ij` is derived from an MSA categorical field, its gauge, marginal normalization, and
geometry must be stated. Otherwise it remains an entropy-contaminated routing heuristic. Neither
MSA weights, structure contacts, nor mutation scores automatically define sparse biological truth.

## Prior-art verdict

Bayes top-k decisions, Fano bounds, information-theoretic feature selection, active feature
acquisition, and conditional-information query policies are established. The project-specific
candidate is their use for selecting unqueried endpoints that then define a whole-neighborhood
categorical cavity. The search remains non-exhaustive and does not support a novelty claim.

## Direction check

This work returns the project to its central question rather than adding router detail for its own
sake. The resulting chain is:

1. strict routing limits available single-sequence information;
2. teacher-relative Bayes risk quantifies an irreducible recovery floor;
3. the router predicts a budgeted teacher-mass envelope;
4. the envelope defines the whole-neighborhood cavity;
5. the cavity background and directional gauge provide function-space separation;
6. MSA supplies optional route and value training targets only.

The envelope is a compute candidate set, not an identified sparse biological signal. The strict
entropy/interaction separation still comes from the cavity reference and categorical projection.
