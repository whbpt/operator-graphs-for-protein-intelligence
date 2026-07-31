# Independent Theory Review, Round 5

Date: 2026-07-12

Scope:

- `paper/theory_main.tex`
- `docs/global_support_no_go_and_directed_design.md`
- `docs/endpoint_excluding_context_architecture.md`
- `docs/mathematical_foundations_and_prior_art.md`
- `docs/prior_art_matrix.md`

Experiments remained paused. The review tested the global support no-go, targetwise versus global
semantics, directional gauges, categorical rank, and impartial-selection prior art.

## Verdict

The global undirected no-go theorem is correct under its stated strong semantics, but the first
directed strict architecture contained a critical context/reference mismatch. That mismatch has
been corrected by replacing pair-only values and target-j references with one target-relative
whole-neighborhood cavity. The architecture is now internally consistent as a targetwise
conditional-ANOVA model relative to a stated product reference. This is still a research
hypothesis, not a novelty or empirical-success claim.

## P0 findings and resolution

### Right-reference endpoint leakage

The rejected formula used target `j`'s own node-cavity reference on the right:

\[
G_{i\leftarrow j}=P_{\bar p_i^E}S_{i\leftarrow j}P_{\bar p_j^E}^{\top}.
\]

For directed non-reciprocal routing, `j in N_i` does not imply `i in N_j`. Therefore
`bar p_j^E` could observe `x_i`, so the projector itself changed when the left endpoint was
varied. This invalidated even the per-arc strict theorem.

Resolution: for target `i`, all references are now predicted from the same target cavity:

\[
\mathcal C_i^E=\sigma(X_{-[\{i\}\cup N_i]}),
\qquad
\bar p_{u\mid i}^E(\cdot\mid\mathcal C_i^E),
\quad u\in\{i\}\cup N_i.
\]

The directional field and teacher use the identical reference pair:

\[
G_{i\leftarrow j}
=P_{\bar p_{i\mid i}^E}S_{i\leftarrow j}(c_i^E)
P_{\bar p_{j\mid i}^E}^{\top},
\]

\[
G_{i\leftarrow j}^{T,E}
=P_{\bar p_{i\mid i}^E}L_{ij}^{MSA}
P_{\bar p_{j\mid i}^E}^{\top}.
\]

Teacher, student, weighted norm, and spectrum must use this same directional gauge.

### Pair-only context was weaker than the targetwise claim

A score based on `c_ij` may read another selected source `x_k`. Then `x_k` can both select its own
message column and modulate `G_{i<-j}`. This is a valid pair-conditional construction after using
pair-specific references, but it is not an additive decomposition over the complete selected
source set.

Resolution: the strict main variant makes every router output, background, value, gate, and
reference for target `i` measurable with respect to `C_i^E`. Selected categories enter only in
their final column lookups. The pair-only construction is retained only as a weaker per-arc gauge.

## P1 findings and resolution

### The no-go does not force directionality

The theorem rules out a nonconstant, jointly graph-invariant undirected router if one realized
graph has no isolated vertices. It does not rule out nontrivial reciprocal targetwise routing,
because a coordinate may change edges among its non-neighbors without changing its own incident
set.

Resolution: the paper now describes directionality as a conservative semantic design choice,
not a theorem consequence. Reciprocal or symmetrized contact outputs remain possible, but do not
upgrade the operator to a globally identified graph.

### Directionally re-gauged teachers need not be transposes

Raw MSA log-density-ratio fields are transpose-related. After projection under different
target-relative references, opposite directional teachers are exact transposes only when the two
reference pairs coincide. The paper and architecture notes now state this qualification.

### Capacity dimension and matrix rank were conflated

For proteins, the complete interaction space is a `19 x 19` tangent-product space with 361 scalar
degrees of freedom and maximum matrix rank 19. The corrected documents no longer call this a
"19-dimensional interaction." Fixed or adaptive low rank is described as compression or
regularization, never as a consequence of disentanglement or identifiability.

### Impartial selection is only an analogy

Impartial selection and pathwise targetwise support invariance are not formally ordered. They
have different input, output, randomization, and strategic semantics. The prior-art wording now
uses impartial selection only as a self-influence-free design analogy.

## Residual risks

- The targetwise theorem is relative to a model-defined conditional product reference; it does
  not prove that true selected sources are conditionally independent.
- Orthogonal logit components do not imply an additive decomposition of final predictive entropy.
- Exact endpoint deletion does not establish adequate approximation power for a compact affine
  recurrence state.
- A learnable, efficient certificate router satisfying pathwise source-set invariance remains
  unresolved.
- The proposed combination remains subject to further prior-art search and cannot yet carry a
  novelty claim.

## Review status

The P0 formula and its dependent teacher/context statements were revised before acceptance. The
reviewer found no flaw in the global no-go proof itself after its scope was narrowed and the
directed design was presented as a choice rather than a logical necessity.
