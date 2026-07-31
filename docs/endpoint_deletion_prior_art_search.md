# Prior-Art Search: Endpoint-Deleted Sequence Context

Date: 2026-07-12

## Question searched

Is there an existing neural sequence architecture that uses associative/range products to compute
the representation of a sequence with arbitrary endpoint tokens replaced by identities, then uses
that representation as a pairwise categorical interaction context?

## Search routes

Targeted searches were run over arXiv and scholarly metadata using combinations of:

- `segment tree` with `state space`, `recurrent`, and `neural`;
- `range product` with `neural` and `dynamic sequence`;
- `token deletion` with `sequence model`;
- `leave-one-out representation` and `leave-one-out attention`;
- `associative scan`, `affine recurrence`, and `data-controlled linear recurrence`.

These searches are evidence gathering, not a proof of novelty. Terminology may differ and the
search is not exhaustive.

## Closest engineering prior art

### Linear Transformers and Performers

They rewrite attention through additive kernel sufficient statistics. Direct subtraction of
token-local key-value terms gives algebraic leave-out summaries for one layer. This does not remain
endpoint-exact after ordinary contextual stacking because retained token states can already contain
the deleted source.

### S5

S5 uses a multi-input/multi-output state-space model with efficient parallel scans. It establishes
scan-compatible sequence recurrence but does not propose arbitrary pair deletion or a categorical
cavity gauge.

### GateLoop

GateLoop uses fully data-controlled linear recurrences, cumulative products, and optimized
associative scan. Its paper explicitly presents `O(L)` recurrent and `O(L log L)` parallel modes.
This is the closest recurrence algebra to the proposed segment-product context. The located paper
does not use identity replacement plus range queries to construct endpoint-excluded pair contexts.

### Mamba

Mamba makes state-space parameters input-dependent and supplies a hardware-aware parallel scan.
Its complete block includes components beyond a token-local affine recurrence. Those components
are not automatically deletion-exact. The affine selective recurrence core is relevant prior art;
the cavity use is not established by the paper.

### Classical segment trees

Balanced trees for associative range products are standard data structures. Replacing deleted
leaves by identities and composing the complement intervals is a direct application of this
classical machinery.

## Closest statistical prior art

Sei and Yano's minimum-information dependence model is closer to the statistical goal than any
sparse-attention paper. It provides:

- arbitrary prescribed marginals;
- an exponential dependence statistic;
- one-body adjusting terms and a potential determined up to their stated gauge;
- Fisher-orthogonal marginal and dependence parameters;
- conditional inference;
- a connection to multi-marginal entropic optimal transport;
- the finite categorical/log-linear special case.

Therefore fixed-marginal dependence, marginal/dependence orthogonality, adjusting functions, and
the entropic-transport connection are prior art.

## Current boundary

No close precedent was located in this targeted search for the full combination:

1. fixed or jointly cavity-measurable sparse candidate support;
2. token-local affine maps stored as associative interval products;
3. arbitrary node/pair endpoint deletion by identity replacement and retained-range queries;
4. model-defined categorical marginal gauge;
5. optional leave-query-out MSA supervision with single-sequence inference.

This is not yet a novelty claim. The remaining search risk includes terminology from dynamic data
structures, editable sequence models, influence-removal methods, and deletion-aware recurrent
inference that may not use the words `cavity` or `endpoint`.

## References

- Katharopoulos et al., *Transformers are RNNs*, ICML 2020.
- Choromanski et al., *Rethinking Attention with Performers*, ICLR 2021.
- Smith, Warrington, and Linderman, *Simplified State Space Layers for Sequence Modeling*, ICLR 2023.
- Katsch, *GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling*, 2023.
- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023/2024.
- Sei and Yano, *Minimum Information Dependence Modeling*, Bernoulli 2024.
