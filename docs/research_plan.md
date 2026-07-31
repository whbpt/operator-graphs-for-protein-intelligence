# Research Plan

## Competing hypotheses

### H0: no separable nuisance component

The apparent low-rank signal is intrinsic to Transformer normalization or
routing and cannot be removed without losing useful interaction information.

### H1: marginal-statistics contamination

A component is predictable from site entropy, conservation, gap rate, or model
surprisal and persists when pairwise correlations are destroyed.

### H2: phylogenetic contamination

A component persists in tree-aware independent-site simulations but is reduced
when phylogenetic structure is removed.

### H3: identifiable coevolutionary interaction

A residual component is present in real homologs, disappears in matched nulls,
and predicts structural contacts or experimental epistasis.

## Observable objects

Analyze these separately:

1. Post-softmax attention matrices.
2. Pre-softmax query-key score matrices.
3. Mean-centered and symmetrized categorical Jacobians.
4. Double-mutant finite-difference interaction tensors.

## Evidence required for disentanglement

A proposed correction should:

- remove signal that survives in a PSSM-matched independent-site null;
- reduce association with prespecified nuisance covariates;
- preserve or improve long-range contact precision;
- preserve marginal amino-acid statistics and masked-language-model quality;
- replicate across proteins, layers, heads, and MSA depths.

## Milestone 1: 3CNBA diagnostic

- Depths: 128, 512, 2,000, and 10,000 sequences.
- Conditions: real MSA and at least five PSSM-null replicates per depth.
- Outputs: spectra, singular vectors, nuisance correlations, APC comparisons,
  contact precision, and uncertainty across replicates.

