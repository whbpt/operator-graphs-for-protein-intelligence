# Initial Findings: 3CNBA Transformer Disentanglement

Date: 2026-07-11

## Scope

This pilot uses the 3CNBA alignment and structural distance map. It compares:

- a real homolog ensemble;
- a PSSM-sampled independent-site null;
- an exact column-permutation null;
- a global-composition null;
- an ensemble containing repeated copies of the query sequence.

Experiments were run with ESM-2 8M and ESM-2 35M. Attention and supervised
contact-head maps were averaged across sequences in each ensemble.

## Classical control

At MSA depth 64, the leading mode of the mutual-information matrix explains
about 98% of its squared spectrum. Its leading vector has Spearman correlation
of about 0.99 with site entropy in both real and independent-site null MSAs.

For the real MSA, APC increases contact P@L from approximately 0.14 to 0.27.
For the null MSAs, APC does not produce a comparable improvement. Real and null
MI matrices remain strongly correlated before APC, but their APC residuals have
correlations near zero.

This reproduces the distinction between a shared marginal background and a
real-MSA-specific interaction residual.

## Transformer background

For ESM-2, real and PSSM/column-null contact maps are almost identical:

- contact-map long-range Spearman correlation: approximately 0.998;
- median per-head APC attention correlation: approximately 0.993-0.998;
- all or nearly all heads remain above 0.9 correlation.

Removing site-specific PSSM information changes the result substantially. With
the global-composition null, the 8M model's long-range contact P@L falls from
about 0.84 to 0.30. Repeating only the query sequence still gives about 0.79.
The 35M model shows the same qualitative behavior.

The dominant Transformer background therefore appears to be driven mainly by
single-sequence motifs and site-specific marginal information, rather than by
coevolution measured from the current family ensemble.

## Evidence for a separable interaction residual

Despite the near-identity of real and null maps, their small difference is
strongly enriched for structural contacts.

Using a mean of two null replicates at depth 64:

- ESM-2 8M `real - column-null` contact-head residual:
  - signed long-range P@L: 0.60;
  - absolute long-range P@L: 0.68.
- ESM-2 35M, using one column-null replicate:
  - signed long-range P@L: 0.65;
  - absolute long-range P@L: 0.69.
- long-range contact prevalence is approximately 0.09.

Individual attention-head residuals also contain contact signal. Exploratory
best-head absolute residual precision reaches roughly 0.63-0.77, depending on
model and null. Head selection currently uses the same structure labels and
must be validated out of sample.

## Current interpretation

There is evidence for disentanglement, but it is not the GREMLIN-LH geometry.

The working decomposition is:

`Transformer pairwise statistic = large motif/PSSM background + small signed interaction residual`

The useful residual is not generally the largest eigenmode. It is identified by
an intervention: preserve one-body marginals while destroying cross-position
dependencies, then subtract the expected null response.

## Limitations

- This is one protein family.
- ESM-2 is a single-sequence model; MSA Transformer remains to be tested.
- Gaps in aligned homologs are passed to a model trained primarily on unaligned
  sequences.
- The supervised contact head is not an unsupervised measure of coevolution.
- Best attention heads were selected using 3CNBA structure labels.
- A phylogeny-aware null has not yet been implemented.

## Next experiments

1. Repeat the null-residual analysis across a stratified subset of the
   383-family benchmark.
2. Select attention heads on training families and evaluate them on held-out
   families.
3. Replace the supervised contact head with categorical Jacobian and
   double-mutant finite-difference tensors.
4. Build a phylogeny-aware independent-site null.
5. Compare null subtraction with covariate projection and learned nuisance
   subspaces.

