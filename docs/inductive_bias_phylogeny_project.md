# Inductive-Bias Phylogenetic Reconstruction

## Research claim

Phylogenetic reconstruction is not generic dense-graph prediction. It is amortized
inference over a tree-structured latent-variable model. A learned constructor should
therefore preserve the tree's output type, separator geometry, Markov composition, and
leaf-permutation symmetry by construction.

The first implementation deliberately separates two roles:

1. a trainable symmetric residual proposes topology moves;
2. exact Felsenstein pruning evaluates the resulting tree under a declared substitution
   model.

The neural component learns search. It does not silently redefine the likelihood.

## Implemented prototype

The module *src/transformer_disentanglement/inductive_phylogeny.py* provides:

- immutable leaf, internal-node, and branch-length types;
- canonical topology, split, Robinson-Foulds, and Newick utilities;
- general finite-state CTMC transition kernels;
- a scaled Felsenstein likelihood;
- simulation on a tree;
- Jukes-Cantor distance estimation;
- a symmetric trainable residual over neighbor-joining candidate features;
- supervised merge targets derived from the true tree's leaf cherries;
- beam search over valid partial trees;
- exact-likelihood reranking after topology-specific optimization of all branch lengths.

At each search step, the candidate score is

\[
S_\theta(C,D)
=S_{\mathrm{NJ}}(C,D)
+r_\theta(\psi(C,D)),
\qquad
S_\theta(C,D)=S_\theta(D,C).
\]

The residual output layer starts at zero. The untrained model is therefore a reproducible
neighbor-joining prior rather than a random neural constructor.

Every merge produces an internal tree node. Consequently all intermediate and final states
are valid trees; no tree penalty or post-hoc adjacency repair is required.

## Current validation gates

The unit tests require:

1. stochastic transition matrices and the CTMC semigroup law;
2. detailed balance for the Jukes-Cantor reference model;
3. equality of pruning and explicit ancestral-state enumeration;
4. root-placement invariance under a reversible process;
5. the four-point condition for patristic distances;
6. exact split recovery from an additive distance matrix;
7. alignment-row permutation equivariance;
8. supervised cherry targets that decrease a trainable merge loss;
9. end-to-end quartet recovery from simulated sequences.

Run them with:

    .venv/bin/python -m pytest -q tests/test_inductive_phylogeny.py

The simulation benchmark compares greedy construction with beam search:

    PYTHONPATH=src .venv/bin/python \
      experiments/evaluate_inductive_phylogeny.py \
      --output results/inductive_phylogeny_quartet_v2

The benchmark records topology recovery, Robinson-Foulds distance, exact likelihood, and
the number of unique candidate trees.

## Frozen quartet baseline

The first difficult baseline uses 100 random quartet trees per condition, short internal
branches, heterogeneous pendant branches, and seed 41. The zero-residual constructor gives:

| Sites | Greedy recovery | Beam plus likelihood recovery |
| ---: | ---: | ---: |
| 25 | 0.54 | 0.55 |
| 50 | 0.68 | 0.70 |
| 100 | 0.79 | 0.82 |
| 250 | 0.92 | 0.92 |

Beam search evaluates all three quartet topologies after optimizing their branch lengths.
The gain over greedy neighbor joining is therefore small but measurable at low and
intermediate information. These numbers are a structural baseline, not evidence for a
learned improvement.

Machine-readable results are in
*results/inductive_phylogeny_quartet_v2/summary.json* and
*results/inductive_phylogeny_quartet_v2/per_replicate.csv*.

## What has not yet been established

The current prototype validates the structural inductive biases but does not yet show that
the learned residual improves on neighbor joining. That claim requires a training protocol
and held-out evolutionary regimes.

The next experiment should train on simulated families while holding out combinations of:

- branch-length distributions;
- substitution generators;
- site-rate heterogeneity;
- compositional shifts;
- indels and missing observations;
- epistatic sequence evolution;
- recombination or local-tree violations.

Evaluation must compare IQ-TREE, neighbor joining, Phyloformer-like distance prediction,
the zero-residual constructor, and the trained constructor. Ablations should remove one
inductive bias at a time rather than compare only model sizes.

## Phase-two learning objective

A supervised topology loss can teach valid merge actions or target splits. A probabilistic
version should optimize

\[
\mathcal L(\theta)
=-\mathbb E_{q_\theta(T,\ell\mid X)}
  [\log p(X\mid T,\ell,Q)]
+\beta\,\mathrm{KL}
  (q_\theta(T,\ell\mid X)\|p(T,\ell)).
\]

For multimodal posteriors, beam search can be replaced by sequential Monte Carlo or a
GFlowNet while retaining the same tree state and exact pruning evaluator.

Taxon-subsampling consistency is a separate generalization test:

\[
\operatorname{Restrict}_S\widehat T(X)
\approx
\widehat T(X_S).
\]

It should be measured, not assumed, because finite alignments and model misspecification can
make exact projectivity impossible.
