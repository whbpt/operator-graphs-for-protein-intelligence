# MSA-Conditioned Protein Dynamics for Design

Status: research hypothesis; not ready for inclusion in the book.

## Core idea

For each homologous sequence $s_a$ in an MSA, estimate a
sequence-conditioned conformational distribution

\[
s_a \longrightarrow P(r\mid s_a),
\]

and, where identifiable, a transition model

\[
K_a(r\rightarrow r').
\]

Transport these sequence-specific ensembles into a common conformational
coordinate system, then separate:

1. dynamics conserved across the protein family;
2. subfamily-specific changes in conformational populations;
3. sequence-background-specific modulation;
4. phylogenetic and sampling effects.

The eventual design objective is to learn how sequence changes reshape a
conformational energy landscape, rather than only whether a sequence is
compatible with one target structure.

## Potential design value

This framework may support:

- stabilization of desired active or inactive states;
- allosteric and fold-switching protein design;
- preservation of conserved functional motions;
- negative design against off-pathway, aggregation-prone, or inactive states;
- mutation design that transfers across multiple sequence backgrounds.

## Critical distinction

A collection of structures predicted for different homologs,

\[
\{\widehat r(s_a):s_a\in\mathrm{MSA}\},
\]

is an evolutionary comparative ensemble. It is not the time-dependent
conformational ensemble of any one sequence.

The method becomes a dynamics model only if each sequence is assigned its own
conformational distribution or transition law, and if phylogenetic variation
is separated from within-sequence physical fluctuations.

## Closest prior work

Relevant but incomplete precedents include:

- family-wide ensemble normal-mode analysis in Bio3D and SignDy;
- coevolution-constrained conformational sampling by Sutto et al.;
- MSA clustering or subsampling methods such as AF-Cluster,
  shallow-MSA AlphaFold2 sampling, and SPEACH_AF.

None of these, by itself, closes the full pipeline from every MSA sequence to
a calibrated sequence-specific energy landscape and experimentally validated
design rule.

## Why this is not currently part of the book

The proposal joins several levels that the book deliberately keeps separate:
evolutionary sequence variation, predicted conformational heterogeneity,
thermodynamic populations, kinetics, and protein-design objectives.

Until the estimand, common conformational coordinates, phylogenetic controls,
and experimental validation are established, it should remain a research
note rather than a methodological case study.
