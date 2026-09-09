# Protein Interaction as an Operator

**From Potts Models to Transformers**

An evolving scientific monograph by Haobo Wang on the mathematical objects that
sit behind protein sequence models, graphical models, neural architectures, and
structure-generating systems.

[Read the latest PDF](output/pdf/operator_graphs_for_protein_intelligence.pdf)

## Scope

The book develops a common operator language for questions that are often
described with the same words but live in different mathematical spaces:

- Potts couplings, inverse covariance, and reference-centered categorical operators;
- static and context-dependent interaction graphs;
- softmax, Gibbs laws, Hopfield retrieval, RBMs, and Transformer routing;
- AlphaFold 1, 2, and 3, including diffusion, SDEs, and probability-flow ODEs;
- graph convolution, Laplacians, wavelets, and multiscale spectral structure;
- architectural realizability, falsification, and typed biological validation.

The aim is a scientific monograph rather than a textbook. The manuscript preserves
derivations and interpretation boundaries while organizing them around one sustained
argument: a graph becomes meaningful only after its state space, edge object,
normalization, dynamics, estimator, and validation target have been declared.

## Book structure

1. **Language and Foundations (Chapters 1--2):** One Edge, Several Meanings; When Equations Change
   Their Meaning.
2. **Operator Geometry of Sequence Interactions (Chapters 3--6):** Reference-Centered
   Operator-Valued Graphs; Static and Dynamic Sequence Interactions; Interaction Order, Rank, and
   Scalar Compression; Probability-Exact Edge Geometry.
3. **Models, Estimation, and Spectra (Chapters 7--11):** Model Correspondences in Operator
   Coordinates; Phylogeny as Tree-Constrained Transport; From Observations to Operator Graphs;
   Three AlphaFolds: The Evolution of Relational State; Spectra After the Graph Has Been Declared.
4. **Realizability, Evidence, and Interpretation (Chapters 12--15):** Architectural Consequences
   and Realizability; Conditional Realizability and Information Limits; Falsification and
   Biological Interpretation; What Kind of Object Is a Protein Interaction?

The optional appendix is:

- **Appendix A:** Population Selection as an Optional Bridge

The default sequence is cumulative: Chapters 1--2 stabilize meanings; Chapters 3--6 construct the
mathematical object; Chapters 7--11 connect it to evolution, models, estimation, and spectra; and
Chapters 12--15 address realization, evidence, and information limits. The appendix provides
the additional population model used at the assay-to-evolution boundary.

## Reading Routes

- **Protein language models and operators:** Chapters 1, 4, 7, 9, 10, and 14.
- **Statistical physics:** Chapters 2, 3, 5, 6, 8, and 11.
- **Architecture and realizability:** Chapters 3, 4, 10, 12, 13, and 14.

## Status

Draft monograph, July 2026. The PDF currently contains 171 pages. Substantive
revisions will be tagged as versioned releases; small corrections may appear on the
default branch between releases.

## Repository layout

- `book/`: book entry point, front matter, notation, and chapter-specific inserts;
- `paper/protein_operator_graph_framework.tex`: shared mathematical body;
- `output/pdf/`: canonical compiled PDF;
- `tools/tectonic/`: reproducible build wrapper.

## Build

Install [Tectonic](https://tectonic-typesetting.github.io/) and run:

```sh
cd book
sh ../tools/tectonic/build.sh main.tex
```

The resulting file is `book/main.pdf`. The checked-in canonical edition is
`output/pdf/operator_graphs_for_protein_intelligence.pdf`.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Until a DOI is
assigned, cite the tagged GitHub release and include the version or access date.

## Rights

The manuscript source and compiled PDF are copyright (c) 2026 Haobo Wang, all rights reserved;
see [`MANUSCRIPT-LICENSE.md`](MANUSCRIPT-LICENSE.md). The software and build utilities in `tools/`
are available under the MIT License in [`LICENSE`](LICENSE).
