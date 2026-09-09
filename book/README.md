# Protein Interaction as an Operator

This directory is the book entry point for the operator-graph framework.

The mathematical body remains in `../paper/protein_operator_graph_framework.tex` so the article
and book do not drift apart. The source uses small wrapper commands: top-level sections become
sections in article mode and chapters in book mode. Book-only front matter lives here.

Chapter insert filenames retain historical drafting prefixes.  Those prefixes group related source
files but do not encode the current printed chapter number.  The authoritative order is the
`\BookChapter`, `\BookOnlyChapter`, and `\BookInsert` sequence in
`../paper/protein_operator_graph_framework.tex`, together with the chapter list below.  For
example, `08_msa_pairformer_case.tex` is currently Section 10.8, `09_opening.tex` opens Chapter 11,
and `10_opening.tex` opens Chapter 12.

Build from this directory:

```sh
sh ../tools/tectonic/build.sh main.tex
```

The current four-part structure is:

1. **Language and Foundations**
   - Chapter 1: One Edge, Several Meanings
   - Chapter 2: When Equations Change Their Meaning
2. **Operator Geometry of Sequence Interactions**
   - Chapter 3: Reference-Centered Operator-Valued Graphs
   - Chapter 4: Static and Dynamic Sequence Interactions
   - Chapter 5: Interaction Order, Rank, and Scalar Compression
   - Chapter 6: Probability-Exact Edge Geometry
3. **Models, Estimation, and Spectra**
   - Chapter 7: Model Correspondences in Operator Coordinates
   - Chapter 8: Phylogeny as Tree-Constrained Transport
   - Chapter 9: From Observations to Operator Graphs
   - Chapter 10: Three AlphaFolds: The Evolution of Relational State
   - Chapter 11: Spectra After the Graph Has Been Declared
4. **Realizability, Evidence, and Interpretation**
   - Chapter 12: Architectural Consequences and Realizability
   - Chapter 13: Conditional Realizability and Information Limits
   - Chapter 14: Falsification and Biological Interpretation
   - Chapter 15: What Kind of Object Is a Protein Interaction?

The optional appendix is a second-reading route:

- Appendix A: Population Selection as an Optional Bridge

The default sequence is cumulative: Chapters 1--2 stabilize meanings; Chapters 3--6 construct the
mathematical object; Chapters 7--11 connect it to evolution, models, estimation, and spectra; and
Chapters 12--15 address realization, evidence, and information limits. The appendix provides
the additional population model needed at the assay-to-evolution boundary.

Short, hand-checkable primers now precede the main abstraction jumps: sequence and MSA notation,
categorical linear algebra, three-variable ANOVA, binary probability geometry, tree-induced
correlation, spectral method selection, Bayes error, and experimental epistasis.

Several long chapters use an explicit internal grammar.  Chapter 7 moves from probabilistic model
portraits and neural transport through the 2019 reconstruction framework to the GNN lineage.
Chapter 10 develops AlphaFold's relational state, diffusion transport, and MSA Pairformer in one
protein-facing sequence.
Chapter 11 separates core spectral constructions, case studies, localized filters, and context
averaging.  Chapter 12 separates exact capacity results, constructive designs, and an empirical
protein-generation case.

## Reading Routes

- **Protein language models and operators:** Chapters 1, 4, 7, 9, 10, and 14.
- **Statistical physics:** Chapters 2, 3, 5, 6, 8, and 11.
- **Architecture and realizability:** Chapters 3, 4, 10, 12, 13, and 14.

The editorial model is a scientific monograph: sustained argument, exact derivations, model
portraits, historical context, and open research questions.
