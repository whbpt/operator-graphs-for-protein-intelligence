# Operator Graphs for Protein Intelligence

This directory is the book entry point for the operator-graph framework.

The mathematical body remains in `../paper/protein_operator_graph_framework.tex` so the article
and book do not drift apart. The source uses small wrapper commands: top-level sections become
sections in article mode and chapters in book mode. Book-only front matter lives here.

Build from this directory:

```sh
sh ../tools/tectonic/build.sh main.tex
```

The current four-part structure is:

1. Language and Foundations
2. Operator Geometry of Sequence Interactions
3. Models, Estimation, and Spectra
4. Realizability, Evidence, and Interpretation

The editorial model is a scientific monograph: sustained argument, exact derivations, model
portraits, historical context, and open research questions.
