# Transformer Disentanglement R&D

Research workspace for testing whether protein Transformers contain separable
one-body, entropy, phylogeny, and coevolutionary components.

## Core question

Does an APC-removable or otherwise low-dimensional background in Transformer
pairwise statistics represent nuisance signals, and can it be removed without
destroying structural or mutational interactions?

## Initial experiment

1. Use the 3CNBA MSA and packaged structural contacts.
2. Construct depth-matched independent-site MSAs with the same PSSM.
3. Extract attention, pre-softmax scores, categorical Jacobians, and
   double-mutant interaction scores.
4. Compare spectra, entropy correlations, APC effects, and contact precision.
5. Scale validated diagnostics to the 383-family seqmodels benchmark.

## Layout

- `configs/`: experiment configurations
- `artifacts/published/`: project-local mirror of user-facing papers, reports, figures, and tables
- `data/raw/`: immutable source datasets, excluded from Git
- `data/processed/`: generated nulls and normalized datasets, excluded from Git
- `docs/`: hypotheses, decisions, and data documentation
- `experiments/`: reproducible experiment entry points
- `notebooks/`: exploratory analysis only
- `results/`: generated metrics and figures, excluded from Git
- `scripts/`: data acquisition and conversion utilities
- `src/`: reusable implementation
- `tests/`: focused tests for data and mathematical transformations
- `tools/`: project-local build tools
- `work/external/`: external repositories and large local caches
- `work/literature/`: source papers, extracted text, and reading material

See `docs/PROJECT_MATERIALS.md` for the complete material map. The paper sources are split between
the framework in `paper/protein_operator_graph_framework.tex` and the ternary-MSA manuscript in
`paper/ternary_msa_reconstruction.tex`; `artifacts/published/` is a delivery mirror rather than an
editable source.

## Working rules

- Keep raw data immutable.
- Record every null-generation seed and sampling depth.
- Separate model extraction from downstream correction.
- Treat attention, logits, Jacobians, and epistasis scores as distinct objects.
- Do not call a component entropy or coevolution without an intervention-based
  test against a matched null distribution.

The original root README was moved here when the public book repository was prepared.
