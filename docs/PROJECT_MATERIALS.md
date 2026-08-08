# Project Materials Map

Date: 2026-07-12

This project directory is the authoritative home for the Transformer disentanglement research.

## Authoritative research sources

- `paper/protein_operator_graph_framework.tex`: authoritative operator-valued graph framework.
- `paper/ternary_msa_reconstruction.tex`: PRL/PRX-direction paper on family-level ternary MSA
  reconstruction, ESM-style dynamic ternary routing, and their teacher--student connection.
- `paper/theory_main.tex`: compatibility entry point for the framework manuscript.
- `paper/theory_main.pdf`: previous compiled framework PDF; regenerate from the renamed source
  before treating it as current.
- `paper/main.tex`: longer empirical research log; historical experimental conclusions remain
  provisional.
- `docs/`: mathematical notes, architecture decisions, prior-art reviews, and reviewer rounds.
- `docs/reviewer_round_1.md` through `docs/reviewer_round_9.md`: independent review record.
- `docs/typed_relational_workspace_llm.md`: cross-domain architecture proposal for transferring
  typed, multiscale, relational inductive biases from protein models to efficient language models.

## Raw data

- `data/raw/3CNBA/`: alignment, structure, distances, and reference mapping.
- `data/raw/GREMLIN_LH/`: retrieved GREMLIN_LH notebook and associated DMS data.
- `data/raw/seqmodels_benchmark/`: standalone compressed benchmark artifact.
- `data/raw/torch/`: local model cache.

Raw data is immutable and excluded from Git.

## External repositories and caches

- `work/external/seqmodels/`: external seqmodels checkout and large benchmark cache.
- `work/external/whbpt-examples/`: source repository snapshot for the 3CNBA example.

These directories are references and are not part of the project implementation.

## Literature

- `work/literature/haobo_wang/`: the original disentanglement/co-evolution paper and related local
  material.
- `work/literature/deepseek_papers/`: DeepSeek architecture papers used in the architecture review.
- `work/literature/deepseek_notes/`: downloaded notes and extracted material from that review.
- `work/literature/certificate_router/`: decision-tree and active-feature-acquisition papers.
- `work/literature/core/`: GateLoop, minimum-information dependence modeling, and other core papers.
- `work/literature/prior_art/`: additional prior-art captures used by the theory draft.

## Generated material

- `results/`: experiment outputs and historical run metadata.
- `tmp/`: PDF renders and other disposable verification artifacts.
- `artifacts/published/`: project-local mirror of the user-facing deliverables in the workspace
  `outputs/` directory.

Edit the source in `paper/` or `docs/`, regenerate the artifact, and then refresh the published
mirror. Do not edit a mirrored artifact as if it were an authoritative source.

## Project-local build dependency

`sh tools/tectonic/build.sh` is the stable build entry point. It wraps the pinned Tectonic 0.16.9
engine and forces compilation through a validated repository-local TeX bundle and format cache.
The build is therefore offline and does not consult macOS system proxy settings or write to the
user cache. Invoke the script explicitly with `sh`; this avoids macOS provenance checks on newly
created executable scripts. The complete `tools/tectonic/` runtime is a local build dependency
rather than research material and is excluded from Git.
