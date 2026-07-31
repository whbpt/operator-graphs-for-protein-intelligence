# Data

## Primary sources

### 3CNBA

- Source: https://github.com/whbpt/examples/blob/master/3CNBA.zip
- Aligned MSA: 25,947 sequences x 120 positions
- Raw A3M: 65,535 sequences
- Includes PDB structure, distance data, reference mapping, and contact matrix
- Local path: `data/raw/3CNBA/`

### Seqmodels benchmark

- Source: https://github.com/sokrypton/seqmodels
- Artifact: `data.pickle.npy.gz`
- Contains 383 protein families with encoded MSAs, sequence weights, structural
  contacts, masks, and PDB-chain identifiers.
- The object array requires `allow_pickle=True`; convert it once in an isolated
  step to a non-pickle format before routine use.
- Local standalone artifact: `data/raw/seqmodels_benchmark/data.pickle.npy.gz`
- External source checkout and its larger cache: `work/external/seqmodels/`

### GREMLIN_LH

- Local path: `data/raw/GREMLIN_LH/`
- Contains the retrieved Colab notebook and associated DMS table.
- The original download archive is retained as `data/raw/gremlin_lh.zip`.

### Source repository snapshot

- `work/external/whbpt-examples/` contains the external examples repository used to obtain 3CNBA.
- External repositories are reference snapshots, not project source code.

Raw data belongs in `data/raw/` and must not be edited in place. Generated
PSSM-null and phylogeny-null datasets belong in `data/processed/`.
