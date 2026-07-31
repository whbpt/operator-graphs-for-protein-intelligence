# Independent Theory Review, Round 4

Date: 2026-07-12

Scope: joint cavity-measurable support, subcube/certificate routing, fixed sparse support coverage,
and sparse-Transformer prior art. Experiments remained paused.

## Verdict before final edits

No P0 issue was found. The joint support invariance condition, leave-two-out top-k counterexample,
decision-tree sufficient construction, and direct-edge coverage count are correct.

## P1 corrections

1. The subcube equivalence requires a deterministic router on the full q-ary product domain with
   `q>=2`. Under those conditions, constancy on every self-induced cylinder implies that the
   distinct cylinders form a partition.
2. The result is per target and directed. An undirected protein graph additionally requires
   reciprocal labels `j in R_i iff i in R_j`; independent per-node certificates do not imply this.
3. Existing cited subcube-partition complexity is primarily Boolean. The proposed q-ary router is
   a label-constrained specialization using that combinatorial language.

## Confirmed statements

- Pairwise endpoint-blind scores followed by top-k need not produce a jointly invariant incident
  set; the two-candidate counterexample has no tie.
- A standard deterministic decision tree is sufficient when the next query and leaf label depend
  only on observed path answers and the leaf selects only unqueried coordinates.
- Pathwise strict randomized routing requires the invariance property for every input on a
  probability-one set of fixed random seeds.
- A fixed candidate universe must be independent of sequence and router seed; selection can use
  only its complement and independent randomness.
- If `H` fixed undirected graphs each have maximum degree `k`, their union has at most `HLk/2`
  direct edges. Covering all unordered pairs requires the necessary condition `Hk>=L-1`; it is not
  sufficient because layers may repeat edges.
- BigBird, Exphormer, and Diffuser are valid precedents for global connectivity or multi-hop
  receptive fields under predefined sparse structures. They do not provide a cavity-measurable
  content router or direct protein-contact guarantees.

## Remaining architecture problem

The project now has three honest routing levels:

1. fixed sparse support with strict semantics but incomplete direct-contact coverage;
2. certificate/subcube routing with strict dynamic semantics but unresolved training and
   undirected reciprocity;
3. ordinary dynamic top-k with practical execution but only local table-gauge semantics.

## Post-edit verdict

The deterministic/full-domain conditions, directed-versus-undirected distinction, Boolean
prior-art qualification, and necessary-not-sufficient coverage wording were added. Within this
scope no unresolved P0 or P1 inconsistency remains.
