# Independent Theory Review, Round 3

Date: 2026-07-12

Scope: endpoint-removable summaries, affine segment products, capacity bounds, strict cavity
measurability, and the updated prior-art boundary. Experiments remained paused.

## Verdict before final edits

No P0 issue was found. The augmented affine-map deletion identity, segment-tree complexity, and
fixed-support cavity sigma-algebra contract are mathematically sound under token-local transition
maps. Three P1 wording conditions required correction.

## P1 corrections

1. **Algebraic versus numerical deletion.** The additive identity

   \[
   T-\sum_{a\in A}\psi_a=\sum_{k\notin A}\psi_k
   \]

   is exact over exact arithmetic. Subtracting endpoint terms from a rounded floating-point global
   sum can leave endpoint-dependent residuals. Strict numerical invariance requires retained-range
   queries that never include the deleted terms, or an exact accumulator.

2. **Capacity-bound quantifiers.** The finite-precision counting bound assumes a fixed known
   deletion set, deterministic zero-error encoding, the full retained Cartesian domain, and at
   most `B` bits per coordinate. The linear-readout bound assumes one fixed summary map and exact
   scalar linear readout for every centered one-site function on the full product domain.

3. **Sei--Yano uniqueness.** Their theorem identifies the aggregate one-body adjustment
   `sum_i a_i(x_i)` and the potential after gauge fixing. Individual adjusting functions retain
   additive-constant gauge freedom.

## Confirmed results

- Token-local affine maps are represented exactly by augmented matrices.
- Replacing deleted maps by identities removes category dependence while preserving retained map
  order on the original position index set.
- A balanced segment tree costs `O(L C_comp)` to build and
  `O((m+1) log L C_comp)` for deletion of `m` positions.
- A fixed degree-`k` candidate graph is subquadratic when state dimension and composition cost do
  not grow with sequence length.
- Fixed support, token-local maps, endpoint-deleted node/edge contexts, and endpoint-blind gates
  satisfy the stated strict cavity conditioning contract.
- The paper treats Deep Sets, linear attention, S5, GateLoop, Mamba, segment trees, and
  minimum-information dependence modeling as prior art rather than individual novelty claims.

## Remaining research risk

Exact deletion is not the same as sufficient expressiveness. A compact recurrence product may
discard information needed to infer a categorical pair field. In addition, a fixed sparse graph
is easy to make identifiable but may not cover arbitrary long-range protein contacts; a dynamic
support must satisfy joint incident-set measurability, not merely pairwise endpoint blindness.

## Post-edit verdict

The three P1 conditions above were incorporated into the theory draft and detailed mathematical
note. Within the current claim scope, no unresolved algebraic inconsistency remains. The active
problem is approximation theory and support construction, not another rank choice.
