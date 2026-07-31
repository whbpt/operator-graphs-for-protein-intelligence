# Global Support No-Go and the Directed Cavity Design

Date: 2026-07-12

Status: mathematical architecture decision; no experiments.

## 1. Three support semantics

The word `support` has been used for three different objects.

1. **Local table support.** A full-context router chooses edges; only each emitted categorical
   table is gauge-fixed.
2. **Targetwise conditional support.** For each target `i`, its complete incident set `N_i` is
   invariant when `x_i` and `x_{N_i}` vary. This supports a directed conditional prediction
   theorem for target `i`.
3. **Joint graph-valued support.** One undirected graph `E(x)` is itself invariant when all active
   endpoint categories vary simultaneously. This treats the graph as one globally identified
   latent object.

Level 3 is much stronger than level 2. Reciprocal targetwise routers do not automatically imply
joint graph-valued invariance because changing one residue may alter edges among other vertices.

## 2. Global endpoint-invariance definition

Let

\[
E:[q]^L\to\mathcal G_L
\]

be a deterministic router into simple undirected graphs on the full product domain. For a graph
`G`, define its
active vertex set

\[
V_+(G)=\{i:\deg_G(i)>0\}.
\]

The global strict condition is

\[
E(x)=G
\Longrightarrow
E(x')=G
\]

whenever

\[
x'_{[L]\setminus V_+(G)}
=x_{[L]\setminus V_+(G)}.
\]

Every category at a vertex participating in an edge can change without changing the complete
graph.

## 3. Spanning-support no-go theorem

### Theorem

If there exists an input `x` for which `E(x)` has no isolated vertices, then a globally
endpoint-invariant router is constant on the entire domain. In particular, if every output graph
has minimum degree at least one, no nonconstant content-dependent router exists.

### Proof

Let `G=E(x)` and assume `V_+(G)=[L]`. The complement `[L] minus V_+(G)` is empty. Therefore every
`x' in [q]^L` agrees with `x` on that complement. Global endpoint invariance gives `E(x')=G` for
all `x'`. Hence `E` is constant.

### Corollary

Within the preimage cell of a realized graph `G`, its label is insensitive to all active-vertex
categories and can vary only through coordinates isolated in `G`. Those vertices may act as
evidence in that cell but cannot simultaneously be active endpoints in the same graph.

For a random seed independent of the sequence, the same result holds pathwise for almost every
fixed seed.

## 4. Why this does not eliminate targetwise routing

Targetwise invariance requires only `N_i` to remain fixed when its own active endpoints vary. A
residue may still alter edges among targets to which it is not currently adjacent. Thus nontrivial
directed certificate routers remain possible.

Even adding reciprocity

\[
j\in N_i(x)\iff i\in N_j(x)
\]

does not by itself imply global graph invariance. It only makes the realized edge relation
undirected. A change at vertex `v` may preserve all edges incident to `v` while altering edges
among non-neighbors.

Therefore the no-go theorem must not be overstated:

- it rules out nonconstant **jointly graph-invariant spanning support**;
- it does not rule out targetwise conditional routers;
- targetwise semantics are not a globally identified graph decomposition.

## 5. Relation to impartial selection

Impartial-selection mechanisms require an agent's own report not to affect its own probability of
selection. This is conceptually related to endpoint-blind routing: a coordinate cannot influence
the edge or role in which it is an endpoint.

The two conditions are not formally ordered. Under a coordinate-as-agent analogy, our pathwise
incident-set invariance imposes additional endpoint exclusions, while impartial-selection
mechanisms address a different strategic selection problem. Their approximation results are
useful design analogies, not direct solutions.

Relevant papers include Fischer and Klimm's *Optimal Impartial Selection* and Bousquet, Norin, and
Vetta's *A Near-Optimal Mechanism for Impartial Selection*.

## 6. Architecture decision

We choose a directed strict content-dependent candidate, matching the native asymmetry of
attention while keeping targetwise conditional semantics distinct from global graph semantics.

For each target `i`:

1. an endpoint-removable certificate router chooses a directed context set `N_i`;
2. one target cavity deletes `{i} union N_i` and predicts `p_i^E` plus target-relative source
   references `p_{j|i}^E`;
3. the relation decoder emits every directed field `G_{i<-j}` from this same target-cavity affine
   recurrence product;
4. the observed neighbor category is used only in the final lookup

   \[
   \ell_i(a)=b_i^E(a)+\sum_{j\in N_i}G_{i\leftarrow j}(a,x_j).
   \]

This is a targetwise conditional operator, not a globally normalized undirected graphical model.
Using only pair-deleted contexts gives a weaker per-arc gauge because another active source can
modulate a field. Using target `j`'s own cavity reference on the right is also invalid in general:
without reciprocity it may still observe `x_i`.

For proteins, the raw MSA log-density-ratio targets in the two directions are transpose-related.
After projection under different target-relative cavity references, the two teachers need not be
exact transposes. A symmetric diagnostic may be formed post hoc, for
example

\[
s_{ij}=\frac12\left(s_{i\leftarrow j}+s_{j\leftarrow i}\right),
\]

without claiming that the predictive support is one globally invariant contact graph.

## 7. Consequence for the Transformer question

Transformer attention is asymmetric because every target chooses its own sources. This makes a
directed targetwise interpretation natural. Forcing a sequence-dependent, spanning, globally
invariant undirected support would collapse the router to a fixed graph under the theorem above,
but reciprocal targetwise routing can still remain nontrivial because it need not satisfy that
global invariance.

We choose to describe the proposed architecture as a **directed cavity interaction mixer**, not
an inferred undirected contact graph. This is a semantic design decision rather than a theorem
that reciprocity is impossible. Physical contact prediction can be an auxiliary
symmetrized readout, not the semantic definition of the sequence operator.

## 8. Remaining problems

1. Quantify the Bayes gap between endpoint-deleted and full-sequence teacher-support prediction,
   then optimize a budgeted teacher-mass envelope rather than claim exact graph recovery.
2. Determine whether directed pair fields can share parameters or transpose-related MSA targets
   without silently reintroducing a global-joint claim.
3. Establish approximation theory for the endpoint-deleted recurrence context.
4. Decide whether the exact transport-coupling decoder or the tangent-field decoder is the better
   directed value primitive.

## 9. Prior-art boundary

- Impartial selection supplies self-influence-free mechanism ideas, not categorical interaction
  decomposition.
- Directed attention and sparse attention are established.
- Fixed-marginal dependence modeling and pair gauges are established.
- The possible contribution remains the combination of targetwise certificate routing,
  endpoint-removable ordered context, common-gauge categorical values, and optional-MSA teaching.

This remains a research hypothesis rather than a novelty claim.
