# Cavity-Measurable Sparse Support

Date: 2026-07-12

Status: mathematical design note; no experiments.

## 1. The support problem

Fix a target position `i`. A router maps a sequence to an incident neighbor set

\[
R_i(x)=N_i(x)\subseteq [L]\setminus\{i\}.
\]

The strict cavity model uses a background reference that excludes `i` and all selected neighbors.
It is not enough for each individual edge score to omit its two endpoints. The entire set
`N_i(x)` must remain unchanged when any active endpoint category is varied.

## 2. Joint cavity invariance

Define the strict support condition:

\[
N_i(x)=N
\quad\Longrightarrow\quad
N_i(x')=N
\]

for every `x'` satisfying

\[
x'_{[L]\setminus(\{i\}\cup N)}
=x_{[L]\setminus(\{i\}\cup N)}.
\]

Thus `x_i` and every selected `x_j`, `j in N`, may be changed arbitrarily without changing the
incident support.

This is stronger than

\[
z_{ij}(x)\text{ does not read }x_i,x_j.
\]

Other decisions `z_ik` may still read `x_j`, alter the top-k competition, and change the complete
neighborhood.

## 3. Subcube-partition equivalence

For each input `x`, define the cylinder

\[
C_i(x)=\left\{x':
x'_{[L]\setminus(\{i\}\cup N_i(x))}
=x_{[L]\setminus(\{i\}\cup N_i(x))}
\right\}.
\]

### Proposition

Let `q>=2`, let the domain be the full product `[q]^L`, and let `R_i` be deterministic. The router
satisfies joint cavity invariance if and only if it is constant on every self-induced cylinder
`C_i(x)`. Consequently, the distinct cylinders form a partition. For fixed degree `k`, each cell
is a `(k+1)`-dimensional q-ary subcube whose free set is exactly `{i} union R_i(x)`.

### Proof

If joint invariance holds, every `x' in C_i(x)` has the same output `N_i(x)`. Therefore it has the
same fixed complement and `C_i(x')=C_i(x)`. Two cells that intersect are identical, so the
distinct cells form a partition.

Conversely, if the router is constant on each such cylinder, changing only its free coordinates
cannot change the output.

The construction is a label-constrained version of subcube partition models in query complexity:
the label determines which coordinates must remain free. Existing cited complexity results are
primarily Boolean; the q-ary, label-constrained router is a specialization using their language,
not the same model already analyzed there.

## 4. Why leave-two-out top-k fails

Let the target have candidates `j,k`, and suppose

\[
s_{ij}=0,
\qquad
s_{ik}=2\mathbf1\{x_j=1\}-1.
\]

The score `s_ij` does not use `x_i,x_j`; `s_ik` does not use `x_i,x_k`. Hence both scores are
pairwise endpoint-blind. With top-1 routing, `x_j=0` selects `j`, but changing the selected value
to `x_j=1` selects `k`. The incident set is not cavity-invariant.

Therefore

> pairwise endpoint-blind scores plus top-k do not imply a cavity-measurable support.

## 5. Decision-tree construction

A sufficient dynamic construction is a standard deterministic decision tree whose internal nodes
query residue coordinates and whose leaves output only unqueried coordinates. The next query and
leaf label depend only on answers observed along the realized path, and the leaf performs no
additional sequence access.

Let `Q_leaf` be the coordinates queried on the realized root-to-leaf path. Require

\[
\{i\}\cup N_i(x)
\subseteq [L]\setminus Q_{\mathrm{leaf}}.
\]

Changing `x_i` or any selected neighbor cannot change an answer on the realized path, so the leaf
and its output remain fixed. This gives a strict dynamic router.

The construction exposes the semantic restriction: a position can be selected only on the basis
of evidence from other positions, never from its own realized residue. The decision-tree router is
only a sufficient subclass. Subcube partitions can be more query-efficient than decision trees,
as established in query-complexity theory.

### 5.1 Hard-forward training theorem

Let `H_t=((a_1,x_a1),...,(a_t,x_at))` be the observed query transcript. With all random choices
fixed by a seed `omega` independent of the sequence, choose the next query or stop action from
`f_theta(i,H_t,omega)`. At the leaf require

\[
N_i(x;\omega)\subseteq[L]\setminus(\{i\}\cup Q_{leaf}).
\]

For every fixed seed, changing only `x_i` and the selected source categories leaves the transcript
unchanged step by step. The hard sampled support is therefore pathwise strict during training.
REINFORCE and stochastic-computation-graph gradients are unbiased under their usual assumptions;
a straight-through backward pass is biased but does not alter the hard forward semantics.
The queried observations must be token-local or independently certified. Ordinary contextual
states may already contain active endpoints and invalidate the stated query transcript. A
nonlocal observation is structurally admissible only if its complete dependency set avoids the
target and every coordinate any descendant leaf can emit.

An ordinary soft leaf mixture is not automatically strict. Let `U(x)` be the union of every
positive-mass leaf and let `Q` be the true dependency closure of its gates and representations.
One auditable sufficient certificate requires `U(x)` itself to be cavity-invariant, derives all
weights, values, directional references, and gauge-dependent teacher quantities from the same
union cavity, and structurally enforces

\[
Q\cap(\{i\}\cup U)=\varnothing
\]

with the background deleting `{i} union U`. Dense softmax routing usually makes `U` large, so a
hard forward is the conservative training contract. Other exact cancellations would need a
direct functional invariance proof.

### 5.2 Necessary blindness properties

Strict invariance implies that the complete `N_i` cannot depend on `x_i`. It also implies that

\[
\mathbf1\{j\in N_i(x)\}
\]

cannot depend on `x_j`. If two inputs differ only at `j` and either output selects `j`, invariance
from that output forces the complete source sets to agree; if neither selects `j`, both membership
indicators are zero.

Therefore strict support cannot selectively open an arc using a nonconstant rule of the two
realized endpoint categories alone. The arc may be opened independently, after which endpoint
categories determine the signed message through `G_{i<-j}(:,x_j)`; they cannot provide evidence
for their own support membership.

### 5.3 Query-depth coverage bound

A deterministic depth-`B` q-ary tree, or one fixed-seed realization, whose queries each reveal one
raw q-ary category has at most `q^B` leaves. If every leaf emits at most `k`
sources, one target can directly select at most `k q^B` distinct source identities over all
inputs. Full positional coverage requires

\[
kq^B\ge L-1.
\]

The ability to realize every size-`k` source set requires the stronger necessary condition

\[
q^B\ge {L-1\choose k}.
\]

These bounds do not establish that useful support can be inferred from the queried evidence.
An independent seed choosing among `S` trees changes the union bound to `S k q^B`, but contributes
random coverage rather than endpoint evidence.
Realizing every size-`k` set across that finite bank requires `S q^B >= choose(L-1,k)`.
For outcome alphabets `m_t`, the leaf bound is `product_t m_t`. Continuous observations or seeds
need a finite-policy or quantization assumption for any cross-seed count.
The complete derivation and training contract are in
`docs/trainable_certificate_router_theory.md`.

## 6. Randomized routers

For pathwise strict semantics, there must be a probability-one set of random seeds `omega`,
independent of the sequence, such that for every fixed seed in that set and every input, the
deterministic router satisfies

\[
R_i(x;\omega)=R_i(x';\omega)
\]

whenever `x,x'` differ only on the active endpoints selected under that seed. Invariance only in
distribution or expectation is a weaker semantics and does not make each sampled support strict.

## 7. Fixed candidate universe

A stronger but simpler construction chooses

\[
U_i\subseteq[L]\setminus\{i\}
\]

fixed independently of the sequence and router seed, and imposes

\[
N_i(x)\subseteq U_i,
\qquad
N_i(x)=R_i(x_{[L]\setminus(\{i\}\cup U_i)}).
\]

The router may select dynamically within `U_i`, but it cannot observe any candidate residue. With
randomness it may additionally use only an independent seed. The cavity background excludes the
full `U_i`, regardless of which gates are active.

This gives a coverage-evidence tradeoff:

- large `U_i` can cover more possible contacts but leaves little sequence evidence for routing and
  weakens the background branch;
- small `U_i` preserves background information but may omit true long-range interactions.

## 8. Three honest architecture levels

### A. Fixed strict support

Use position-only local, dilated, random, or expander edges. This is immediately cavity-measurable
and hardware-friendly. BigBird, Exphormer, and Diffuser provide precedents for global connectivity
or multi-hop receptive fields under predefined sparse structures. They do not make the direct edge
set coincide with protein contacts or supply categorical marginal semantics.

### B. Certificate or subcube router

Use a hard decision program whose leaf certifies a neighbor set from evidence outside that set.
This is genuinely content-dependent and strict. A hard sampled forward can be trained with
score-function gradients or a biased straight-through estimator. The unresolved questions are
approximation from third-party evidence, gradient variance, coverage, and efficient panel
generation. A standard soft mixture of leaves generally loses the exact support semantics.

### C. Ordinary dynamic top-k

Use full-context embeddings or leave-two-out pair scores followed by global competition. This is
the most practical router, but the resulting model has only local table-gauge semantics. It must
not be presented as global entropy/interaction disentanglement.

## 9. Implication for the candidate architecture

The simplest strict version uses a fixed sparse candidate graph `E_0` and an endpoint-removable
affine recurrence context. This gives a fully closed mathematical baseline:

\[
E_0\text{ fixed},
\qquad
c_{ij}=M_{-\{i,j\}}h_0,
\qquad
c_i^E=M_{-(\{i\}\cup N_i)}h_0.
\]

An endpoint-blind gate can attenuate values on `E_0`, but the background cavity continues to
exclude the whole fixed candidate neighborhood.

The strict dynamic candidate is an oblivious scout certificate router: each target reads a bounded
position-only scout set, a hard controller emits a bounded teacher-mass source envelope disjoint
from those scouts, and the realized envelope defines the whole-neighborhood cavity. The leaf panel
is a posterior decision under missed teacher mass plus compute cost, not a claim to recover a
latent contact graph. Ordinary top-k is not an admissible replacement if the strict theorem is
retained. The recoverability and envelope derivation is in
`docs/support_recoverability_and_envelope_theory.md`.

## 10. What static sparse support does not solve

Protein contact graphs are sequence-dependent and can contain arbitrary long-range edges. A fixed
bounded-degree graph cannot guarantee a direct edge for every possible contact. Expander or
multi-hop communication can move information globally, but a path-mediated effect is not the same
identified object as a directly supervised categorical field `G_ij`.

More formally, let `E_1,...,E_H` be fixed undirected candidate graphs, each with maximum degree
`k`. Their union has at most

\[
\left|\bigcup_{h=1}^{H}E_h\right|
\le\sum_h|E_h|
\le\frac{HLk}{2}
\]

direct edges. To guarantee that every unordered position pair occurs directly in at least one
layer requires

\[
\frac{HLk}{2}\ge\frac{L(L-1)}{2},
\qquad\text{hence}\qquad
Hk\ge L-1.
\]

Thus bounded degree and `O(log L)` depth cannot guarantee direct coverage of arbitrary contacts.
The condition `Hk>=L-1` is necessary, not sufficient, because layers may repeat edges.
This lower bound does not deny global communication through expander paths; it distinguishes
communication reachability from direct pair-object coverage.

Consequently, there is no completed architecture solution yet. The current choices are:

1. preserve strict semantics with fixed support and accept incomplete direct-contact coverage;
2. build a hard certificate envelope router and quantify Bayes recoverability, teacher-mass
   recall, coverage, and optimization gaps;
3. use practical dynamic routing and explicitly weaken the claim to local gauge separation.

The proposition above is per target and directed. An undirected graph additionally requires

\[
j\in R_i(x)\iff i\in R_j(x),
\]

which is not guaranteed by independent per-node decision trees or subcube certificates.

## 11. Prior art

- Kothari, Racicot-Desloges, and Santha, *Separating Decision Tree Complexity from Subcube
  Partition Complexity*, 2015/2017.
- Ambainis and Kokainis, *Almost Quadratic Gap Between Partition Complexity and Query/Communication
  Complexity*, 2015/2017.
- Bengio, Leonard, and Courville, *Estimating or Propagating Gradients Through Stochastic Neurons
  for Conditional Computation*, 2013.
- Mnih et al., *Recurrent Models of Visual Attention*, NeurIPS 2014.
- Schulman et al., *Gradient Estimation Using Stochastic Computation Graphs*, 2015.
- Kontschieder et al., *Deep Neural Decision Forests*, ICCV 2015.
- Tanno et al., *Adaptive Neural Trees*, ICML 2019.
- Janisch, Pevny, and Lisy, *Classification with Costly Features Using Deep Reinforcement
  Learning*, AAAI 2019.
- Zaheer et al., *Big Bird: Transformers for Longer Sequences*, NeurIPS 2020.
- Shirzad et al., *Exphormer: Sparse Transformers for Graphs*, ICML 2023.
- Feng et al., *Diffuser: Efficient Transformers with Multi-hop Attention Diffusion for Long
  Sequences*, 2022/2023.

Subcube-partition theory supplies the combinatorial language. Hard attention, stochastic gates,
neural trees, and active feature acquisition supply optimization templates. Sparse-Transformer
work supplies static global-connectivity constructions. None of the inspected literature imposes
the unqueried-leaf certificate for a whole-neighborhood categorical cavity with optional-MSA
supervision.
