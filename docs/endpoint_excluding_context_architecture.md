# Endpoint-Excluding Context as the Architecture Bottleneck

Date: 2026-07-12

Status: theory proposal; no experiments should be started from this note yet.

## 1. Why the projector is not enough

For a fixed context `c_ij`, weighted double-centering gives an identifiable categorical table:

\[
G_{ij}(c_{ij})
=P_{p_i(\cdot\mid c_{ij})}S_{ij}(c_{ij})
P_{p_j(\cdot\mid c_{ij})}^{\top}.
\]

If the network that emits `S_ij`, its reference marginals, or its routing decision observes
`x_i` or `x_j`, it can encode one-body information in that context dependence before the table is
centered. The output matrix still has weighted-zero rows and columns, but the full neural function
is not a functional-ANOVA pair component.

The central architecture question is therefore:

> Can we construct useful pair contexts that exclude their two endpoint residues without one
> separately masked encoder pass per pair?

## 2. Shared-reference exclusion constraint

Let `N_i` be the selected neighbors of position `i`. If one shared site marginal `p_i` is used for
all edges `(i,j)`, strict pairwise ANOVA requires `p_i` to remain fixed as every `x_j`, `j in N_i`,
is varied. Hence

\[
\mathcal C_i^E
=\sigma\{X_k:k\notin\{i\}\cup N_i\},
\qquad
p_i^E(\cdot)=p_i(\cdot\mid\mathcal C_i^E).
\]

This has two consequences.

1. Selected neighbor residues must be removed from the background branch and enter the prediction
   only through relation values.
2. A router that uses `x_i,x_j` can itself transmit one-body information through support choice.
   Strict support semantics therefore require endpoint-blind routing as well.

If every other site is a candidate neighbor, a strict shared marginal cannot depend on any other
residue. Sparse support is not only a compute device here: it determines which variables the
background is allowed to observe.

## 3. Candidate architecture: Cavity-Gauge Sequence Mixer

Working name only; no novelty claim.

### 3.1 Removable local features

Start from token-local features

\[
u_k=\psi(x_k,k),
\qquad
T=\sum_{k=1}^{L}u_k.
\]

For a candidate edge and a selected neighborhood,

\[
T_{-ij}=T-u_i-u_j,
\qquad
T_i^E=T-u_i-\sum_{j\in N_i}u_j.
\]

These are algebraically endpoint-removable in `O(d)` per edge or selected neighbor after an
`O(Ld)` sequence summary. Exactness requires `u_k` to be token-local; using ordinary contextual
hidden states would reintroduce indirect endpoint leakage. In floating point, subtracting endpoint
terms from a rounded global sum need not be bitwise invariant. A strict implementation should
query retained blocks that never include the endpoints, or use an exact accumulator.

### 3.2 Ordered removable summaries

A single sum is too permutation-invariant. Preserve order with several additive summaries:

\[
T^{(m)}=\sum_k \omega_m(k)\psi_m(x_k),
\]

where `omega_m(k)` may be a fixed Fourier feature, multiresolution bin, or learned position-only
basis. Range/block summaries additionally provide ordered statistics for the segments before,
between, and after `(i,j)`. The algebraic deletion stays exact, and a selected edge query remains
small; strict finite-precision invariance again requires an arithmetic path that never includes
the deleted terms.

This construction is related to Deep Sets and additive kernel features; subtraction is an
engineering property, not a new statistical primitive.

### 3.3 Support contract

The clean theorem conditions on a fixed sparse support `E`. Pairwise removable scores followed by
top-k are not sufficient for a strict dynamic support: changing one selected residue can alter the
scores of other candidates and change the complete incident set.

\[
N_i(x)=N
\Longrightarrow
N_i(x')=N
\quad\text{whenever}\quad
x'_{[L]\setminus(\{i\}\cup N)}=x_{[L]\setminus(\{i\}\cup N)}.
\]

This induces a label-constrained subcube partition. A sufficient hard construction is a decision
tree whose leaf selects only coordinates not queried on the realized path. Fixed multiscale sparse
graphs satisfy the condition automatically; ordinary content-dependent top-k does not. Optional
MSA contacts, structure, or distillation can supervise a fixed prior or certificate router during
training while the inference contract remains a single sequence. The full derivation is in
`docs/cavity_support_routing_theory.md`.

### 3.4 Cavity background and full-rank categorical edge

After selecting a directed source set `N_i`, predict the background after deleting the target and
its complete selected source set:

\[
p_i^E=\operatorname{softmax} f_B(i,T_i^E),
\qquad
H_i^E=H[p_i^E].
\]

From the same target cavity, predict target-relative reference distributions for the target and
all selected sources, then decode the full `19 x 19` categorical tangent-product capacity, whose
maximum matrix rank is 19:

\[
\bar p_{u\mid i}^E
=\operatorname{stopgrad}\operatorname{EMA}
\bigl(p_{u\mid i}^\theta(\cdot\mid T_i^E)\bigr),
\qquad u\in\{i\}\cup N_i,
\]

\[
S_{i\leftarrow j}=f_G(i,j,T_i^E),
\qquad
G_{i\leftarrow j}=P_{\bar p_{i\mid i}^E}S_{i\leftarrow j}
P_{\bar p_{j\mid i}^E}^{\top}.
\]

The observed neighbor residue is used only after the table is fixed:

\[
\ell_i(a)=b_i^E(a)+\sum_{j\in N_i}G_{i\leftarrow j}(a,x_j).
\]

This gives a literal structural division:

- non-neighbor residues contribute through the cavity background and its entropy;
- selected neighbor residues contribute through signed categorical interaction tables;
- the selected support is targetwise and directed rather than a globally identified contact graph;
- full `19 x 19` tangent-product capacity is the default; fixed low rank is only a measured
  compression or regularization hypothesis;
- MSA changes training supervision, not the forward input.

The statement remains conditional, not causal. `G_ij` may depend on other non-endpoint residues,
but under the strict targetwise contract those residues must lie outside `{i} union N_i`.
Therefore context can modulate the complete targetwise decomposition, while selected source
categories enter only through their own final lookups. This does not assert conditional
independence of the true sources or additive decomposition of the final predictive entropy.

A weaker pair-cavity model can use `T_-ij` and pair-specific references for each arc. It gives a
valid per-arc conditional gauge, but another selected source can then modulate `G_{i<-j}`. It must
not be presented as the strong shared-background targetwise decomposition.

## 4. Alternative removable summaries

### Linear-attention statistics

For token-local keys and values,

\[
S=\sum_k\phi(k_k)v_k^\top,
\qquad
Z=\sum_k\phi(k_k),
\]

and both endpoints can be subtracted algebraically. This follows the sufficient-statistic
structure of linear attention and Performer-style feature maps. It is deletion-exact over exact
arithmetic for a separate one-layer cavity
stream. Stacking ordinary contextualized keys or values breaks endpoint exclusion because other
tokens then carry indirect endpoint information.

### Associative segment summaries

Map each token to an associative operator `M_k` and query products over the segments separated by
the removed endpoints. Prefix/suffix products, segment trees, or group inverses can answer range
queries efficiently. This connects to associative scans and state-space sequence models. The
affine recurrence construction is explicit:

\[
F_k(h)=A_k(x_k)h+b_k(x_k),
\qquad
M_k=
\begin{bmatrix}A_k&b_k\\0&1\end{bmatrix}.
\]

Replacing endpoint maps by the identity and composing the remaining maps in order gives an exact
endpoint-removable context. A segment tree answers a pair-deletion query by composing three
retained intervals in `O(log L)` range-query time up to transition-composition cost. A node cavity
with fixed degree `k` uses at most `k+2` retained intervals and costs `O(k log L)`.

GateLoop, S5, Mamba, and related models establish scan-compatible affine/data-controlled
recurrences, but not this deletion-aware categorical cavity use. The main limitation remains
finite-dimensional compression: exact removal does not imply that the summary retains all useful
sequence information.

### Expressivity limits

For fixed `L`, positional one-hot additive features in `R^(Lq)` make a deleted summary injective,
so exact endpoint removal is theoretically universal with an unrestricted decoder. This is not a
compact result. With `d` coordinates at `B` bits each, lossless encoding requires

\[
Bd\ge(L-|A|)\log_2q.
\]

If all centered one-site functions must be linearly readable, then

\[
d\ge(L-|A|)(q-1).
\]

Therefore a fixed-width summary can be exact and useful for a restricted function class, but
cannot be described as lossless for arbitrary growing sequences at fixed precision.

### Multiple masked views

Cross-fitted masked encodings can ensure that a chosen pair is hidden in at least one view. They
are useful as a diagnostic or training teacher, but a pair-covering family of masks adds many
encoder passes and does not yet provide an attractive inference primitive.

## 5. Three honest model levels

| Level | Context source | Guaranteed claim | Main cost/risk |
|---|---|---|---|
| Local coupling-valued edge | Ordinary full-context backbone | Each emitted table/coupling has the stated local gauge or marginals | Dynamic context can hide one-body information |
| Strict fixed-support cavity | Endpoint-removable summaries on a fixed sparse directed pattern | Conditional function-space separation for active arcs | Cannot guarantee direct coverage of arbitrary protein contacts |
| Strict certificate-routed cavity | Per-target hard certificate that never queries the target or selected sources | Dynamic directed support with targetwise conditional separation | Learnable, efficient certificate routing is unresolved |

The local model is deployable and may be useful, but its semantics are weaker. The two cavity
variants are the strict architecture hypotheses.

## 6. No-go conditions and falsifiers

The strict claim fails if any of the following occurs:

- `p_i^E`, `S_ij`, or `z_ij` directly consumes `x_i` or `x_j`;
- a strict targetwise table or projector consumes any selected source through a pair-only cavity;
- the right projector uses target `j`'s own cavity reference instead of a target-relative
  `p_{j|i}^E` built from target `i`'s common cavity;
- a supposedly removable feature was produced by a contextual layer that already mixed endpoints;
- node or edge aggregates reintroduce `h_i(x_i)` or `h_j(x_j)` before table construction;
- selected neighbor residues remain visible to the cavity background;
- teacher and student use different reference marginals;
- only individual edge scores are endpoint-blind while the full incident support changes with an
  active endpoint residue;
- pair support is called identifiable solely because the categorical table is gauge-fixed.
- directed predictions are symmetrized and then described as a globally invariant latent graph;
- a fixed low rank is presented as part of disentanglement rather than a separately tested
  approximation.

## 7. Remaining mathematics before experiments

1. Identify a useful restricted function class for low-dimensional affine segment summaries and
   quantify its approximation gap; unrestricted fixed-length universality requires large or
   ill-conditioned state.
2. Analyze the hard scout-certificate envelope's Bayes recoverability, captured teacher-mass
   fraction, set recall, coverage, and gradient-variance limits; test reciprocity or contact
   symmetrization only as an auxiliary readout.
3. Separate support identifiability from table-gauge identifiability, including overlapping-edge
   and higher-order ambiguity.
4. Compare this construction with conditional ANOVA, pseudo-likelihood/cavity methods, Deep Sets,
   linear attention, associative scans, and marginal dependence models before making a novelty
   claim.

## 8. Relevant prior art

- Zaheer et al., *Deep Sets*, NeurIPS 2017.
- Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear
  Attention*, ICML 2020.
- Choromanski et al., *Rethinking Attention with Performers*, ICLR 2021.
- Smith, Warrington, and Linderman, *Simplified State Space Layers for Sequence Modeling*, 2023.
- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023/2024.
- Katsch, *GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling*, 2023.
- Besag, *Statistical Analysis of Non-Lattice Data*, 1975, for conditional/pseudo-likelihood
  modeling context.

None of these references supplies the complete directed cavity-gauge categorical operator. Conversely,
Sei and Yano already supply the fixed-marginal dependence geometry, and scan/SSM papers already
supply the associative recurrence machinery. The remaining candidate contribution is their
deletion-aware use as a targetwise common-gauge sparse categorical context engine, whose useful
expressivity remains unproved.
