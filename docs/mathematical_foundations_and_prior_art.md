# Mathematical Foundations and Prior-Art Boundary

Date: 2026-07-12

## 1. The central distinction

There are two different common-mode problems.

1. **Token-space common mode.** A row-stochastic attention matrix mixes positions and has a
   Perron right eigenvector `1`. Repeated pure attention can drive token representations toward
   uniformity or rank collapse.
2. **Categorical marginal mode.** A categorical pair field can contain functions of the left
   residue alone or the right residue alone. These one-body functions are confounded with
   conservation and entropy.

The first acts on the sequence-position axis. The second acts on the alphabet-state axes.
They require different projections. Removing one leading eigenvector from a token attention
matrix does not make a categorical interaction identifiable, and a categorical zero-sum gauge
does not prevent token uniformity.

## 2. Conditional functional ANOVA: the actual identifiability claim

The conditioning variables must be explicit. For an edge `(i,j)`, let `C_ij` be a sigma-algebra
that excludes the two endpoint categories `X_i,X_j`. Conditional on `c in C_ij`, define

\[
\mu_{ij}(a,b\mid c)=p_i(a\mid c)p_j(b\mid c),
\]

with full-support marginals. The score table `S_ij(a,b;c)` and edge selector `z_ij(c)` must be
measurable with respect to this same endpoint-excluding context while `a,b` vary. Define

\[
P_i(c)=I-\mathbf 1p_i(\cdot\mid c)^\top,
\qquad
G_{ij}(c)=P_i(c)S_{ij}(c)P_j(c)^\top.
\]

Then, for every fixed `c`,

\[
p_i(\cdot\mid c)^\top G_{ij}(c)=0,
\qquad
G_{ij}(c)p_j(\cdot\mid c)=0.
\]

This is a conditional functional-ANOVA statement: the pair table cannot represent a left-only
or right-only categorical function for fixed `c`. It does **not** say interaction strength must
be empirically uncorrelated with entropy, and it does not imply that the population pair support
is sparse. Sparsity is an extra structural assumption.

The endpoint-exclusion condition is essential. If the marginal network, router, or score network
uses the observed `x_i` or `x_j`, double-centering proves only that its emitted `q x q` table has
weighted-zero rows and columns. The dynamic network can still hide one-body information in the
context dependence of that table or in edge selection. There are therefore two honest variants:

1. **Strict conditional ANOVA:** construct a leave-pair-out or otherwise endpoint-removable
   relation context and impose the measurability condition above.
2. **Local table gauge:** use ordinary full-context hidden states, retain the useful numerical
   projection, but make no global model-function ANOVA claim.

There is a shared-reference exclusion constraint. If one `p_i` is used for all active neighbors
`j in N_i`, strict pairwise ANOVA requires

\[
p_i\text{ measurable with respect to }
\mathcal C_i^E=\sigma\{X_k:k\notin\{i\}\cup N_i\}.
\]

This follows by intersecting the required `C_ij` over active neighbors. If every other position
is a possible neighbor, a strict shared `p_i` cannot depend on any sequence residue. Sparse support
relaxes the condition, but a router that observes `x_i,x_j` can itself leak one-body information.
Thus a rich shared marginal, endpoint-dependent routing, and strict all-edge ANOVA cannot simply
be assumed simultaneously.

The reference is also part of the function space. Teacher, student, norm, and spectral analysis
must use one identical reference. The training construction below uses the same detached `bar p`
directly for both branches, rather than comparing tables projected under different gauges.

## 3. Entropy is a marginal coordinate, not the interaction score

The site entropy is

\[
H_i=-\sum_a p_i(a)\log p_i(a).
\]

It is fully determined by the marginal branch. A pair interaction should be measured in the
weighted geometry induced by the same marginals:

\[
\lVert G_{ij}\rVert_{p_i,p_j}^{2}
=\sum_{a,b}p_i(a)p_j(b)G_{ij}(a,b)^2.
\]

Let `D_i=diag(p_i)` and define the whitened field

\[
\overline G_{ij}=D_i^{1/2}G_{ij}D_j^{1/2}.
\]

It satisfies

\[
\sqrt{p_i}^{\,\top}\overline G_{ij}=0,
\qquad
\overline G_{ij}\sqrt{p_j}=0,
\]

and

\[
\lVert G_{ij}\rVert_{p_i,p_j}=\lVert\overline G_{ij}\rVert_F.
\]

This gives a marginal-aware edge strength. Plain Frobenius norms of unwhitened blocks can be
dominated by rare states or conservation and are not comparable across sites.

## 4. Exact marginal preservation: the transport-polytope view

Assume all marginal probabilities are strictly positive, using explicit smoothing when needed.
The linear gauge identifies functions, but zero-mean interaction logits do not guarantee that
post-softmax probabilities retain the background marginals. Exact pairwise separation uses a
coupling with prescribed marginals:

\[
\mathcal U(p_i,p_j)=
\{Q\ge0:Q\mathbf1=p_i,\ Q^\top\mathbf1=p_j\}.
\]

This transportation polytope has dimension `(q-1)^2`. Independence is

\[
Q_{ij}^{0}=p_i p_j^\top.
\]

Define the centered density-ratio interaction

\[
C_{ij}=D_i^{-1}(Q_{ij}-p_ip_j^\top)D_j^{-1}.
\]

It obeys

\[
p_i^\top C_{ij}=0,
\qquad
C_{ij}p_j=0.
\]

The standardized residual

\[
R_{ij}=D_i^{-1/2}(Q_{ij}-p_ip_j^\top)D_j^{-1/2}
\]

is the correspondence-analysis representation of the coupling. Its singular values are
association modes, and its rank is at most `q-1`.

An unconstrained score matrix `S_ij` can be mapped to an exact coupling by matrix scaling:

\[
K_{ij}(a,b)=\exp S_{ij}(a,b),
\qquad
Q_{ij}=\operatorname{diag}(u_{ij})K_{ij}\operatorname{diag}(v_{ij}),
\]

where Sinkhorn/IPF chooses positive `u_ij,v_ij` so the row and column sums are `p_i,p_j`.
The additive row and column corrections in log space are the nonlinear counterparts of gauge
fixing. For strictly positive marginals and kernel this is a generalized-KL/I-projection. Around
independence, its first-order perturbation is

\[
\delta Q_{ij}=D_i(P_iS_{ij}P_j^\top)D_j,
\]

so the exact coupling linearizes to the weighted double-centering operator used above.

The exact pair conditional is

\[
q_{i\mid j}(a\mid b)=\frac{Q_{ij}(a,b)}{p_j(b)}.
\]

It is normalized for every `b`, and averaging over `b~p_j` returns `p_i` exactly. Therefore a
single bivariate distribution has prescribed marginal coordinates and a separately parameterized
dependence coordinate. This is established territory: marginal log-linear models, discrete
dependence/copula models, and minimum-information dependence modeling already study closely
related separation.

The mutual information

\[
I_{ij}=\mathrm{KL}(Q_{ij}\|p_ip_j^\top)
\]

is an exact non-negative interaction strength. Near independence,

\[
I_{ij}=\tfrac12\lVert R_{ij}\rVert_F^2+O(\lVert R_{ij}\rVert_F^3).
\]

This expansion requires fixed, strictly positive marginals and

\[
\max_{a,b}\left|
\frac{Q_{ij}(a,b)-p_i(a)p_j(b)}{p_i(a)p_j(b)}
\right|\to0.
\]

The remainder constant depends on the smallest marginal probability. The approximation can be
poor for unsupported or extremely rare categories, and MSA estimates need an explicit
pseudocount/full-support convention.

There are three distinct levels of separation:

1. **Functional:** weighted-zero-sum logits identify the ANOVA pair subspace.
2. **Pair-probabilistic:** a transport coupling preserves both specified site marginals exactly.
3. **Global-probabilistic:** all pair couplings arise from one consistent joint distribution.

Level 3 is substantially harder: locally consistent pair marginals need not belong to the global
marginal polytope. A sparse neural layer should not claim a globally normalized graphical model
unless it performs or approximates that inference.

For a single selected neighbor, `q_i|j` is a marginal-calibrated pair conditional. Multiple
neighbors can be combined as a convex pool

\[
\widehat p_i
=(1-\gamma_i)p_i
+\gamma_i\sum_{j\in\mathcal N_i}\alpha_{ij}
q_{i\mid j}(\cdot\mid x_j),
\]

with `gamma_i in [0,1]`, non-negative `alpha_ij`, and `sum_j alpha_ij=1`. This always produces
a valid distribution. Averaging returns `p_i` under any context distribution whose individual
marginals are `p_j`, provided `gamma_i` and `alpha_ij` do not depend on the sampled context
residues. Context-dependent routing weakens that exact expectation guarantee and must be stated
explicitly. Even with fixed weights, the pool is generally not the posterior of one globally
consistent joint distribution and does not preserve `p_i` pointwise for an observed context.

## 5. Sparse support and categorical rank are different axes

Sparsity answers **which position pairs** need expensive interaction computation:

\[
z_{ij}\in\{0,1\},
\qquad
E=\{(i,j):z_{ij}=1\}.
\]

Rank answers **how many categorical modes** an already selected pair requires. For the whitened
field,

\[
\overline G_{ij}
=\sum_{r=1}^{q-1}\sigma_{ijr}u_{ijr}v_{ijr}^\top.
\]

The maximum nontrivial rank is `q-1`, because the marginal direction has been removed. For a
nonzero field define the error-controlled approximation rank

\[
r_{ij}(\epsilon)=\min\left\{r:
\frac{\sum_{s>r}\sigma_{ijs}^{2}}
{\sum_s\sigma_{ijs}^{2}}\le\epsilon\right\}.
\]

and define `r_ij(epsilon)=0` when the field is zero. Different pairs need not share the same
approximation rank, but adaptive rank is not required by the mathematics. Fixed full rank `q-1`
is valid. Fixed low rank is a testable approximation hypothesis, not an error in itself. Adaptive
rank also has selection cost: if the full spectrum is computed before gating, no compute is saved.

For proteins, the full tangent-product space is `19 x 19` and its maximum matrix rank is 19. The
theory-first default should retain this capacity while sparsifying the position-pair support.
Low-rank decoding can be reconsidered as compression or regularization only after an
accuracy/compute curve over the singular tail and task-relevant geometry are reported.

## 6. Non-symmetric attention and the Perron component

For a fixed irreducible row-stochastic attention matrix `A`, let `pi` be its stationary left
distribution:

\[
A\mathbf 1=\mathbf 1,
\qquad
\pi^\top A=\pi^\top.
\]

The Perron projector is

\[
\Pi_0=\mathbf 1\pi^\top,
\]

and the centered token mixer is

\[
A_\perp=(I-\Pi_0)A(I-\Pi_0).
\]

Because Transformer attention is input-dependent and generally non-normal, eigenvectors alone
are insufficient: transient amplification and information transport are governed by singular
vectors and pseudospectral behavior as well. The relevant lesson from rank-collapse work is not
to delete the first eigenvector blindly, but to preserve an explicit residual/local path and to
control the token-common mode separately from the categorical marginal gauge.

## 7. Training-time separation with optional MSA

The architecture should use one reference marginal in every loss. Let the background branch
produce `p_theta`, and let

\[
\bar p_i=\operatorname{stopgrad}(\operatorname{EMA}(p_i^\theta))
\]

be the slowly moving gauge used during one optimization step. The interaction branch predicts

\[
G_{ij}^\theta
=P_{\bar p_i}S_{ij}^\theta P_{\bar p_j}^\top.
\]

When an MSA is present, estimate marginals and a leave-query-out log-density-ratio field

\[
L_{ij}^{\mathrm{MSA}}(a,b)
=\log\frac{P_{ij}^{\mathrm{MSA}}(a,b)}
{p_i^{\mathrm{MSA}}(a)p_j^{\mathrm{MSA}}(b)},
\]

then reproject it into the model gauge:

\[
G_{ij}^{T}=P_{\bar p_i}L_{ij}^{\mathrm{MSA}}P_{\bar p_j}^\top.
\]

A theory-aligned objective is

\[
\begin{aligned}
\mathcal L={}&\mathcal L_{\mathrm{task}}(b+I)
+\lambda_m\sum_i\mathrm{KL}(p_i^{\mathrm{MSA}}\|p_i^\theta)\\
&+\lambda_G\sum_{(i,j)\in E}
\lVert G_{ij}^{\theta}-G_{ij}^{T}\rVert_{\bar p_i,\bar p_j}^{2}
+\lambda_E\lvert E\rvert.
\end{aligned}
\]

Gradient ownership is part of the definition:

- marginal KL and entropy losses update the background branch;
- interaction reconstruction updates the relation/value branch;
- under the endpoint-excluding context contract, the hard projection prevents the interaction
  table from representing one-body categorical functions;
- routing control variables may affect selection but not semantic value weights;
- the task loss may update both branches after the two subspaces have been defined.

The EMA and stop-gradient stabilize optimization and keep teacher/student in one instantaneous
gauge. They are not an identifiability theorem if the context network itself observes the two
endpoint residues.

Without an MSA, the same model runs from one sequence. The marginal branch is trained by masked
prediction or downstream likelihood. Interaction supervision can come from conditional response,
double-mutation data, structure, or the task loss itself. MSA changes the teacher, not the
forward contract.

## 8. What can and cannot be novel

The following are established prior art and cannot carry the novelty claim:

- weighted/zero-sum gauges in Potts and direct-coupling models;
- Hoeffding or functional-ANOVA decomposition;
- sparse attention, top-k routing, low-rank values, MoE, or load balancing;
- persistent pair representations and triangle updates in AlphaFold2/3;
- single-sequence pair tracks in ESMFold;
- edge states and node-edge-node updates in Interaction Networks and Graph Networks;
- marginal/dependence parameter separation in marginal log-linear, discrete-copula, and
  minimum-information dependence models;
- IPF/Sinkhorn and convex pools of conditional distributions.

The narrow candidate contribution is the combination:

> a single-sequence-native sparse relation operator with an endpoint-excluding conditional
> reference, whose categorical values are decoded in the model-marginal tangent-product space,
> with optional leave-query-out MSA supervision in that same gauge.

This is still a hypothesis, not a novelty claim. It must be compared with an ordinary sparse
edge GNN, an ungauged relation decoder, and a dense pair-track baseline.

## 9. Architecture implied by the mathematics

Maintain:

- node/background state `h_i`, which predicts `p_i` and local logits;
- an endpoint-excluding relation context `c_ij` for strict identifiability;
- a cheap router that chooses a bounded edge set `E`, with endpoint blindness required only for
  the strict cavity claim;
- optional edge state `r_ij` only on `E`;
- a value decoder that defaults to the full `q-1` tangent space for proteins.

The probability-exact variant decodes a transport coupling rather than an arbitrary field:

\[
S_{ij}=f_S(r_{ij}),
\qquad
Q_{ij}=\operatorname{Sinkhorn}_{p_i,p_j}(S_{ij}),
\qquad
q_{i\mid j}(:,b)=Q_{ij}(:,b)/p_j(b).
\]

The conservative name is a **coupling-valued sparse mixer**, or more exactly a convex pool of
marginal-calibrated pair conditionals. Sinkhorn scaling, transport polytopes, conditional
mixtures, correspondence analysis, and minimum-information dependence models are prior art. The
research question is whether an endpoint-excluding neural implementation can preserve the stated
identifiability while serving as an optional-MSA single-sequence interface.

A practical local-gauge layer can be written

\[
m_i=\sum_{j:(i,j)\in E}\phi(h_j,r_{ij}),
\]

\[
r_{ij}'=r_{ij}+F_r(r_{ij},h_i,h_j,m_i,m_j),
\]

\[
G_{ij}=P_{\bar p_i}f_G(r_{ij}')P_{\bar p_j}^\top,
\qquad
\ell_i=b_i+\sum_{j:(i,j)\in E}G_{ij}(:,x_j).
\]

For the main sequence operator, interpret the support and values directionally as
`G_{i<-j}`. Transpose-related targets from an MSA and a symmetrized contact score are optional
auxiliary constructions; neither implies reciprocal routing or a globally invariant graph.

This resembles existing pair-track and edge-GNN architectures. The mathematical distinction is
only the model-marginal local table gauge and the common-gauge teacher interface; ordinary
contextual states do not satisfy the strict conditional theorem.

The strict targetwise cavity variant uses one common context
`C_i^E = sigma(X_k: k not in {i} union N_i)` and predicts directional references

\[
\bar p_{u\mid i}^E
=\operatorname{stopgrad}\operatorname{EMA}
\bigl(p_{u\mid i}^\theta(\cdot\mid\mathcal C_i^E)\bigr),
\qquad u\in\{i\}\cup N_i.
\]

\[
r_{i\leftarrow j}'=r_{i\leftarrow j}+F_r(r_{i\leftarrow j},c_i^E),
\qquad
G_{i\leftarrow j}=P_{\bar p_{i\mid i}^E}f_G(r_{i\leftarrow j}')
P_{\bar p_{j\mid i}^E}^\top,
\]

where every background, router, reference, and input to `F_r` is measurable with respect to the
same target cavity. The observed `x_j` is used only after every table is fixed, to select its own
message column. This closes a targetwise additive conditional-ANOVA model relative to the stated
product reference. It does not assert that the true sources are conditionally independent or that
the final predictive entropy is additive. If the support is data-dependent, the entire targetwise
source set `N_i`, not just each `z_ij`, must be invariant when the target or any active source
category is varied.
The strict MSA teacher is correspondingly

\[
G_{i\leftarrow j}^{T,E}=P_{\bar p_{i\mid i}^E}L_{ij}^{\mathrm{MSA}}
P_{\bar p_{j\mid i}^E}^\top,
\]

and teacher, student, norm, and spectral analysis all use the same directional references. Using
target `j`'s own `bar p_j^E` can fail even per-arc endpoint blindness when `i not in N_j`, because
that reference may read `x_i`. The construction
does not require quadratically many encoder calls if the context engine is deletion-compatible.
One exact ordered construction maps each token to an affine recurrence

\[
F_k(h)=A_k(x_k)h+b_k(x_k),
\qquad
M_k=\begin{bmatrix}A_k&b_k\\0&1\end{bmatrix}.
\]

Replacing every map in `{i} union N_i` by identities and composing the remaining maps in order is
exactly target-cavity invariant. A segment tree answers this bounded-degree deletion query in
`O(k log L C_comp)`, and its result is shared across all values directed into target `i`. The
unresolved issue is no longer deletion correctness but whether a compact affine state retains
enough information for `f_G`.

A weaker per-arc construction may delete only `i,j`, but must then use pair-specific references
predicted from that same pair context. It proves only a per-arc conditional gauge: other selected
sources may modulate `G_{i<-j}`, so it is not the shared-background targetwise decomposition.

There are unavoidable capacity qualifications. At `B` bits per coordinate, an injective
`d`-dimensional summary over retained sequences requires

\[
Bd\ge(L-|A|)\log_2 q.
\]

If every centered retained one-site function must be linearly readable, then

\[
d\ge(L-|A|)(q-1).
\]

The detailed constructions and proofs are in `docs/removable_context_theorems.md`.

Strict dynamic support has a separate combinatorial condition. If `R_i(x)=N_i(x)`, then changing
`x_i` and every selected `x_j`, `j in N_i(x)`, must leave the entire output set unchanged. The
induced cylinders

\[
\{x':x'_{[L]\setminus(\{i\}\cup N_i(x))}
=x_{[L]\setminus(\{i\}\cup N_i(x))}\}
\]

must partition sequence space with the router constant on each cell. This is a label-constrained
`q`-ary subcube partition. Pairwise endpoint-blind scores followed by top-k do not imply this
property because a selected residue can alter the scores of other candidate edges.

A decision tree is sufficient when each leaf selects only coordinates never queried on its
realized path. A hard sampled forward remains strict pathwise for every fixed independent random
seed and can use score-function gradients; straight-through estimation preserves the hard forward
but biases the gradient. Strict support also forces `N_i` to be independent of `x_i` and forces
the membership of candidate `j` to be independent of `x_j`. Thus an endpoint-only rule cannot
define strict support: endpoint categories enter the value only after third-party evidence opens
the channel.

A depth-`B` q-ary tree with at most `k` sources per leaf covers at most `k q^B` distinct source
identities for one target. Full direct positional coverage requires `k q^B >= L-1`; realizing all
size-`k` source sets requires `q^B >= choose(L-1,k)`. These are necessary counting bounds.

Fixed sparse support is the simplest strict case. An oblivious scout tree with a hard generated
source panel is the first subquadratic dynamic candidate. BigBird, Exphormer, and Diffuser
show that static sparse/expander graphs can carry global information efficiently, but they do not
guarantee a direct edge for every sequence-specific protein contact. The full derivation is in
`docs/cavity_support_routing_theory.md` and `docs/trainable_certificate_router_theory.md`.

Strict membership has a separate statistical limit. For any training-only label `T_ij`, every
strict membership decision is measurable with respect to `Y_ij=X_-{i,j}`. Writing

\[
\eta_{ij}(y)=P(T_{ij}=1\mid Y_{ij}=y),
\]

its error is at least

\[
E\min\{\eta_{ij}(Y_{ij}),1-\eta_{ij}(Y_{ij})\}.
\]

For randomized routing this lower bound assumes the inference seed is independent of the latent
family, sequence, and teacher labels or weights, not merely independent of the observed sequence.

MSA supervision can estimate this posterior across families but cannot remove conditional
uncertainty at single-sequence inference. The unavailable endpoint label information is

\[
I(T_{ij};X_i,X_j\mid Y_{ij}),
\]

which is a recoverability quantity rather than a causal interaction score.

The design target is therefore a stable-by-certificate, budgeted teacher-mass envelope. At a
certificate leaf with transcript `H`, an integrable teacher importance `W_ij`, per-edge cost
`lambda`, and degree budget `k`, the
Bayes-optimal leaf selects the largest positive values of

\[
E[W_{ij}\mid H]-\lambda
\]

among unqueried positions. Scout acquisition is an active-feature problem with an extra
opportunity cost because a queried coordinate cannot be emitted. The complete proof is in
`docs/support_recoverability_and_envelope_theory.md`.

If `W_ij` comes from an MSA categorical field, its gauge, marginal normalization, and geometry
must be stated; otherwise it may remain entropy-contaminated and is only a routing heuristic.

A stronger global condition would require one undirected graph to remain unchanged when all
categories at its active vertices vary. If any realized graph has no isolated vertices, that
condition makes the router constant on the full sequence domain. We choose a directed targetwise
operator to avoid conflating conditional routing with a globally identified graph; reciprocity
can remain nontrivial, but it does not upgrade the semantics to global graph invariance. Contact
symmetrization may be a useful output, but it is not part of the conditional-identifiability
theorem. The proof and semantic
distinction are in `docs/global_support_no_go_and_directed_design.md`.

## 10. Reviewer-imposed research gates

Before further architecture experiments:

1. prove every teacher and student field uses the same gauge;
2. report marginal KL to the MSA rather than freezing it silently;
3. audit MSA/profile overlap across family splits;
4. use family as the highest bootstrap cluster;
5. distinguish pair-support sparsity from categorical rank;
6. compare against ungauged and ordinary edge-state baselines;
7. do not call the current convolution-GRU prototype a Transformer replacement;
8. require endpoint-excluding marginals, scores, and routing for a global functional-ANOVA claim;
9. do not describe pairwise calibrated couplings as a globally consistent joint model;
10. use full `19 x 19` tangent-product capacity as the protein default; report fixed/adaptive
    low-rank variants only as compression or regularization choices with singular-tail error,
    task effect, and realized compute;
11. for a strict targetwise claim, use the same whole-neighborhood cavity for the support,
    background, every value, both directional references, teacher, norm, and spectrum;
12. test numerical endpoint invariance on retained-range arithmetic rather than assuming that
    subtracting endpoints from a rounded global summary is exact;
13. treat strict dynamic routing as a joint incident-set invariance problem; leave-two-out scores
    plus top-k are insufficient.
14. do not equate reciprocal targetwise predictions or a symmetrized contact readout with a
    globally invariant undirected support.
15. require a hard certificate-valid forward for strict dynamic training; a soft tree needs a
    separate proof, with its positive-mass union and every gauge-dependent quantity using one
    union cavity in the auditable construction.
16. report a certificate envelope's captured teacher-mass fraction, set recall, and Bayes
    approximation gap separately; do not call
    optional-MSA supervision exact recovery of a latent interaction graph.

## 11. Selected prior art

- Hoeffding, *A Class of Statistics with Asymptotically Normal Distribution*, 1948.
- Lancaster, *The Structure of Bivariate Distributions*, 1958.
- Deming and Stephan, *On a Least Squares Adjustment of a Sampled Frequency Table When the
  Expected Marginal Totals Are Known*, 1940.
- Sinkhorn, *A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic
  Matrices*, 1964.
- Greenacre, *Theory and Applications of Correspondence Analysis*, 1984.
- Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*, NeurIPS 2013.
- Darroch and Ratcliff, *Generalized Iterative Scaling for Log-Linear Models*, 1972,
  DOI 10.1214/aoms/1177692379.
- Csiszar, *I-Divergence Geometry of Probability Distributions and Minimization Problems*, 1975,
  DOI 10.1214/aop/1176996454.
- Bergsma and Rudas, *Marginal Models for Categorical Data*, 2002,
  DOI 10.1214/aos/1015362188.
- Geenens, *Copula Modeling for Discrete Random Vectors*, 2020,
  DOI 10.1515/demo-2020-0022.
- Sei and Yano, *Minimum Information Dependence Modeling*, Bernoulli 2024,
  DOI 10.3150/23-BEJ1687.
- Berchtold and Raftery, *The Mixture Transition Distribution Model for High-Order Markov Chains
  and Non-Gaussian Time Series*, 2002, DOI 10.1214/ss/1042727943.
- Morcos et al., *Direct-coupling analysis of residue coevolution captures native contacts*,
  PNAS 2011.
- Ekeberg et al., *Improved contact prediction in proteins: Using pseudolikelihoods to infer
  Potts models*, Physical Review E 2013.
- Cocco et al., *Inverse statistical physics of protein sequences: a key issues review*,
  Reports on Progress in Physics 2018.
- Wang et al., *Disentanglement of Evolutionary Constraints in Statistical Models of Proteins*,
  PRX Life 2024, DOI 10.1103/PRXLife.2.023005.
- Dong, Cordonnier, and Loukas, *Attention is Not All You Need: Pure Attention Loses Rank
  Doubly Exponentially with Depth*, 2021/2023.
- Rao et al., *MSA Transformer*, ICML 2021.
- Jumper et al., *Highly accurate protein structure prediction with AlphaFold*, Nature 2021.
- Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language
  model*, Science 2023 (ESMFold).
- Abramson et al., *Accurate structure prediction of biomolecular interactions with AlphaFold 3*,
  Nature 2024.
- Battaglia et al., *Interaction Networks for Learning about Objects, Relations and Physics*,
  NeurIPS 2016.
- Battaglia et al., *Relational inductive biases, deep learning, and graph networks*, 2018.
- DeepSeek-V2/V3/V3.2 and Native Sparse Attention for compression, routing/value separation,
  and hardware-aware sparsity; none supplies categorical identifiability.
- Zaheer et al., *Deep Sets*, NeurIPS 2017.
- Katharopoulos et al., *Transformers are RNNs*, ICML 2020.
- Choromanski et al., *Rethinking Attention with Performers*, ICLR 2021.
- Smith, Warrington, and Linderman, *Simplified State Space Layers for Sequence Modeling*, ICLR 2023.
- Katsch, *GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling*, 2023.
- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023/2024.
- Kothari, Racicot-Desloges, and Santha, *Separating Decision Tree Complexity from Subcube
  Partition Complexity*, 2015/2017.
- Ambainis and Kokainis, *Almost Quadratic Gap Between Partition Complexity and
  Query/Communication Complexity*, 2015/2017.
- Zaheer et al., *Big Bird: Transformers for Longer Sequences*, NeurIPS 2020.
- Shirzad et al., *Exphormer: Sparse Transformers for Graphs*, ICML 2023.
- Feng et al., *Diffuser: Efficient Transformers with Multi-hop Attention Diffusion for Long
  Sequences*, 2022/2023.
- Bengio, Leonard, and Courville, *Estimating or Propagating Gradients Through Stochastic Neurons
  for Conditional Computation*, 2013.
- Mnih et al., *Recurrent Models of Visual Attention*, NeurIPS 2014.
- Schulman et al., *Gradient Estimation Using Stochastic Computation Graphs*, 2015.
- Kontschieder et al., *Deep Neural Decision Forests*, ICCV 2015.
- Tanno et al., *Adaptive Neural Trees*, ICML 2019.
- Janisch, Pevny, and Lisy, *Classification with Costly Features Using Deep Reinforcement
  Learning*, AAAI 2019.
- Cover and Thomas, *Elements of Information Theory*, second edition, 2006.
- Feder and Merhav, *Relations Between Entropy and Error Probability*, IEEE Transactions on
  Information Theory 1994, DOI 10.1109/18.272494.
- Brown et al., *Conditional Likelihood Maximisation: A Unifying Framework for Information
  Theoretic Feature Selection*, JMLR 2012.
- Covert et al., *Learning to Maximize Mutual Information for Dynamic Feature Selection*, 2023.
