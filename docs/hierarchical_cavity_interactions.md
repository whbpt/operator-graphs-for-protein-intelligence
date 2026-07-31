# Hierarchical Cavity Interactions Beyond Pairwise Fields

Date: 2026-07-12

Status: mathematical design note; no experiments.

## 1. Why pair sparsity is not interaction completeness

The current strict cavity layer writes, for a fixed target `i`,

\[
\ell_i(a)=b_i^E(a)+\sum_{j\in N_i}G_{i\leftarrow j}(a,x_j).
\]

This is a first-order expansion in the selected source variables. The marginal gauge proves that
each pair table contains no target-only or source-only categorical function under its stated
reference. It does not prove that the target conditional contains no interaction involving two or
more selected sources jointly.

Consequently, three notions must remain separate:

1. positional support sparsity: only a bounded source panel is read;
2. pairwise categorical interaction: one target and one source category interact;
3. interaction order: one target may interact jointly with several source categories.

A sparse signal can be a sparse hyperedge rather than a sparse pair.

## 2. Product-reference hierarchical decomposition

Fix one whole-neighborhood cavity value `c`, a target `i`, and a selected source set `N`. For every
`u in {i} union N`, let `mu_u(.|c)` be strictly positive. Define the averaging and centering
operators on a score tensor `S(x_i,x_N;c)` by

\[
(E_uS)(x)=\sum_{a\in[q]}\mu_u(a\mid c)S(x_{-u},a;c),
\qquad P_u=I-E_u.
\]

Under the conditional product reference

\[
\mu^E_i(x_i,x_N\mid c)
=\mu_i(x_i\mid c)\prod_{j\in N}\mu_j(x_j\mid c),
\]

the operators are commuting self-adjoint projections in `L2(mu_i^E)`. For every subset
`B subseteq {i} union N`, define

\[
\Pi_B
=\left(\prod_{u\in B}P_u\right)
\left(\prod_{v\notin B}E_v\right).
\]

Then

\[
S=\sum_{B\subseteq\{i\}\cup N}\Pi_BS
\]

is the unique orthogonal functional-ANOVA decomposition. Every nonzero component `Pi_B S` is
centered in each variable in `B` and depends only on those variables.

Terms not containing `i` are independent of the candidate target category and cancel from the
softmax. The target-relevant contrast is therefore

\[
S_i^{\mathrm{rel}}
=\sum_{A\subseteq N}S_{i,A},
\qquad
S_{i,A}=\Pi_{\{i\}\cup A}S.
\]

The hierarchy is:

- `A=empty`: a target-only term;
- `|A|=1`: pairwise target--source fields;
- `|A|>=2`: higher-order target--source interactions.

The dependence branch should exclude the arbitrary target-only component. Define

\[
D_i=\sum_{\emptyset\ne A\subseteq N}S_{i,A},
\qquad
D_i^{(1)}=\sum_{j\in N}S_{i,\{j\}},
\qquad
R_i^{(\ge2)}=D_i-D_i^{(1)}.
\]

Orthogonality gives the exact pair-truncation identity

\[
\|D_i-D_i^{(1)}\|_{L^2(\mu_i^E)}^2
=\sum_{A\subseteq N:\,|A|\ge2}
\|S_{i,A}\|_{L^2(\mu_i^E)}^2.
\]

If a raw target score also contains `S_{i,empty}`, omitting it adds its squared norm to the error.
The architecture deliberately assigns arbitrary one-body terms to the marginal/calibration side,
not to the signed dependence potential.

This theorem uses a product reference. If the source variables are decomposed under their observed
correlated conditional law, the simple `E_u` projections generally do not commute and the
components are not mutually orthogonal. Generalized functional ANOVA and hierarchically
orthogonal decompositions for dependent inputs are relevant prior art. Product-reference and
observed-measure decompositions answer different questions and must not be mixed.

## 3. Pairwise teachers cannot identify pure higher-order dependence

Let `A,B,C in {-1,+1}` and, for finite `eta`, define the strictly positive distribution

\[
p_\eta(a,b,c)=\frac{\exp(\eta abc)}{8\cosh\eta}.
\]

Every one-variable marginal is uniform. Every two-variable marginal is also uniform and
independent because, for example,

\[
\sum_c\exp(\eta abc)=2\cosh\eta
\]

does not depend on `(a,b)`. Thus every pair log-density ratio and every pair categorical field is
zero. Nevertheless,

\[
p_\eta(a\mid b,c)
=\frac{\exp(\eta abc)}{2\cosh\eta}
\]

depends jointly on both sources, and the log joint has the pure triple component `eta abc`.

Therefore pairwise marginals do not identify all sparse dependence, even with infinite data.
An MSA contains full aligned rows and can in principle supply higher-order counts, but a teacher
pipeline that reduces it to PSSMs and pair counts has irreversibly discarded this information.
Finite MSA depth makes direct high-order estimation substantially harder; optional MSA supervision
does not remove that statistical problem.

## 4. Pairwise gauge is not closed under nonlinear composition

Suppose two centered source messages are `m_j=x_j` and `m_k=x_k` for binary centered variables,
and let `P_i u` be a nonzero target-centered contrast. A nonlinear target decoder can produce

\[
(P_i u)(a)(m_j+m_k)^2
=2(P_i u)(a)+2(P_i u)(a)x_jx_k.
\]

After removing the target-only term, a target--source--source interaction remains. Hence a layer
may receive separately centered pair messages and still create higher-order interactions after
aggregation and nonlinearity.

The local pair gauge is therefore not compositionally closed. A deep network can claim a global
pairwise decomposition only if the final target score has the stated additive form, or if the
higher-order residual is explicitly represented and audited. Stacking ordinary nonlinear blocks
after a gauged pair layer weakens the claim to layer-local table separation.

## 5. Base architecture and optional extension

The current strict targetwise operator remains the explicit pairwise truncation

\[
\ell_i^{(1)}(a)
=b_i^E(a)
+\sum_{j\in N_i}G_{i\leftarrow j}(a,x_j;c_i^E),
\qquad R_i^{(\ge2),E}=0.
\]

This is the base trainable architecture, not a claim that the population residual vanishes. A
future optional extension may use

\[
\ell_i(a)
=b_i^E(a)
+\sum_{j\in N_i}G_{i\leftarrow j}(a,x_j;c_i^E)
+R_i^{(\ge2),E}(a,x_{N_i};c_i^E).
\]

where the roles would be:

- `b_i^E`: cavity marginal/background and entropy coordinate;
- `G_i<-j`: first-order source interaction in
  `H_i tensor H_j`, with full categorical capacity on selected arcs;
- `R_i^(>=2),E`: target-containing higher-order residual, orthogonal to target-only and
  all one-source subspaces under the same whole-neighborhood product reference.

Membership in that subspace is not guaranteed by naming an MLP a residual. For a raw tensor-valued
function `F_i(a,x_N;c)`, define the exact projector

\[
\mathcal P_{i,\ge2}
=P_i\left[
I-\prod_{j\in N}E_j
-\sum_{j\in N}P_j\prod_{k\in N\setminus\{j\}}E_k
\right],
\qquad
R_i^{(\ge2),E}=\mathcal P_{i,\ge2}F_i.
\]

The tensor-generating parameters of `F_i` must be measurable with respect to the same
whole-neighborhood cavity and cannot observe the active endpoint categories. The category
arguments enter only when the projected tensor is evaluated. Without this projector or an
equivalent structural parameterization, the residual is only a proposed interface and can absorb
target-only and pairwise terms.

An exact residual projection over a degree-`k` panel requires expectations over the product
reference and is exponential if represented as a dense `q^(k+1)` tensor. Tractable possibilities
are separate research hypotheses:

1. explicit sparse hyperedges of bounded order, such as selected triples;
2. structured tensor networks with an approximation audit;
3. Monte Carlo ANOVA projection during training;
4. no residual branch, accompanied by a measured pair-explained fraction whenever a richer
   teacher is available.

None of these follows from the pair gauge. In particular, fixed low tensor rank should not be
silently reintroduced as the definition of the higher-order branch.

## 6. Probability-exact version

The additive logit hierarchy identifies function subspaces but does not preserve marginals after
normalization. A probability-exact neighborhood model can instead use a fixed-marginal
exponential dependence model

\[
Q_i^E(x_i,x_N\mid c)
\propto
\left[\mu_i(x_i\mid c)\prod_{j\in N}\mu_j(x_j\mid c)\right]
\exp\left\{
\sum_u\alpha_u(x_u;c)+\Phi_i^E(x_i,x_N;c)
\right\},
\]

where the adjusting functions `alpha_u` are chosen so that all one-site marginals remain `mu_u`,
and the dependence potential has the hierarchy

\[
\Phi_i^E
=\sum_jG_{i\leftarrow j}
+R_i^{(\ge2),E}.
\]

Existence and uniqueness under suitable conditions, marginal/dependence orthogonality, and the
adjusting-function construction are already covered by minimum-information dependence models and
multiway log-linear modeling. After gauge fixing, the aggregate one-body adjustment and potential
are unique; individual adjusting functions retain additive-constant freedom. They are determined
calibration terms, not new sparse signals. Computing them over a large neighborhood is a separate
inference problem.

Overlapping targetwise neighborhood models still need not be marginals of one global sequence
distribution. The probability-exact local construction does not solve the global marginal-polytope
consistency problem.

## 7. Supervision consequences

The teacher hierarchy must match the model hierarchy:

- one-site MSA counts supervise `b_i^E` or its marginal reference;
- pair counts supervise `G_i<-j` in the common directional gauge;
- pair counts alone provide no target for `R_i^(>=2),E`;
- joint higher-order MSA statistics, mutation combinations, structure-dependent objectives, or the
  task loss are needed to supervise or audit the residual.

The absence of a higher-order teacher is not evidence that the residual is zero. Conversely, an
unconstrained task residual may absorb one-body entropy unless it is projected or parameterized in
the stated higher-order subspace.

## 8. Prior-art boundary

- Hoeffding decomposition gives the product-reference orthogonal expansion.
- Hierarchical log-linear models give multiway categorical interaction parameters.
- Hooker and Chastaing--Gamboa--Prieur treat functional decompositions with dependent inputs.
- Sei and Yano give fixed-marginal minimum-information dependence models with a unique aggregate
  one-body adjustment and potential after gauge fixing, while individual adjusting functions retain
  additive-constant freedom.
- Potts/DCA models are pairwise log-linear models and do not identify unrestricted higher-order
  interactions from pair marginals.

The possible architectural contribution is not hierarchical ANOVA or multiway log-linear
modeling. The current hypothesis is a single-sequence cavity-computable pairwise truncation with an
explicit audit boundary. A projected higher-order branch is only an optional future extension when
independent supervision and structured computation are available; MSA never becomes an inference
input.
