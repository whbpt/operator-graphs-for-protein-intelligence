# Endpoint-Removable Context: Formal Constructions and Limits

Date: 2026-07-12

Status: mathematical design note; no experiments.

## 1. Definition

Let `x=(x_1,...,x_L)` with `x_k in [q]`, and let `A` be a set of positions to hide. An
endpoint-removable context family consists of

\[
T_{-A}(x_{-A})
\]

that is exactly invariant to every `x_a`, `a in A`, while preserving the order of the remaining
tokens. For a strict pair edge, `A={i,j}`. For a shared node cavity on a fixed sparse graph,

\[
A_i=\{i\}\cup N_i.
\]

The invariance requirement is

\[
x_{-A}=x'_{-A}
\quad\Longrightarrow\quad
T_{-A}(x)=T_{-A}(x').
\]

Approximate insensitivity is not sufficient for the exact conditional-ANOVA theorem, although it
may still define a practical weaker model.

## 2. Additive deletion theorem

Let

\[
T(x)=\sum_{k=1}^{L}\psi_k(x_k),
\qquad \psi_k:[q]\to\mathbb R^d.
\]

Then

\[
T_{-A}(x)=T(x)-\sum_{a\in A}\psi_a(x_a)
=\sum_{k\notin A}\psi_k(x_k)
\]

is algebraically endpoint-removable over exact arithmetic.

**Proof.** The final expression contains no term indexed by `A`.

This is the algebra behind Deep Sets, kernel sufficient statistics, and the direct-deletion form
of one-layer linear attention. Exact deletion requires each summand to be token-local or otherwise
free of the hidden endpoints. In floating point, evaluating the full `T` and then subtracting
endpoint terms can leave rounding residuals that depend on the deleted categories. Bitwise or
strict numerical invariance requires an exact accumulator or a range/block query whose arithmetic
never includes the deleted summands.

## 3. Fixed-length universality exists, but can be expensive

For fixed `L` and finite alphabet, choose

\[
\psi_k(a)=e_{(k,a)}\in\mathbb R^{Lq}.
\]

The deleted summary records the category at every retained position and zeros at deleted
positions. It is injective on `[q]^{L-|A|}`. Consequently, for every function

\[
f_A:[q]^{L-|A|}\to\mathbb R^m
\]

there exists a decoder `rho_A` satisfying

\[
f_A(x_{-A})=\rho_A(T_{-A}(x)).
\]

Thus endpoint removal itself does not force information loss. The direct universal construction,
however, uses `O(Lq)` state and an unconstrained decoder that may simply memorize a lookup table.

A scalar real number could also encode a finite sequence using exponentially separated weights.
Under bounded numerical range and a fixed minimum separation or stable-decoding requirement,
that construction needs precision growing linearly with sequence length and becomes exponentially
ill-conditioned. Without such a stability condition, this is an operational warning rather than
an unconditional dimension theorem.

## 4. Finite-precision information lower bound

Fix a known deleted set `A`. Suppose a deterministic zero-error endpoint-removable encoder is
defined on the full Cartesian retained domain and has `d` coordinates represented with at most
`B` bits each. It has at most `2^(Bd)` distinguishable values. Injectivity on all retained
sequences requires

\[
2^{Bd}\ge q^{L-|A|},
\]

and hence

\[
Bd\ge (L-|A|)\log_2 q.
\]

This is only a counting bound, but it prevents claims that a fixed-width, fixed-precision summary
is lossless for arbitrary sequence length.

## 5. Linear-readout lower bound

Consider the vector space of all centered one-site functions on retained positions:

\[
\mathcal V_A
=\bigoplus_{k\notin A}
\{g_k(x_k):\sum_{a=1}^{q}g_k(a)=0\}.
\]

Its dimension is

\[
\dim\mathcal V_A=(L-|A|)(q-1).
\]

Fix one summary map on the full product domain. If every member of `V_A` must be recovered exactly
as a scalar linear readout from its `d` coordinates, then necessarily

\[
d\ge(L-|A|)(q-1).
\]

This follows because scalar linear readouts from `R^d` span at most a `d`-dimensional function
space. Any full-support weighted centering gives the same per-site dimension `q-1`.
Nonlinear decoders can evade this rank bound, but then a robustness, precision, Lipschitz, or
parameter-count condition is needed to prevent arbitrary memorization.

## 6. Why subtracting contextual states fails

Let a contextual network emit `y_k(x_1,...,x_L)` and define

\[
T_{-A}^{\mathrm{naive}}(x)=\sum_{k\notin A}y_k(x).
\]

This is endpoint-removable if and only if, for all `x,x'` agreeing outside `A`,

\[
\sum_{k\notin A}y_k(x)=\sum_{k\notin A}y_k(x').
\]

Ordinary contextual mixing does not satisfy this condition. For a linear layer

\[
y_k=\sum_s W_{ks}u_s,
\]

the coefficient of a hidden source `u_a`, `a in A`, after deleting output rows in `A` is

\[
\sum_{k\notin A}W_{ka},
\]

which is generically nonzero. Subtracting the endpoint output vectors therefore does not remove
their influence after they have propagated into other positions.

Exact deletion after arbitrary contextual layers would require source-resolved states
`y_{k<-s}` and removal of every contribution from sources in `A`. Tracking all sources is
quadratic in sequence length unless additional structure or approximation is imposed.

## 7. Associative affine deletion theorem

Let each token define a token-local affine state transition

\[
F_k(h)=A_k(x_k)h+b_k(x_k).
\]

Represent it by the augmented matrix

\[
M_k(x_k)=
\begin{bmatrix}
A_k(x_k)&b_k(x_k)\\
0&1
\end{bmatrix}.
\]

Composition of affine maps is matrix multiplication and is associative. For a deleted set `A`,
replace every `M_a`, `a in A`, by the identity. The resulting ordered product

Define

\[
M_k^{(A)}=
\begin{cases}
I,&k\in A,\\
M_k(x_k),&k\notin A,
\end{cases}
\qquad
M_{-A}=M_L^{(A)}M_{L-1}^{(A)}\cdots M_1^{(A)}.
\]

is exactly invariant to the deleted token categories and preserves the order of all retained
transitions. Applying `M_-A` to one or more learned initial states gives an endpoint-removable
ordered context.

**Proof.** Every deleted token map is replaced by an identity independent of its category. The
remaining maps appear in their original order, and associativity permits arbitrary parenthesizing.

This is deletion as identity masking on the original indexed sequence. If `M_k` depends on the
fixed position `k`, the construction is not the same as physically shortening and re-indexing all
later tokens. The strict cavity theorem needs category invariance, for which identity masking is
sufficient.

This theorem applies to the affine recurrence core used by parallel-scan SSMs and data-controlled
linear recurrences. It does not imply that every surrounding convolution, gate, normalization, or
contextual parameter generator in a complete SSM block is endpoint-removable.

## 8. Segment-product algorithm

Store interval products in a balanced segment tree. If `m=|A|` positions are removed, the retained
sequence is the ordered composition of at most `m+1` disjoint intervals. A cavity query therefore
costs

\[
O((m+1)\log L\,C_{\circ}),
\]

where `C_comp` is the cost of composing two transition summaries. A balanced tree has `O(L)`
nodes and therefore costs `O(L C_comp)` to construct.

For one pair, `m=2`, so only three retained intervals are composed. For a node cavity on a fixed
degree-`k` sparse graph, `m<=k+1`, giving

\[
O(k\log L\,C_{\circ})
\]

per node. If the transition summaries form a group and stable inverses are available, prefix
products can reduce range queries to constant time. General affine maps can be singular, so a
segment tree is the safer construction.

Diagonal or elementwise state transitions reduce `C_comp`; dense `(d+1)x(d+1)` matrices are
usually too expensive. Forward and reverse products, multiple learned initial states, and
multiscale trees can increase expressiveness while preserving exact deletion.

## 9. Proposed strict sequence operator

Use a fixed sparse candidate graph `E_0`. Fixing the candidate support avoids the self-reference
problem in a data-dependent neighborhood. For each node and edge, query the segment-product data
structure with the corresponding deleted set:

\[
c_i^E=M_{-(\{i\}\cup N_i)}h_0,
\qquad
c_{ij}=M_{-\{i,j\}}h_0.
\]

Then

\[
b_i^E=f_B(c_i^E,i),
\qquad
p_i^E=\operatorname{softmax}b_i^E,
\]

\[
G_{ij}
=P_{p_i^E}f_G(c_{ij},i,j)P_{p_j^E}^{\top},
\qquad (i,j)\in E_0,
\]

and

\[
\ell_i(a)=b_i^E(a)+\sum_{j\in N_i}g_{ij}(c_{ij})G_{ij}(a,x_j).
\]

The gate `g_ij` may depend on the endpoint-excluding `c_ij`, but the cavity background excludes the
entire fixed candidate neighborhood regardless of the gate value. MSA can supervise `p_i^E`,
`G_ij`, and the gate without becoming an inference input.

For proteins, `G_ij` retains the full `19 x 19` tangent-product capacity, with maximum matrix rank
19. The compute approximation is sparse
support and compressed recurrence state, not a universal fixed categorical rank.

## 10. Unresolved tradeoff

The strict model forbids the background from using selected neighbor residues. This is the source
of identifiability, but it may remove useful ordinary predictive context. A large candidate graph
therefore makes the cavity background weak, while a small graph risks omitting true interactions.
The tradeoff cannot be solved by the categorical gauge alone.

The next mathematical question is whether a hierarchy can assign effects without ambiguity:

1. exogenous/local background;
2. fixed sparse candidate-neighbor main effects;
3. gauge-fixed pair interactions;
4. residual higher-order context.

That hierarchy may preserve more predictive information than forcing every selected neighbor to
enter only through a pure pair table, but it requires a new hierarchical-ANOVA specification.

## 11. Compact recurrence capacity is not a fixed intrinsic rank

Exact endpoint deletion and compact expressivity are separate questions. Consider first one
scalar target on a retained ordered sequence of length `n`, represented as the tensor

\[
F(a_1,\ldots,a_n)=f(a_1,\ldots,a_n),
\qquad a_k\in[q].
\]

For a cut after position `t`, reshape `F` into the prefix--suffix matrix

\[
F^{\langle t\rangle}
\in\mathbb R^{q^t\times q^{n-t}}.
\]

If `f` has a matrix-product or tensor-train realization with a linear terminal readout and bond
dimension `D`,

\[
f(a_1,\ldots,a_n)
=\alpha^\top M_n(a_n)\cdots M_1(a_1)\beta,
\]

then

\[
\operatorname{rank}F^{\langle t\rangle}\le D
\]

for every cut. The proof factors the unfolding through the `D`-dimensional linear state at the cut.
For a scalar linear readout of an affine recurrence of state width `d`, the augmented matrices give
`D=d+1`. An arbitrary nonlinear decoder of the final state does not inherit this unfolding-rank
bound; without finite-precision, regularity, or decoder-complexity restrictions, a low-dimensional
real summary can encode a finite domain in an ill-conditioned way. Conversely, for one fixed
finite tensor, its minimal exact *unconstrained tensor-train* bond dimensions are the ranks of
these successive unfoldings. This equality does not characterize the minimum affine state width,
because augmented affine cores have a constrained last row. Thus a universal fixed small linear
state cannot exactly represent targets whose prefix--suffix ranks grow with sequence length.

The same statement gives an approximation lower bound. For fixed `n` and target tensor `F`, let
`sigma_1^(t)>=sigma_2^(t)>=...` be the singular values of `F^<t>`. Every approximation
`F_hat` whose `t`-th unfolding has rank at most `D_t` satisfies, under the unweighted Frobenius
geometry,

\[
\|F-\widehat F\|_F^2
\ge
\sum_{s>D_t}\left(\sigma_s^{(t)}\right)^2
\]

for every cut `t`. The strongest single-cut lower bound is the maximum of these tails. This is a
necessary bound, not a guarantee that independently good truncations at all cuts assemble into a
good shared recurrence. Distribution-weighted claims require specifying the sampling measure;
the displayed result does not automatically transfer to an arbitrary correlated sequence law.

For a homogeneous variable-length linear representation, define the Hankel matrix

\[
H_f(u,v)=f(uv)
\]

over all finite prefixes and suffixes. Classical weighted-automata theory states that finite
Hankel rank equals the minimal linear realization dimension. This is exact for rational series
with shared symbol transitions. It is only a boundary result for our position-dependent affine
maps, finite-length targets, nonlinear decoders, and identity-masked deletion family; those extra
degrees of freedom invalidate a direct equality with one ordinary Hankel rank.

The following criterion applies to nonlinear decoders as well and is therefore more directly
relevant to the proposed architecture. Let `Y` be the full endpoint-deleted context, let
`C=phi(Y)` be the recurrence summary, and let `W in L2` be the teacher quantity that the downstream
branch must predict. For squared loss, define the Bayes risks

\[
R_Y=\inf_g\mathbb E\|W-g(Y)\|^2,
\qquad
R_C=\inf_h\mathbb E\|W-h(C)\|^2.
\]

Because `C` is a function of `Y`, orthogonal projection in `L^2` gives

\[
R_C-R_Y
=\mathbb E\left\|
\mathbb E[W\mid Y]-\mathbb E[W\mid C]
\right\|^2.
\]

Hence `C` is prediction-sufficient for this target under squared loss if and only if the two
conditional means agree almost surely. This is target-specific mean sufficiency, not lossless
encoding and not full statistical sufficiency. Under logarithmic loss for the complete conditional
law of `W`, assuming regular conditional laws and finite Bayes log risks, the corresponding gap is
`I(W;Y|C)`. These identities provide a more relevant criterion for choosing recurrence width than
reconstructing every retained token.

State width also interacts with stability. A perturbation introduced at position `s` propagates as

\[
\delta h_L=A_LA_{L-1}\cdots A_{s+1}\delta h_s.
\]

For token-local affine transitions whose later matrices do not change under the perturbation,
uniform contraction, `||A_k||<=rho<1`, makes its norm at most
`rho^(L-s)||delta h_s||`. This proves exponential sensitivity attenuation. It yields effective
forgetting only with finite precision, state noise, or a uniformly Lipschitz bounded-gain decoder;
an unrestricted decoder in exact arithmetic could amplify an arbitrarily small difference. For a
general nonlinear recurrence, the corresponding local statement uses a Jacobian product. Large
product norms can amplify perturbations and gradients; nonnormal products can be ill-conditioned
even when every eigenvalue appears stable. Orthogonal or otherwise norm-preserving transitions
avoid this simple contraction--expansion dilemma, but they do not remove the linear cut-rank bound.

Finally, within a linear realization class, our architecture uses one collection of token-local
transition cores to answer many node and edge deletion queries, while allowing target-specific
boundary and readout vectors. Low tensor-train rank of every target considered separately does not
upper-bound the minimum simultaneous state dimension: separate factorizations may use incompatible
prefix state spaces. For a finite target family at fixed `L`, a block-direct-sum construction gives
a loose upper bound by adding their widths, but may erase the desired compression. A sharp lower
bound would require target-stacked unfoldings or a simultaneous Hankel construction, which is not
proved here. The relevant object is therefore a simultaneous realization of the complete
cavity-target family, not the maximum rank of one field.

Architectural consequence: use a fixed implementation ceiling `D_max` only to obtain regular
tensor shapes and efficient scan kernels. Do not identify `D_max` with a universal intrinsic rank.
The theory does not require all contexts to have the same approximation rank, but neither does it
prove that adaptive width is superior. Prefix--suffix spectral tails diagnose linear matrix-product
readouts; target-specific Bayes-risk gaps diagnose general summaries and decoders. Conditional
state blocks or width buckets are optional compute strategies and save work only when inactive
blocks are not evaluated and their controllers preserve the same endpoint-removal contract.

## 12. Prior-art boundary

- Deep Sets proves the usefulness and universality of additive set summaries, not endpoint-aware
  ordered categorical interaction.
- Linear Transformers and Performers use additive kernel sufficient statistics, but ordinary
  stacked contextual features do not remain deletion-exact.
- S5, GateLoop, Mamba, and related SSMs exploit scan-compatible recurrences. GateLoop explicitly
  uses data-controlled cumulative products and associative scan.
- Segment trees and range-product queries are classical data structures.
- Sei and Yano's minimum-information dependence model already supplies fixed marginals, a unique
  sum of adjusting functions and potential after gauge fixing, Fisher orthogonality, and an
  entropic-OT interpretation, including the finite categorical/log-linear case. Individual
  adjusting functions retain additive-constant gauge freedom.

The possible contribution is therefore not any individual component. It is the use of
endpoint-removable ordered recurrence products as the context engine for a sparse, common-gauge,
single-sequence categorical interaction operator with optional-MSA supervision. This remains a
hypothesis until a closer deletion-aware sequence-model precedent is ruled out and the hierarchy
in Section 10 is resolved.

## 13. References to inspect

- Zaheer et al., *Deep Sets*, NeurIPS 2017.
- Katharopoulos et al., *Transformers are RNNs*, ICML 2020.
- Choromanski et al., *Rethinking Attention with Performers*, ICLR 2021.
- Smith, Warrington, and Linderman, *Simplified State Space Layers for Sequence Modeling*, 2022/2023.
- Katsch, *GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling*, 2023,
  arXiv:2311.01927.
- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023/2024.
- Sei and Yano, *Minimum Information Dependence Modeling*, Bernoulli 2024,
  DOI 10.3150/23-BEJ1687.
- Oseledets, *Tensor-Train Decomposition*, SIAM Journal on Scientific Computing 2011,
  DOI 10.1137/090752286.
- Balle, Carreras, Luque, and Quattoni, *Spectral Learning of Weighted Automata*, Machine Learning
  2014, DOI 10.1007/s10994-013-5416-x.
- Jaeger, *Observable Operator Models for Discrete Stochastic Time Series*, Neural Computation
  2000, DOI 10.1162/089976600300015411.
- Khrulkov, Novikov, and Oseledets, *Expressive Power of Recurrent Neural Networks*,
  arXiv:1711.00811.
- Li, Precup, and Rabusseau, *Connecting Weighted Automata, Tensor Networks and Recurrent Neural
  Networks through Spectral Learning*, arXiv:2010.10029.
