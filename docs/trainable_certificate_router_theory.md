# Trainable Certificate Routing for a Strict Cavity Mixer

Date: 2026-07-12

Status: theory-first architecture note; no experiments.

## 1. Purpose

The strict targetwise mixer needs a content-dependent source set `N_i(x)` that remains unchanged
when the target category and every selected source category are varied. A decision tree whose
leaf selects only unqueried coordinates is already known to be sufficient. This note closes the
next gap: it specifies a trainable hard-forward policy, proves its pathwise invariance, derives
coverage and expressivity limits, and separates established optimization machinery from the
project-specific certificate constraint.

## 2. Sequential certificate policy

Fix a target `i` and an independent random seed `omega`. At step `t`, the router has queried the
ordered set

\[
Q_t=(a_1,\ldots,a_t)
\]

and observed the transcript

\[
H_t=((a_1,x_{a_1}),\ldots,(a_t,x_{a_t})).
\]

The next hard action is a deterministic function of the transcript and the fixed seed:

\[
A_{t+1}=f_{\theta,t}(i,H_t,\omega).
\]

It either queries a new coordinate outside `{i} union Q_t` or stops. Define `tau` as the number of
query actions, so `A_{tau+1}=STOP`. After that terminal action, the leaf decoder outputs

\[
N_i(x;\omega)=g_\theta(i,H_\tau,\omega)
\subseteq [L]\setminus(\{i\}\cup Q_\tau),
\qquad |N_i|\le k.
\]

The leaf performs no further sequence access. Position identities and relative-position features
are allowed because they do not reveal candidate categories.

The observation `x_a` must be token-local or produced by an independently certified context. A
nonlocal observation at a tree node is structurally admissible only when its complete dependency
set is disjoint from the target and from the union of all coordinates that any descendant leaf may
emit. Reading an ordinary contextual embedding at queried coordinate `a` is not admissible: that
state may already contain `x_i` or a future selected `x_j`, so the apparent query set would not
describe the router's true information set. Checking only the final realized leaf would be
circular rather than a local certificate.

### Theorem 1: pathwise targetwise invariance

For every fixed seed `omega`, the policy above satisfies

\[
N_i(x;\omega)=N
\Longrightarrow
N_i(x';\omega)=N
\]

whenever `x'` differs from `x` only on `{i} union N`.

### Proof

Every queried coordinate lies outside `{i} union N`. Therefore its category is unchanged in
`x'`. Inductively, the two executions have the same transcript at every step. They choose the same
next query, stop at the same time, and emit the same leaf label. The seed is held fixed, so it
cannot break the induction.

The theorem applies to a sampled categorical policy implemented by inverse-CDF sampling or
Gumbel-max, provided the random numbers are independent of the sequence, treated as `omega`, and
ties use a deterministic rule.

## 3. Necessary semantic restrictions

The invariance condition has consequences that are easy to hide in an architecture diagram. The
following statements hold pathwise for every fixed admissible random seed, and therefore also for
the induced membership probabilities.

### Proposition 2: target blindness

For every `x_i,x_i'` with all other coordinates fixed,

\[
N_i(x_i,x_{-i})=N_i(x_i',x_{-i}).
\]

The router cannot use the target category at all. This is natural for masked prediction, but it
is a genuine restriction for an encoder that normally observes every token.

### Proposition 3: self-membership blindness

For any candidate `j`, the membership indicator

\[
z_{ij}(x)=\mathbf 1\{j\in N_i(x)\}
\]

is invariant to changes in `x_j` with all other coordinates fixed.

### Proof

Let `x,x'` differ only at `j`. If `j` belongs to either realized source set, invariance from that
realization allows its category to be changed back to the other input and forces the two complete
sets to agree. If `j` belongs to neither set, both membership indicators are zero.

The complete source set may still change with `x_j` when `j` remains unselected: an unselected
coordinate may act as evidence for selecting other coordinates. It cannot simultaneously use its
own category as evidence for its own membership.

### Corollary 4: endpoint-only support no-go

A strict support indicator cannot selectively open an arc using a nonconstant rule of the two
realized endpoint categories alone:

\[
z_{ij}(x)=h(x_i,x_j).
\]

It may still open that arc independently of the endpoint categories and represent their signed
effect later through `G_{i<-j}(:,x_j)`. What is impossible is using the realized endpoints alone
to decide whether the same channel is opened.

This separates two objects that ordinary attention merges:

- the support certificate says where an interaction channel may be evaluated;
- the categorical value says what the selected source category does to the target distribution.

## 4. Hard-forward training

Let the sampled route have query actions `A_1,...,A_tau` followed by
`A_{tau+1}=STOP`. A generic objective is

\[
J(\theta,\phi)
=\mathbb E_\omega\left[
\mathcal L_{task}(\theta,\phi;N_i)
+\lambda_q\tau+\lambda_e|N_i|
+\lambda_T\mathcal L_{teacher}(N_i)
\right].
\]

`phi` denotes differentiable background and value parameters. The optional teacher term may use
leave-query-out MSA scores as labels or rewards, but the MSA is not a router input and does not
override the sampled forward support.

For a stochastic policy `pi_theta`, write the objective as a cost and include the terminal stop
decision in the score-function gradient:

\[
\nabla_\theta J
=\mathbb E\left[
\sum_{t=0}^{\tau}
(C-b_t(H_t))
\nabla_\theta\log
\pi_\theta(A_{t+1}\mid i,H_t)
\right]
+\text{pathwise differentiable terms},
\]

where `C` is the sampled total cost and each baseline is conditionally independent of the current
sampled action given `H_t`. This is established stochastic-computation-graph and hard-attention
machinery. It gives an unbiased estimator when the stopping time, cost, and score terms satisfy
the usual integrability and interchange conditions, but it may have high variance. A reward
formulation uses the corresponding maximization sign convention.

A hard Gumbel-max forward pass with deterministic tie-breaking also preserves Theorem 1. A straight-through backward pass leaves
the forward semantics intact but is a biased gradient estimator. The distinction must be reported:

- **semantic exactness** is a property of the sampled forward graph;
- **gradient correctness** is a separate optimization question.

### Why an ordinary soft tree is not enough

Suppose a soft relaxation assigns positive mass to several leaves. Let `U(x)` be the union of all
source coordinates emitted by positive-mass leaves, and let `Q` be the true dependency closure of
every gate and representation used to evaluate those leaves. A simple auditable sufficient
condition is:

1. `U(x)` is itself targetwise cavity-invariant;
2. the background deletes all of `{i} union U`;
3. all mixture weights, values, directional references, and gauge-dependent teacher quantities
   are measurable with respect to the one union cavity `C_i^U`;
4. a structural implementation satisfies

\[
Q\cap(\{i\}\cup U)=\varnothing
\]

Standard softmax routing gives every leaf positive mass, so `U` can become large and destroy
sparse execution. If a branch reads a
coordinate selected by another active branch, the structural certificate fails. Exact functional
cancellation could still produce an invariant output, but then invariance must be proved directly
and is not guaranteed by the graph structure.

`Q` is not the list of nominal token indices passed to a router. It is the transitive dependency
closure. An ordinary contextual scout state generally makes `Q` include every upstream token.

Thus soft routing is useful as a surrogate only if the actual training forward remains hard, or
if the union-support condition is enforced explicitly. Annealing a soft model does not make the
earlier training forwards strictly disentangled retroactively.

## 5. Coverage and query lower bounds

Consider a deterministic q-ary decision tree, or one fixed random-seed realization, of depth at
most `B`, where each query reveals one raw q-ary category, with at most `k` selected sources per
leaf.

### Proposition 5: direct source coverage

The union of source identities that target `i` can select over all input sequences has size at
most

\[
kq^B.
\]

Therefore direct coverage of all possible sources requires

\[
kq^B\ge L-1,
\qquad
B\ge\left\lceil\log_q\frac{L-1}{k}\right\rceil.
\]

### Proof

A depth-`B` q-ary tree has at most `q^B` leaves. Every leaf emits at most `k` identities. The union
bound follows by counting.

If the router must be able to realize every size-`k` source set, the stronger necessary condition
is

\[
q^B\ge {L-1\choose k}.
\]

These are necessary counting conditions, not sufficient expressivity results. They concern
directly parameterized source channels, not multi-hop communication.

If an independent discrete seed chooses among `S` different trees, the union bound becomes
`S k q^B`. Randomness can expand coverage across executions, but it does not create
sequence-evidence about an endpoint and may make the realized support stochastic at inference.
Across that finite seed bank, realizing every size-`k` source set requires

\[
S q^B\ge {L-1\choose k}.
\]

For query outcomes with alphabet sizes `m_1,...,m_B`, replace `q^B` by `product_t m_t`. A
continuous-valued certified observation or continuous seed has no finite cross-seed leaf-count
bound without an additional quantization or finite-policy assumption; the fixed-seed theorem
still applies.

For a bank of `H` predeclared source panels of size at most `k`, the corresponding condition is

\[
Hk\ge L-1.
\]

An algorithmic panel generator can represent a very large panel menu without an explicit table,
so it can evade a storage-based `H` limit. It does not remove the per-realization degree `k`, the
fixed-policy leaf count, or the fact that random coverage is not sequence evidence. Claims about
coverage must therefore state whether they are per seed, across a finite seed bank, or over a
continuous randomized generator.

## 6. Hardware-conscious strict instantiation

A fully adaptive next-coordinate policy can be expensive. If every target scores all `L`
coordinates at each of `B` query steps, routing alone costs `O(BL^2)`.

A conservative first architecture is an **oblivious scout certificate router**:

1. Each target and layer has a small position-only scout set `Q_i` of size `B`.
2. The router reads only token-local categorical features `x_{Q_i}` and target/position metadata,
   not contextual scout states that already mixed the sequence.
3. A hard controller chooses one source panel or one algorithmic panel seed.
4. Every possible emitted panel is disjoint from `{i} union Q_i`.
5. The selected set has size at most `k` and is passed to the common target-cavity operator.

With a bounded action alphabet and `O(k)` panel generation, routing costs `O(L(B+k))`. If the
controller scores `H` panels, add `O(LH)`; if it scores every source position, routing is quadratic.
An explicit bank of `H` panels costs `O(LHk)` storage. An oblivious tree is less expressive than
the general adaptive certificate policy, but its access pattern is static enough for batched
kernels.

This is router cost only. With an affine segment tree, the complete layer additionally pays
`O(L C_comp)` to build the tree, `O(L(k+1) log L C_comp)` for target-cavity range products, and
the costs of decoding one background, `k+1` target-relative references, and `k` categorical
fields per target. Write those decoder costs as `C_B`, `C_P`, and `C_G`; the added term is

\[
O\bigl(L(C_B+(k+1)C_P+kC_G)\bigr).
\]

Panel generation, duplicate removal, bounds handling, and disjointness from `Q_i` must also be
`O(k)` for the stated router bound. Subquadratic scaling requires the total per-target controller,
panel, cavity, reference, and value cost to be `o(L)`. Bounded widths and degrees are one
sufficient regime, not a necessary one.

The complete strict layer is then

\[
N_i=R_i(x_{Q_i};\omega),
\qquad
c_i^E=C(x_{-[\{i\}\cup N_i]}),
\]

\[
G_{i\leftarrow j}
=P_{\bar p_{i\mid i}^E}
S_{i\leftarrow j}(c_i^E)
P_{\bar p_{j\mid i}^E}^{\top},
\qquad
\ell_i=b_i^E+\sum_{j\in N_i}G_{i\leftarrow j}(:,x_j).
\]

The hard support is selected first. Background, values, directional references, teacher norm, and
spectrum then use the same whole-neighborhood cavity defined by that realized support.

## 7. Prior-art boundary

### Established ingredients

- Bengio, Leonard, and Courville study stochastic gates, score-function gradients,
  straight-through estimation, and conditional computation.
- Mnih et al. train non-differentiable hard visual attention with reinforcement learning.
- Schulman et al. formalize gradient estimation in stochastic computation graphs.
- Kontschieder et al. use differentiable stochastic routing and mixtures of leaf predictions in
  deep neural decision forests.
- Tanno et al. use stochastic neural-tree routing and single-path conditional inference.
- Jang, Gu, and Poole provide the Gumbel-Softmax relaxation for categorical variables.
- Janisch, Pevny, and Lisy formulate costly feature acquisition as a sequential policy that
  chooses the next feature from previously acquired values.
- Neural feature-selection methods such as L2X, concrete autoencoders, and stochastic gates learn
  discrete or relaxed subsets, but may inspect the full sample, including selected features.

### Not supplied by those papers

The work located in the current, non-exhaustive search does not impose the project-specific leaf certificate

\[
N_i\cap(\{i\}\cup Q_{leaf})=\varnothing
\]

for the purpose of preserving a targetwise categorical ANOVA decomposition. It also does not tie
the hard route to a whole-neighborhood cavity, directional marginal references, and optional-MSA
teacher in a common gauge.

The optimization methods are prior art. The possible contribution is the semantic contract and
its use inside the cavity mixer, subject to a broader search before any novelty claim.

## 8. Research decision

The strict architecture should not use ordinary content top-k. Its theory-first dynamic router is
a hard sampled certificate policy. The first implementable form should use a bounded scout set
and a hard generated source panel. This preserves single-sequence inference and allows optional
MSA supervision without feeding an MSA to the forward router.

The cost is explicit: selected endpoints cannot provide evidence for their own support, hard
optimization may have high variance, and bounded certificates limit direct coverage. A practical
full-context top-k model remains a legitimate weaker baseline, but it must be labeled local gauge.
The certificate solves endpoint access control, not recovery of the true interaction support and
not statistical independence between entropy and interaction strength. The decision target is now
specified as a stable-by-certificate, budgeted teacher-mass envelope; its Bayes limits and leaf-optimal construction
are derived in `docs/support_recoverability_and_envelope_theory.md`.

## 9. Remaining open questions

1. How large is the Bayes gap between full-sequence and endpoint-deleted teacher-support
   prediction under the family distribution?
2. Can an algorithmic panel generator achieve broad coverage without an `O(LHk)` table?
3. What controller class gives low-variance hard training while retaining pathwise invariance?
4. How should multiple layers rotate scout and source roles without losing the per-layer theorem?
5. What approximation gap separates the best certificate router from an unconstrained router or
   an MSA-derived support teacher?

These are mathematical and statistical questions. No router experiment should resume until the
desired approximation target and falsification criterion are stated.

## 10. References

- Bengio, Leonard, and Courville, *Estimating or Propagating Gradients Through Stochastic Neurons
  for Conditional Computation*, 2013, arXiv:1308.3432.
- Mnih et al., *Recurrent Models of Visual Attention*, NeurIPS 2014.
- Schulman et al., *Gradient Estimation Using Stochastic Computation Graphs*, 2015,
  arXiv:1506.05254.
- Kontschieder et al., *Deep Neural Decision Forests*, ICCV 2015,
  DOI `10.1109/ICCV.2015.172`.
- Jang, Gu, and Poole, *Categorical Reparameterization with Gumbel-Softmax*, ICLR 2017.
- Tanno et al., *Adaptive Neural Trees*, ICML 2019.
- Janisch, Pevny, and Lisy, *Classification with Costly Features Using Deep Reinforcement
  Learning*, AAAI 2019, DOI `10.1609/aaai.v33i01.33013959`.
- Chen et al., *Learning to Explain: An Information-Theoretic Perspective on Model
  Interpretation*, ICML 2018.
- Abid, Balin, and Zou, *Concrete Autoencoders: Differentiable Feature Selection and
  Reconstruction*, ICML 2019.
- Yamada et al., *Feature Selection Using Stochastic Gates*, ICML 2020.
