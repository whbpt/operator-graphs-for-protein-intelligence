# Single-Sequence Support Recoverability and Budgeted Teacher-Mass Envelopes

Date: 2026-07-12

Status: theory-first architecture note; no experiments.

## 1. The inference problem

An optional MSA can provide training labels for a protein family, but inference receives one
sequence. Let

- `F` be a latent family or generative state;
- `X in [q]^L` be the observed single sequence;
- `T_ij in {0,1}` be a binary teacher support label, or `W_ij >= 0` a teacher interaction weight;
- `T_i=(T_ij)_{j != i}` be the complete teacher support object for target `i`.

All inference-time router seeds are assumed independent of `(F,X,T,W)`. Independence from `X`
alone is insufficient for a teacher-relative Bayes lower bound because an externally correlated
seed could carry teacher or latent-family information.

The teacher may be derived from an MSA, structure, mutation data, or another training-only source.
It is not assumed to be causal or uniquely correct. The question is decision-theoretic:

> What part of a training teacher can any single-sequence, endpoint-blind router predict at
> inference, and what support object should the architecture ask it to predict?

## 2. Edgewise measurability forced by strict support

Assume the strict router condition holds on the full q-ary product domain. For candidate `j`,
define the endpoint-deleted observation

\[
C_{ij}=X_{-[\{i,j\}]}.
\]

### Theorem 1: strict membership is leave-two-out measurable

For every deterministic strict targetwise router,

\[
Z_{ij}(X)=\mathbf1\{j\in N_i(X)\}
\]

is a function of `C_ij` alone. For a pathwise randomized router, this holds for every fixed
admissible seed, so

\[
Z_{ij}=r_{ij}(C_{ij},\omega),
\qquad \omega\perp(F,X,T,W).
\]

### Proof

Targetwise invariance makes the entire `N_i` independent of `x_i`. Self-membership blindness
makes `Z_ij` independent of `x_j` while every other coordinate is fixed. Therefore `Z_ij` is
constant on every fiber with fixed `X_-{i,j}` and is measurable with respect to `C_ij`.

This is only an edgewise necessary property. Arbitrary collections of leave-two-out membership
functions need not satisfy the stronger joint source-set invariance required by a certificate
router.

If strictness is imposed only on the support of a data distribution, the statement is correspondingly
almost-sure on supported fibers and cannot be extended silently to unseen category combinations.

## 3. Bayes error floor for a strict router

For a binary teacher label, define

\[
\eta_{ij}(c)=\Pr(T_{ij}=1\mid C_{ij}=c).
\]

The best possible endpoint-blind membership error is

\[
e_{ij}^{\mathrm{cav}}
=\mathbb E\left[
\min\{\eta_{ij}(C_{ij}),1-\eta_{ij}(C_{ij})\}
\right].
\label{eq:cavity-bayes-error}
\]

### Theorem 2: certificate lower bound

Every strict deterministic or independently randomized router satisfies

\[
\Pr(Z_{ij}\ne T_{ij})\ge e_{ij}^{\mathrm{cav}}.
\]

For Hamming support loss,

\[
\mathbb E\sum_{j\ne i}\mathbf1\{Z_{ij}\ne T_{ij}\}
\ge
\sum_{j\ne i}e_{ij}^{\mathrm{cav}}.
\]

The inequality may be strict because joint certificate constraints and degree budgets further
restrict the membership functions.

### Proof

Conditional on `C_ij=c`, every admissible predictor chooses a label without observing the two
endpoints. The minimum conditional 0-1 risk is the smaller posterior mass. Independent
randomization cannot improve a linear conditional risk. Sum the edgewise inequalities for
Hamming loss.

An unconstrained single-sequence predictor has the smaller Bayes error

\[
e_{ij}^{\mathrm{full}}
=\mathbb E\min\left\{
\Pr(T_{ij}=1\mid X),
\Pr(T_{ij}=0\mid X)
\right\}
\le e_{ij}^{\mathrm{cav}}.
\]

The difference `e_cav - e_full` is the teacher-relative edgewise Bayes-risk penalty attributable
to endpoint deletion under binary membership loss when the entire complement is available. It is
only part of the actual strict-router gap, because
joint certificate, transcript, and degree restrictions may add error. It is not the price of
losing endpoint information from the categorical value: after an arc is opened, the value branch
still uses the final source lookup `G_{i<-j}(:,x_j)`.

`C_ij` gives the router the entire endpoint-deleted sequence and is therefore an optimistic
information benchmark. A bounded scout transcript is a function of less information and may have
a larger Bayes error by data processing.

## 4. Conditional information and exact recovery

The endpoint label information unavailable even to the full complement benchmark is

\[
I(T_{ij};X_i,X_j\mid C_{ij})
=H(T_{ij}\mid C_{ij})-H(T_{ij}\mid X).
\]

This is an information quantity, not a causal or structural score. If it is zero, the realized
endpoints contain no additional teacher-label information after the complement is known. If it is
positive, some support-label information is unavailable to a strict router, although the exact
increase in 0-1 Bayes risk depends on the posterior geometry.

### Corollary 3: zero-error condition

A binary strict membership predictor can have zero error only if

\[
\eta_{ij}(C_{ij})\in\{0,1\}
\quad\text{almost surely}.
\]

In the latent-family model, if

\[
0<\Pr(T_{ij}=1\mid C_{ij})<1
\]

on a set of endpoint-deleted contexts with positive probability, exact single-sequence strict
recovery is impossible on that set. MSA supervision can help learn the posterior across training
families but cannot remove its irreducible uncertainty at inference.

For a finite support label `T_i` with `M` possible values and any transcript/seed observation `Y`
available to a certificate router, Fano's inequality gives

\[
H(T_i\mid Y)
\le h_2(P_e)+P_e\log_2(M-1),
\]

where `P_e=Pr(\widehat T_i != T_i)`. If the label alphabet includes every exact size-`k` support, its
maximal size is

\[
M={L-1\choose k};
\]

otherwise `M` is the number of support sets with positive probability. A large ambient alphabet
alone does not make recovery difficult: the operative quantity is `H(T_i|Y)`. The inequality is a
classical error bound, not a claim that mutual information alone determines the optimal router.

## 5. Why the router should predict an envelope

Exact support prediction is stronger than the architecture needs. A false negative removes a
potential interaction channel, while a false positive spends computation but can still decode a
zero or weak tangent-space field. This asymmetry suggests a
**budgeted posterior teacher-mass envelope**:

> The certificate router should select a bounded candidate set that maximizes posterior expected
> captured teacher importance from third-party evidence; the common-cavity categorical value
> decides the signed effect inside that envelope.

Let `W_ij >= 0` be an integrable training-only importance weight. It may be a structural contact
weight, a mutation-derived score, or a cautiously defined MSA teacher strength. It is not assumed
to be ground truth. Given a certificate transcript `H` and queried set `Q`, define

\[
m_j(H)=\mathbb E[W_{ij}\mid H].
\]

An admissible envelope is

\[
N\subseteq[L]\setminus(\{i\}\cup Q),
\qquad |N|\le k.
\]

Use missed teacher mass plus a per-edge compute cost:

\[
\mathcal L_{env}(N,W_i)
=\sum_{j\ne i,\,j\notin N}W_{ij}+\lambda|N|.
\]

### Theorem 4: Bayes-optimal leaf envelope

Conditional on transcript `H`, an optimal leaf selects up to `k` unqueried positions with the
largest positive values of

\[
m_j(H)-\lambda.
\]

For a fixed size `k`, it selects the top-`k` posterior expected weights among unqueried positions.

### Proof

The conditional expected loss is

\[
\mathbb E[\mathcal L_{env}\mid H]
=\sum_{j\ne i}m_j(H)-\sum_{j\in N}(m_j(H)-\lambda).
\]

The first term is independent of `N`, so the constrained optimum keeps the largest positive
reductions.

This result does not say that independent per-edge top-k scores form a strict router. It applies
inside a realized certificate leaf, after the transcript and unqueried candidate set are fixed.

## 6. Bayes-optimal scout acquisition

The scout policy has an opportunity cost: once coordinate `a` is queried, it can no longer be an
output source on that path. Define

\[
A(H,Q)=[L]\setminus(\{i\}\cup Q).
\]

If `m_(1)>=...` are the posterior weights sorted only over `A(H,Q)`, the stop value is

\[
V_{stop}(H,Q)
=\sum_{j\ne i}m_j(H)
-\sum_{r=1}^{k}[m_{(r)}(H)-\lambda]_+,
\]

with missing ranks omitted. Queried positions remain in the first sum but cannot enter the second,
which is exactly their lost-source opportunity cost.

For a fixed budget, let `V_b(H,Q)` be the optimal expected future loss with `b` queries remaining.
Then

\[
V_0(H,Q)=V_{stop}(H,Q),
\]

and for `b>0`,

\[
V_b(H,Q)
=\min\left\{
V_{stop}(H,Q),
\min_{a\in A(H,Q)}
\left[c_a+
\mathbb E\bigl[
V_{b-1}(H\cup\{(a,X_a)\},Q\cup\{a\})
\mid H
\bigr]
\right]
\right\}.
\]

Acquisition costs may be retained or set to zero depending on the budget convention.

Dynamic feature selection often chooses the next feature by conditional mutual information, such
as

\[
I(T_i;X_a\mid H).
\]

That is a useful information-gain heuristic and has established amortized-learning precedents. It
is not the exact objective because mutual information is generally not equivalent to reduction in
the envelope Bayes loss, and querying `a` additionally forbids selecting `a` as a source. The
Bellman recursion uses the actual loss and includes this source-opportunity cost.

## 7. Optional MSA supervision

During training, a leave-query-out MSA can provide targets `W_i` or a policy reward. Across
training families, the student estimates posterior leaf scores. The MSA must not be a router input
at inference, and it should not override the hard sampled support during training. The student
learns an amortized posterior decision rule:

\[
X\longmapsto \Pr(W_i\mid H(X)).
\]

If the teacher remains ambiguous given the single-sequence transcript, the correct target is a
posterior expected-importance decision, not memorization of one teacher graph. Calibration and
set recall are separate properties that Theorem 4 does not establish.

Route and value teachers must also remain distinct:

- a route weight `W_ij` ranks the cost of omitting an arc and may come from structure or a
  training-only heuristic;
- once the arc is selected, the categorical teacher field is reprojected using the targetwise
  directional references from the realized whole-neighborhood cavity.

Calling an MSA score a support ground truth would exceed the mathematics.
If `W_ij` is derived from an MSA categorical field, its gauge, marginal normalization, and geometry
must be stated. Otherwise it remains an entropy-contaminated routing heuristic rather than a
separated interaction target.

## 8. Architecture consequence

The strict layer now has a decision-theoretic interpretation:

1. token-local scouts produce a hard certificate transcript;
2. the leaf emits a budgeted teacher-mass envelope using posterior expected importance;
3. the target and complete envelope are deleted from the common cavity;
4. the background predicts marginal entropy from that cavity;
   more precisely, it predicts the cavity marginal distribution and its entropy, not an additive
   decomposition of the final posterior entropy;
5. full tangent-product fields are decoded for envelope arcs;
6. selected source categories enter only through final column lookup.

Failure to recover the exact teacher support does not invalidate the function-space separation.
It changes efficiency and omitted-interaction error. The model should therefore report three
separate quantities in any future experiment:

- captured teacher-mass fraction and set recall under a fixed compute budget, reported separately;
- categorical field approximation inside the envelope;
- downstream loss relative to the cavity background.

## 9. Prior-art boundary

Established ingredients include:

- Bayes classification risk and Fano-type entropy/error bounds;
- information-theoretic feature selection and conditional mutual information;
- active or costly feature acquisition with sequential policies;
- dynamic feature selection by amortized conditional mutual information;
- decision-tree posterior leaf decisions.

The project-specific combination is narrower: estimate a budgeted teacher-mass set of **unqueried**
interaction endpoints, then use that set to define a whole-neighborhood categorical cavity. The
current literature search is not exhaustive, so this remains a candidate interface rather than a
novelty claim.

Relevant references:

- Cover and Thomas, *Elements of Information Theory*, second edition, 2006.
- Feder and Merhav, *Relations Between Entropy and Error Probability*, IEEE Transactions on
  Information Theory 1994, DOI `10.1109/18.272494`.
- Brown et al., *Conditional Likelihood Maximisation: A Unifying Framework for Information
  Theoretic Feature Selection*, JMLR 2012.
- Janisch, Pevny, and Lisy, *Classification with Costly Features Using Deep Reinforcement
  Learning*, AAAI 2019.
- Covert et al., *Learning to Maximize Mutual Information for Dynamic Feature Selection*, 2023,
  arXiv:2301.00557.

## 10. Research decision

The architecture should no longer describe dynamic support as recovery of a latent contact graph.
Its strict object is a stable-by-certificate, budgeted teacher-mass envelope predicted from
third-party single-sequence evidence. The MSA is an optional teacher for posterior decision-making, not an
inference input and not a source of certainty.

The next mathematical question is the approximation class of the endpoint-removable context:
even when `W_i` is recoverable from the complement in principle, a compact affine recurrence may
not retain the sufficient statistic needed to approximate `m_j(H)` or the categorical field.
