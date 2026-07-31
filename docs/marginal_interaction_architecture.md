# Marginal-Interaction Architecture Specification

> **Status, 2026-07-12:** The empirical sections below are a historical research log. Earlier
> target-alias and teacher/student-gauge bugs make their architecture conclusions provisional.
> The authoritative design is now `paper/theory_main.tex`: full `19 x 19` tangent-product
> capacity is the protein default, pair-support sparsity is separate, and strict targetwise
> functional-ANOVA claims require one common whole-neighborhood cavity.

## Execution contract

- Required inference input: one protein sequence.
- Optional training input: MSA-derived PSSM and connected interaction teacher.
- Optional inference enhancement: an MSA adapter may add a family-specific residual.
- The base model must execute without an MSA.

## Current architecture decision

The content-tile router and leave-query-out MSA teacher remain candidate components, not settled
results. The current theory-first choice is an explicit sparse directed relation-state stream with
a full `(q-1) x (q-1)` tangent-product decoder on selected arcs. Fixed rank 8 is retained only as
a historical experimental budget and must not define the architecture.

The deployable local-gauge variant initializes and updates

\[
r_{ij}^{0}=f_{\mathrm{init}}(V_i,V_j,Q_i,Q_j,\rho_{ij}),
\qquad
m_i=\sum_{j:(i,j)\in E}\phi(V_j,r_{ij}),
\]

\[
r_{ij}^{l+1}=r_{ij}^{l}
+F_r(r_{ij}^{l},V_i,V_j,m_i,m_j),
\]

where `rho_ij` is a relative-position feature. Because ordinary `V_i,V_j,m_i,m_j` contain the
endpoint residues, this model has only a local table gauge. The categorical field is decoded from
the relation state and projected only after assembly:

\[
G_{ij}=\mathcal P_{p_i,p_j}[f_G(r_{ij})].
\]

This path is evaluated only on selected edges, so a single update costs `O(L k d_r)` rather
than `O(L^2)`. The MSA supervises `G_ij` during training but is not an inference input.

The strict cavity variant is different. It uses a targetwise certificate router and one context
that removes the target plus its complete selected source set. Background, every relation value,
and both target-relative references use this same cavity:

\[
\mathcal C_i^E=\sigma\{X_k:k\notin\{i\}\cup N_i\},
\qquad
\bar p_{u\mid i}^E
=\operatorname{stopgrad}\operatorname{EMA}
\bigl(p_{u\mid i}^\theta(\cdot\mid\mathcal C_i^E)\bigr),
\quad u\in\{i\}\cup N_i,
\]

\[
r_{i\leftarrow j}'=F_r(r_{i\leftarrow j},c_i^E),
\qquad
G_{i\leftarrow j}=P_{\bar p_{i\mid i}^E}f_G(r_{i\leftarrow j}')
P_{\bar p_{j\mid i}^E}^\top.
\]

Only the final lookup observes each selected `x_j`. This form satisfies the strict targetwise
conditional-ANOVA theorem relative to the model-defined product reference. A pair-only cavity
with pair-specific references supports only a weaker per-arc conditional gauge. The efficient
context construction is developed in `docs/endpoint_excluding_context_architecture.md`.

## Statistical decomposition

The marginal branch predicts `p_i(a)` and entropy `H_i`. The interaction branch predicts
`G_ij(a,b)` under the weighted gauge

\[
p_i^T G_{ij}=0,
\qquad
G_{ij}p_j=0.
\]

This is the hard separation criterion. Scalar interaction strength is allowed to correlate with
entropy because conserved sites can carry genuine structural constraints.

## Factorized interaction

For rank `R`, centered and per-site normalized categorical factors are combined as

\[
G_{ij}(a,b)=\alpha_{ij}\widehat G_{ij}(a,b).
\]

The shape is a two-sided projected, normalized low-rank field:

\[
\widehat G_{ij}
=\mathcal N\left[(I-\mathbf 1p_i^T)
\left(\sum_r\kappa_{ij,r}U_i(:,r)U_j(:,r)^T\right)
(I-p_j\mathbf 1^T)\right],
\]

where the right multiplication above is understood as applying the second projector to the
unprojected field. The scalar `alpha_ij >= 0` carries interaction strength; `Ghat_ij` carries
only signed categorical geometry. Candidate selection is a separate compute decision and must
not be conflated with either value amplitude or field shape.

The separation is required because physical epistasis RMS varies by more than two orders of
magnitude across held-out families, while family-normalized field shape is measurably more
transferable. Family target normalization is only a diagnostic during development. A static
strength MLP on frozen residue states has now failed held-out-family tests, so the deployable
strength gate must be conditioned on the current layer/task state and trained jointly or by
dense-teacher distillation.

## Full backbone proposal

Selected layers maintain two representation streams and three functional paths:

- `stable state V`: slow/content representation used for marginals and categorical values;
- `task state Q`: current layer/task representation used only for dynamic routing keys.

The paths are:

1. `background/local`: local mixer or static memory for motifs and marginal statistics;
2. `strength/index`: a narrow dynamic mixer predicts candidate rank from current task state;
3. `typed shape`: a symmetric two-sided-zero-sum categorical field evaluated only on candidates.

Candidate generation should not assume that interacting residues are contiguous in primary-
sequence coordinates. The current systems proposal assigns stable keys to learned, capacity-
controlled interaction tiles, routes each task query through a shallow tile hierarchy, and exact-
scores only the bounded candidate set returned by the selected leaves. The tile hierarchy is
trained on conditional-response mass, not on raw attention weights. Sequence-local windows
remain in the background path and are not reused as the nonlocal interaction partition.

The routing score and value are deliberately computed from different states:

\[
s_{ij}^{l}=f_{\mathrm{index}}(Q_i^l,Q_j^l),
\qquad
\widehat G_{ij}^{l}=f_{\mathrm{value}}(V_i^l,V_j^l;p_i,p_j).
\]

The layer update is

\[
V^{l+1}=V^l+F_{bg}^l(V^l)+\Pi_V I^l,
\qquad
Q^{l+1}=F_{task}^l(Q^l,V^l)+\Pi_Q I^l.
\]

Here `I` evaluates `Ghat_ij` only on the top-k pairs selected by `s_ij`. The interaction scale
should be initialized small. Bottleneck outputs require RMS normalization and explicit scale
control.

## Training sequence

1. Freeze a single-sequence backbone and train the marginal/interaction head.
2. Train normalized interaction shape on explicitly non-additive double-mutation effects.
3. Warm up a dynamic strength/index gate against dense non-additive task effects. Do not use a
   frozen-state static pair MLP as the final router.
4. Convert the dense gate to a fixed top-k budget and jointly adapt it with the task model; do
   not claim DSA-style selection itself as a novel component.
5. Compare against dense MLA-style, scalar sparse, and whitened relation baselines on held-out
   protein families.
6. Only after family-split success, insert interaction blocks into a newly trained backbone.

## Current evidence

On 3CNBA with frozen ESM-2 8M, rank 32 reaches score-map Spearman `0.895` and long-range
contact `P@L=0.792`, versus teacher `0.800`. Approximately `7%` of pair gates exceed `0.5`.
This is a single-family feasibility result, not evidence of generalization.

In the explicit epistasis experiment, family-normalized shape is trained on 64 families and
evaluated on 24 unseen families in two seeds. The unprojected pair model explains `9.14%` and
`6.26%` of held-out shape variance; the projected pair explains `6.16%` and `4.17%`; the
site-only control explains none. The projected-pair improvement over site-only is `0.0528`
MSE with family-bootstrap 95% CI `[0.0168, 0.0875]`. Its weighted gauge error is about
`5e-8`. Absolute amplitude prediction, sparse routing, and an efficient end-to-end backbone
remain unverified.

For strength prediction, we decompose log epistasis RMS into a family location, an
entropy-derived background, and a residual pair score. Using all 245 training families and 57
held-out families, neither a direct pair MLP nor an entropy-plus-pair residual model improves
reliably over entropy-only across two seeds. Direct-pair top-10% AP improvement is `0.0122`
with 95% CI `[-0.0233, 0.0492]`, while its MSE is significantly worse. This rejects the static
strength gate currently tested.

Replacing the static ESM input with hidden states from the current masked-task forward pass
changes the result decisively. Across two seeds, the dynamic gate reaches held-out-family
Spearman `0.725/0.700` and top-10% AP `0.651/0.644`, versus `0.034/-0.005` and
`0.199/0.205` for the static pair gate. Dynamic-over-static paired improvements are `0.698`
for Spearman (95% CI `[0.655, 0.738]`) and `0.445` for AP (`[0.402, 0.491]`).

The same task-conditioned state does not reliably improve projected categorical shape over the
stable ESM state: MSE improvement is `-0.0083` with CI `[-0.0599, 0.0430]`. This is the
empirical basis for separating dynamic routing keys from stable typed values.

## End-to-end dual-stream demo

A runnable single-sequence model now replaces the dense task Transformer with a bidirectional
GRU, while the stable stream uses a local convolution initialized from the local MLM checkpoint.
After 300 sampled-dense teacher updates and 300 top-k updates, the two seeds reach index
Spearman `0.595/0.628`, top-10% AP `0.579/0.712`, and shape correlation `0.164/0.285` on
24 held-out families. Thus the dual streams and top-k conversion are trainable without a dense
Transformer at execution.

The final categorical message does not yet improve MLM reproducibly. Interaction-minus-
background CE improvements are `+0.00218` and `-0.00252`; the two-seed family-bootstrap mean
is `-0.00017` with 95% CI `[-0.00471, 0.00459]`.

The diagnosed issue is semantic, not routing optimization. The current double-mutation target
describes two context mutations acting jointly on a set of masked probes. It is not the direct
message field used by the layer. The next value teacher must be aligned to execution:

\[
R_{ij}(a,b)
=\ell_i(a\mid x_j=b,x_i=\mathrm{mask})
-\mathbb E_{b'\sim p_j}\ell_i(a\mid x_j=b',x_i=\mathrm{mask}),
\]

followed by the two-sided weighted projection. Then `R_ij(a, x_j)` is exactly a teacher for the
message added to site `i`, unlike the current context-context epistasis block.

## Conditional-response replication

The execution-aligned teacher was tested in two formal seeds. The background branch was frozen;
each masked target enumerated all 20 candidate residues at sampled context sites, and the model
was trained for 300 dense warm-up steps followed by 300 top-k adaptation steps.

After top-k adaptation, the two seeds reach index Spearman `0.688/0.729`, index AP
`0.657/0.709`, and projected-shape correlation `0.423/0.392`. Relative to the frozen background,
teacher KL improves in both runs. A seed-family-example hierarchical bootstrap gives a mean KL
gain of `0.000384` with 95% CI `[0.000058, 0.000850]`.

True-residue cross-entropy also improves in both point estimates, by `0.000769` and `0.002247`,
but the hierarchical mean of `0.001508` has 95% CI `[-0.000388, 0.003887]`. The current result
therefore establishes reproducible teacher-message reconstruction, not yet a statistically
resolved improvement in the biological prediction target.

## Rank policy after the DeepSeek comparison

DeepSeek MLA validates a fixed-width latent as an efficient implementation interface, but it
does not imply a fixed intrinsic rank for every token or pair. We therefore tested a fixed
maximum rank with learned pair-specific mode gates:

\[
G_{ij}=\mathcal P_{p_i,p_j}
\left[\sum_{r=1}^{R_{\max}}g_{ijr}u_{ir}v_{jr}^{T}\right],
\qquad g_{ijr}\ge 0.
\]

In that experimental variant, effective rank is measured after projection, the dynamic task
stream predicts routing and mode gates, and the stable stream predicts categorical factors. The
hard gauge is applied to the assembled field. The historical ablation below rejected this gate
as a compute mechanism, but it does not justify fixed rank 8 as the current default. MLA-style latent caching,
DSA-style dense-to-sparse training, and top-k selection are prior art and should be treated as
engineering components rather than the novelty claim.

## Adaptive-rank ablation

The maximum-rank policy was tested rather than assumed. Four variants used the same conditional-
response data, two seeds, 300 dense warm-up steps, and 300 sparse adaptation steps: fixed rank-8,
an unregularized soft dynamic gate, explicitly rank-penalized gates, and true fixed rank-5.

The unregularized gate settles at effective rank `5.13/5.06`. Relative to fixed rank-8, its
shape-correlation change is `+0.00539` with paired 95% CI `[0.00129, 0.00986]`, while shape-MSE
improvement is `-0.00568` with CI `[-0.02550, 0.01314]`. CE-gain change is unresolved at
`-0.000117` with CI `[-0.000581, 0.000273]`; teacher-KL gain is slightly worse by `0.0000614`
with CI `[-0.000165, -0.0000027]`.

This is not evidence for pair-adaptive compute. Across all 1,536 validation pairs per seed, gate
vectors have mean cosine similarity `0.993/0.990` to a seed-specific global template. The
5th--95th percentile effective-rank range is only about `4.35--6.16`, and top-gate identity is
dominated by one or two modes. Explicit penalties drive effective rank near 3 but significantly
reduce shape correlation and worsen shape MSE.

A true fixed rank-5 model is also not a reliable replacement. Its paired shape-correlation change
versus rank-8 is `-0.0522` with CI `[-0.1169, 0.00547]`, with strong seed heterogeneity. Because
soft gates still evaluate all eight modes, they provide no measured runtime reduction. The
historical prototype therefore retained fixed rank 8. The theory-first architecture now uses the
full `19 x 19` tangent-product capacity, with maximum matrix rank 19, on selected protein arcs.
Mode pruning should return only if a hard selector can
skip modes and preserve held-out shape under a matched-compute benchmark.

## Non-quadratic candidate routing ablation

The current top-k layer still materializes all pair scores, so we tested whether its symmetric
bilinear index could be served by subquadratic locality-sensitive hashing. The score has an exact
maximum-inner-product representation:

\[
s_{ij}\propto [r_i,l_i]^T[l_j,r_j].
\]

A multi-table random-hyperplane router hashes these query/key features, probes exact and
Hamming-distance-one buckets, and computes the original index score only on retrieved candidates.
Across two model seeds, 96 validation targets per seed, and three hash seeds, the most faithful
post-hoc configuration evaluates `70.7%` of valid pairs and recovers `97.7%` of dense top-8
neighbors. Its message cosine is `0.985`, and its CE-gain difference versus dense is `-0.000076`
with family-clustered 95% CI `[-0.000763, 0.000523]`. This saves only `29.3%` of pair evaluations,
before accounting for hash projections and bucket operations.

At approximately half of the pair evaluations, neighbor recall falls to `91.2%`, message cosine
to `0.926`, and the mean CE-gain difference to `-0.000847`; family-level tails are substantially
worse. More aggressive 32--35% pair budgets produce unstable or sign-flipped messages.

We also trained a 2,048-parameter shared hash adapter using dense-index distillation, conditional-
response strength, straight-through binary codes, quantization, bit balance, and decorrelation.
On a full 256-target training run it reaches only `82.4%` top-8 recall at `50.3%` pair evaluation,
message cosine `0.860`, and CE-gain difference `-0.00195`. This is worse than the post-hoc router,
so the learned-hash path was stopped after its first formal seed.

LSH is therefore rejected as the default candidate generator. The implementation remains a
diagnostic, but the executable model continues to use dense score construction. The next
non-quadratic candidate should expose a supervised hierarchy whose coarse decisions can be
trained directly from conditional-response mass, rather than relying on accidental hash
collisions.

## Hierarchical routing and sequence-block falsification

A learned contiguous-segment hierarchy was trained first because it provides a simple
hardware-friendly tree. With beam 16 and 64 exact candidates, joint sparse adaptation evaluates
an estimated `62.3%` of dense vector work. Across two seeds it recovers `92.5%` of dense top-8
neighbors (95% CI `[89.6%, 95.1%]`) and reaches message cosine `0.914`
(`[0.834, 0.978]`). Routed teacher-KL gain is positive in both seeds, with hierarchical mean
`0.001349` and CI `[0.000433, 0.002488]`.

This does not establish a task-level win. Routed true-residue CE gain is `+0.01338` in seed 12
but `-0.000649` in seed 13. The hierarchical mean is `0.00637` with CI
`[-0.00515, 0.02334]`. Routed-minus-adapted-dense CE is `0.000222` with CI
`[-0.00180, 0.00279]`. The hierarchy is therefore a useful routing prototype, not a validated
Transformer replacement.

More importantly, its contiguous-segment assumption was tested directly. On 24 held-out
families and 96 masked targets, we enumerated all nonlocal context positions and all 20 amino-
acid substitutions, projected each conditional-response block into the weighted gauge, and
measured positive interaction energy above the per-target median. At a 32-position budget, an
oracle allowed to place four arbitrary non-overlapping windows of width 8 captures only `67.4%`
of the mass captured by 32 oracle individual positions (CI `[63.7%, 71.3%]`). Its captured mass
is only `1.022` times a position-permuted null (`[1.003, 1.043]`). Wider windows are worse.

The signal is sparse but not meaningfully contiguous along primary sequence. This rejects an
NSA-style sequence-block selector as the default protein interaction router and explains why
the current segment hierarchy requires a high beam. The next candidate generator should use
learned interaction tiles:

1. assign stable keys to balanced content tiles using a routing-only control bias;
2. train tile mass from the sum of gauge-projected conditional-response strength in each tile;
3. route task queries through a content hierarchy and exact-score a bounded leaf candidate set;
4. retain dense routing as the teacher/reference path during warm-up;
5. benchmark actual gather patterns and wall-clock time, not only pair counts.

This design borrows load control from MoE and native sparse execution from NSA, but the tile
membership and supervision are specific to marginal-orthogonal interaction structure.

## Implemented content-tile execution

The first learned-tile prototype is now executable as `ContentTileProteinLM`. It accepts only a
single token sequence and owns both the dual-stream backbone and the tile router. No MSA,
teacher model, or dense attention matrix is required by its forward pass.

For stable state `V_j`, dynamic task state `Q_i`, and learned tile prototypes `c_m`, define

\[
a_{jm}=\langle W_V V_j,c_m\rangle,
\qquad
u_{im}=\langle W_Q Q_i,c_m\rangle.
\]

During training, Sinkhorn normalization turns `a_jm` into an approximately balanced soft
assignment `A_jm`. The router predicts a partner distribution through a tile mixture,

\[
\widehat P_i(j)
=\sum_m \operatorname{softmax}_m(u_i)
\frac{A_{jm}}{\sum_{j'}A_{j'm}},
\]

and is trained by KL divergence to both the dense index distribution and the conditional-
response distribution. During execution, a deterministic capacity-constrained assignment puts
at most `ceil(L/M)` tokens in each tile. Queries rank tiles, accumulate a fixed candidate budget,
compute the original exact index only on those candidates, and evaluate the categorical value
only for the final top-k neighbors.

The implemented flat router costs approximately

\[
O(LMd_{tile}+Lk d_{index}),
\]

where tile assignment and query-to-tile scoring both contribute to the first term. This is not
yet an asymptotic `O(L log L)` claim: if `M` must grow with sequence length to keep tile capacity
bounded, flat query-to-tile scoring also grows. A tree over learned tiles remains the long-
sequence extension. The current result validates the content partition and bounded candidate
operator, not the final kernel complexity.

### Router-only result

With 12 tiles and a 32-candidate budget, two held-out-validation seeds achieve dense top-8
neighbor recall `95.8%/97.7%`, message cosine `0.980/0.958`, and estimated vector-work ratios
`28.5%/28.5%`. The seed-family-example means are recall `0.967` (95% CI
`[0.949, 0.984]`), message cosine `0.969` (`[0.907, 1.000]`), and work ratio `0.285`
(`[0.256, 0.319]`). Increasing the candidate budget to 64 changes message cosine by less than
`0.0001` while raising estimated work to `0.493`.

At the same 32-candidate budget, the tile router improves neighbor recall over the contiguous-
segment hierarchy by `0.235` (`[0.182, 0.286]`) and message cosine by `0.183`
(`[0.040, 0.365]`), while reducing estimated work by `0.0593`. It also exceeds the 64-candidate
segment router in recall and message fidelity at less than half its candidate count.

### Joint adaptation and independent test

We compared three variants at the same 32-candidate budget: the trained router with the original
categorical value, 300-step adaptation with the categorical value frozen, and full joint
adaptation. Evaluation uses 57 untouched test families, 12 targets per family, and three model
seeds, for 2,052 target examples. Uncertainty is estimated by a seed--family--example
hierarchical bootstrap rather than treating targets as independent.

The original-value router has neighbor recall `0.970` (`[0.959, 0.980]`), message cosine `0.960`
(`[0.932, 0.982]`), and positive teacher-KL gain `0.001230`
(`[0.000763, 0.001692]`). Its true-residue CE gain is `0.002143`, but the interval
`[-0.001592, 0.007546]` still crosses zero. All three seed point estimates are positive.

Freezing the categorical value during adaptation gives CE gain `0.002655`
(`[-0.001084, 0.008061]`) and teacher-KL gain `0.001015`
(`[0.000589, 0.001450]`). Full joint adaptation gives CE gain `0.001963`
(`[-0.004157, 0.008334]`) and teacher-KL gain `0.001039`
(`[0.000195, 0.001867]`). Neither adapted variant significantly improves CE over the original
value. Full adaptation also underperforms its dense counterpart in all three seeds, with mean
routed-minus-dense CE gain `-0.000479` (`[-0.001932, 0.000700]`).

The supported default is therefore a stable categorical value with a dynamic trained router.
The tile router is the first supported low-budget execution mechanism, but adaptation of the
present value objective is not supported and should not be part of the default recipe. A future
value experiment must change the supervision or the non-additive message class rather than add
more joint fine-tuning.

### Gauge-preserving multi-neighbor aggregation

We next tested whether the remaining error came from the additive combination of otherwise
useful pair messages. A 3,240-parameter permutation-invariant set module encodes the selected
pair messages and routing scores, pools them with the same routing weights, and predicts a
nonlinear correction. The complete interaction is projected back into the target marginal
gauge, so the new module cannot reintroduce a one-body target effect. Its decoder is initialized
at zero, making the initial model exactly equal to the additive baseline.

An unconstrained 1,000-step pilot is an important negative engineering result: the correction
grows to `9.05` times the additive-message RMS and substantially degrades validation CE and
teacher KL. We therefore tested a prespecified stable version in which a smooth norm constraint
bounds the correction RMS to at most `0.5` times the additive RMS. Training uses 300 steps, and
the backbone, router, pair index, and categorical value are all frozen.

On the same three-seed, 57-family, 2,052-target independent test, the nonlinear module improves
teacher KL over the additive message by `0.000523` with interval
`[0.000313, 0.000823]`; all three seed estimates are positive. This fidelity does not transfer
to the observed residue. The nonlinear-minus-additive CE gain is `-0.001320` with interval
`[-0.004537, 0.001108]`, and only one of three seed estimates is positive. The nonlinear model's
absolute CE gain is `0.000823` (`[-0.002507, 0.004625]`), below the additive point estimate
`0.002143`.

The learned correction uses nearly the full allowed amplitude: its correction-to-additive RMS
ratio is `0.490` (`[0.482, 0.496]`). This rules out optimization collapse but suggests that the
conditional-response teacher contains a learnable direction that is not aligned with the true
residue target. The nonlinear aggregator remains implemented as a diagnostic, but it is rejected
from the default architecture. More aggregation capacity is not the next experiment; supervision
must become more directly tied to held-out biological outcomes.

### Teacher-target alignment and failed reliability gating

We measured the mismatch directly for every independent test target. The dense Transformer
teacher improves true-residue CE over the local background by `0.0283` on average
(`95%` interval `[0.0124, 0.0441]`), but it is better on only `54.2%` of targets
(`[51.2%, 57.2%]`). Teacher quality is therefore heterogeneous even though its aggregate CE is
better.

This heterogeneity predicts where teacher distillation is harmful. On the 940 targets where the
teacher is worse than the background, the nonlinear-minus-additive CE gain is `-0.00303`
(`[-0.00611, -0.00071]`) despite a positive teacher-KL gain. On the 1,112 targets where the
teacher is better, the CE difference is `0.00050` (`[-0.00355, 0.00356]`). The paired
family-stratum difference between these groups is `0.00353` (`[0.00101, 0.00578]`). Thus closer
teacher matching causes measurable harm precisely where the teacher is wrong.

A direct intervention did not solve the problem. We retrained all three set aggregators while
applying KL distillation only when the teacher had lower training-target CE than the background;
the true CE loss and gauge constraint remained active for every example. On independent test,
reliability gating changes nonlinear CE gain relative to ungated training by only `-0.000067`
(`[-0.000824, 0.000780]`). It significantly reduces teacher-KL gain by `-0.000400`
(`[-0.000593, -0.000270]`) but does not convert that reduction into biological improvement.

The diagnosis is supported, but hard per-example teacher filtering is rejected as the remedy.
The optional MSA or Transformer teacher should define candidate interaction directions, not the
final message amplitude by itself. Future supervision must use a demonstrably calibrated signal,
such as agreement across independent teachers, experimentally grounded mutation effects, or an
explicit reliability model evaluated on held-out families. Teacher KL and true-residue CE must
remain separate primary outcomes.

### Cross-teacher consensus is better prediction but not better interaction supervision

Two independently initialized Transformer teachers were available at matched scale. Averaging
their output probabilities improves masked-residue CE over the first teacher by `0.00838` with
interval `[0.00166, 0.01511]`, and the consensus itself improves over the local background by
`0.02557` (`[0.00918, 0.04178]`). This establishes a genuinely better auxiliary predictor.
However, pairwise teacher agreement is not a useful reliability score: cosine agreement has
Spearman correlation only `0.034` with consensus CE gain, and negative Jensen--Shannon divergence
has correlation `0.014`.

We tested consensus at two levels. First, replacing only the nonlinear aggregator's output target
improves its CE point estimate by `0.00100` relative to single-teacher training, but the paired
interval `[-0.00042, 0.00258]` crosses zero. The consensus-trained nonlinear model still does not
beat the additive interaction baseline.

Second, we rebuilt the complete conditional-response teacher. For every context mutation, the two
teacher probability distributions were averaged before taking log probabilities and applying the
two-sided weighted gauge. Rank-8 value and routing models were then trained from initialization in
three seeds. Consensus targets are easier to reconstruct in squared error: mean shape MSE is
`0.644`, versus `0.806` for the single teacher. But shape correlation is almost unchanged
(`0.390` versus `0.398`), and routing Spearman falls from `0.718` to `0.637`.

Most importantly, consensus conditional-response CE gain is `0.00143`
(`[-0.00083, 0.00400]`), compared with `0.00131` (`[-0.00047, 0.00323]`) for the single teacher.
The paired consensus-minus-single difference is only `0.000126`
(`[-0.001074, 0.001503]`). A better ensemble predictor and a lower reconstruction error therefore
do not make its conditional-response decomposition a better interaction target.

Cross-teacher consensus remains useful for calibrating ordinary output distributions, but it is
rejected as sufficient interaction supervision. The next value target should come from a
non-model intervention: experimentally measured mutation effects, a held-out evolutionary
conditional that is not used as inference input, or another causal perturbation objective. Pure
teacher KL, hard filtering, agreement gating, and probability averaging have now all failed to
produce a significant interaction CE gain.

### Leave-query-out evolutionary conditional supervision

The first non-model target uses the optional MSA only during training. The query row receives
zero statistical weight. For each pair, weighted joint counts are shrunk toward the independent
distribution with prior weight one, converted to log odds, and projected into the two-sided
weighted gauge:

\[
J_{ij}=\mathcal P_{p_i,p_j}
\left[
\log\frac{C_{ij}+\alpha p_i p_j}{N_{ij}+\alpha}
-\log(p_i p_j)
\right],\qquad w_{\mathrm{query}}=0.
\]

For a masked target, 16 nonlocal context sites are scored by block RMS. Their observed-residue
columns are combined with normalized strength weights. The raw log-odds message is too large:
at unit amplitude it worsens validation CE by `0.0449`. We therefore selected one global message
scale on validation. Three independent target/context sampling seeds all select `lambda=0.25`.
After locking this value, the leave-query-out evolutionary message improves CE on 57 untouched
test families and 2,052 targets by `0.05275`, with interval `[0.03207, 0.07401]`. This is the
first strong interaction teacher in the project and does not use the held-out query residue in
its statistics.

The categorical blocks are not intrinsically too high rank. Per-block rank eight explains
`94.1%` of energy on average and at least `86.4%` at the 10th percentile. The site-shared basis
is more restrictive: one shared left rank-eight subspace across the 16 context blocks of a target
explains `79.3%` of energy, rising to `96.7%` at rank sixteen.

The current single-sequence student nevertheless fails to transfer this teacher. With the
original 300+300-step schedule, validation index Spearman is `0.053`, shape correlation `0.034`,
and CE gain is negative. A sufficient-training run with 2,000 dense shape/index updates followed
by 1,000 sparse updates reaches only validation index Spearman `0.037`, shape correlation `0.087`,
and CE gain `-0.00130`. On its training families, the same checkpoint reaches index Spearman
`0.863` but shape correlation only `0.103`. A four-target memorization control reaches perfect
index ranking and shape correlation `0.726`, so the implementation can optimize but cannot encode
the diversity of 64-family evolutionary categorical fields in the present 67k-parameter shared
site-factor model.

This sharply changes the next architecture experiment. Routing is learnable, and the
evolutionary target is useful; the bottleneck is the value basis. The next operator should keep
stable site factors as a common component but generate a bounded pair-conditioned rotation or
residual basis only for the final top-k pairs:

\[
G_{ij}=\mathcal P_{p_i,p_j}
\left[(U_i+\Delta U_{ij})D_{ij}(V_j+\Delta V_{ij})^\top\right].
\]

The residual factors must be computed after sparse routing so they do not restore quadratic
cost. This pair-conditioned basis, not higher global rank or more router adaptation, is the next
candidate invention.
