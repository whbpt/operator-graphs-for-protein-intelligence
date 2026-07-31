# DeepSeek Architecture Reading and Implications

Date: 2026-07-12

> **Status:** This note preserves the chronology of earlier ablations. Statements that fixed
> rank 8 was the current default are superseded by the round-2 theory review. The current protein
> design defaults to the full `19 x 19` tangent-product capacity on sparse support; low rank is a later
> compression hypothesis, not an identifiability assumption.

## Scope

This note focuses on the DeepSeek components that are relevant to a more efficient,
statistically disentangled sequence architecture:

- DeepSeek-V2: Multi-head Latent Attention (MLA) and DeepSeekMoE.
- DeepSeek-V3: auxiliary-loss-free expert balancing and multi-token prediction.
- DeepSeek-V3.2-Exp: DeepSeek Sparse Attention (DSA).
- Native Sparse Attention (NSA): natively trained blockwise compression, selection,
  and local-window branches.
- Engram: conditional static memory as a complement to conditional computation.

DeepSeek-R1 is important for post-training, but it does not materially change the base
sequence operator and is therefore not central to the present architecture question.

## Reread Verdict for the Current Bottleneck

This reread used the original DeepSeekMoE, DeepSeek-V2, DeepSeek-V3, Native Sparse
Attention, and DeepSeek-V3.2 reports. The most useful result is not another generic case
for low rank. It is a repeated architectural pattern: separate variables that serve
different purposes before optimizing them.

- MLA separates compressed content from the position-sensitive RoPE channel.
- DeepSeek-V3 separates the bias used to control expert load from the affinity used to
  weight expert outputs.
- NSA separates local, compressed-global, and selected-detail values so that the fast-
  learning local path cannot become a shortcut for every function.
- DSA separates indexer optimization from language-model optimization by detaching the
  indexer input and training it with its own distribution-matching objective.

This pattern changes the interpretation of our latest failure. The MSA teacher is strong
and the router is already adequate; increasing a site-shared categorical rank does not
address the observed train-to-family-generalization gap. The next architectural variable
to separate is therefore the categorical value basis itself:

1. a shared site basis carries reusable amino-acid semantics;
2. a bounded pair-conditioned residual carries relation-specific changes of basis;
3. the residual is generated only after sparse routing;
4. the assembled field is projected through the exact two-sided weighted gauge.

The conclusion is deliberately narrower than "DeepSeek validates our architecture."
DeepSeek validates the engineering discipline of role separation. The statistical
definition of a marginal-orthogonal interaction and the optional-MSA teacher remain our
own hypotheses and still require held-out-family validation.

## What DeepSeek Actually Separates

### MLA: content compression versus positional information

MLA jointly compresses keys and values into a latent vector and caches that latent
instead of per-head keys and values. A separate RoPE-bearing channel carries positional
information so that the content up-projection can still be absorbed into adjacent linear
maps at inference.

In DeepSeek-V2, 128 heads with head dimension 128 would expose 16,384 key/value channel
coordinates before compression, while the cached joint KV latent has dimension 512. The
query bottleneck has dimension 1,536 and the decoupled RoPE channel has head dimension 64.
The report also adds RMS normalization and explicit scale factors at the bottlenecks. This
is important evidence that a narrow latent can work at scale, but also that the bottleneck
cannot simply be inserted without controlling activation scale.

This is an engineering factorization. It reduces KV-cache size, but it does not identify
background and interaction effects. The core operator is still dense softmax attention,
and a dominant common mode can still enter the attention scores and values.

### DeepSeekMoE: shared computation versus routed computation

DeepSeekMoE keeps shared experts active for every token and routes each token to a small
set of fine-grained experts. DeepSeek-V3 adjusts an expert-specific routing bias using
observed load, while the expert output weights continue to use the unmodified affinity.
This avoids forcing semantic representations to pay the full price of a load-balancing
auxiliary objective.

This is relevant to optimization: a constraint needed for systems behavior can be placed
in the routing decision without contaminating the value computation. It is not, however,
a decomposition of token marginals and token-token interactions.

DeepSeek-V3 strengthens this principle. Expert affinity determines the output mixture,
whereas a separately updated expert bias affects only top-k selection. Load balancing is
therefore implemented as a control variable on routing rather than as an auxiliary gradient
that reshapes the semantic representation. The paper's ablations show that batch-wise
balancing preserves specialization better than forcing every sequence to use experts evenly.

### DSA: cheap indexing versus expensive aggregation

DSA uses a small ReLU indexer to rank preceding tokens, selects top-k latent KV entries,
and applies the main MLA operator only to the selected set. Its training has two stages:

1. Freeze the main model and train the indexer to match the dense attention distribution.
2. Enable top-k selection and jointly adapt the model while keeping a detached indexer
   objective.

The formal DeepSeek-V3.2 recipe uses 1,000 dense warm-up steps with the main model frozen,
then 15,000 sparse-training steps with 2,048 selected KV tokens per query. The indexer's
input is detached: its gradients come only from the attention-distribution distillation loss,
while the main model is trained only by the language-modeling objective. This is a strong
precedent for separating router optimization from value optimization.

This strongly supports the dense-warm-up-to-sparse schedule already observed in our
experiments. It also exposes two limitations:

- The indexer is still quadratic in sequence length, even if its constant is small.
- Distilling summed attention weights teaches the router to reproduce whatever common-mode
  contamination already exists in dense attention.

### NSA: compressed context versus selected detail versus local context

Native Sparse Attention uses three separately parameterized attention branches:

1. a compressed branch that maps overlapping sequential blocks to coarse tokens;
2. a selected branch that restores fine-grained tokens from the highest-scoring blocks;
3. a sliding-window branch dedicated to local dependencies.

The branch outputs are combined by learned gates. Block-selection scores are derived from
the already-computed compressed-attention scores, avoiding a separate quadratic token-level
indexer. The paper also gives each branch independent keys and values. Its reason is directly
relevant to our problem: local patterns learn quickly and otherwise become a shortcut that
prevents the global branches from specializing.

NSA is stronger evidence than DSA that sparsity must be designed jointly with training and
hardware. Its reported 64K-context speedups reach 9.0x for the forward pass, 6.0x for the
backward pass, and 11.6x for decoding, using continuous blocks rather than random token
gathers. The exact numbers are hardware- and configuration-dependent; the architectural
lesson is that FLOP reduction alone is not an efficiency result.

There are also two boundaries for our use case:

- NSA assumes that useful fine-grained tokens cluster in sequence-contiguous blocks. This is
  plausible for language, but nonlocal protein contacts can be discontinuous in primary-sequence
  coordinates. We must measure block concentration rather than assume it.
- NSA's compressed scores and branch values are still trained by language modeling. Separate
  parameters reduce optimization interference, but they do not make the branches statistically
  identifiable or remove marginal/common modes from pair interactions.

### Engram: static memory versus dynamic computation

Engram retrieves hashed local n-gram embeddings in constant work per token and gates them
with the contextual hidden state. It explicitly argues that a neural network should not
recompute static local knowledge through expensive dynamic computation.

This is the closest DeepSeek component to our background/residual view. Nevertheless,
Engram separates functions by implementation role, not by a statistical identifiability
condition. Its retrieved memory and dynamic backbone can still represent the same effects.

## Main Conclusion

DeepSeek provides five useful engineering principles:

1. Compress values before caching them.
2. Separate universal and conditional computation.
3. Train a dense routing teacher before making routing discrete.
4. Move static/local regularities out of expensive dynamic computation.
5. Make sparsity blockwise and natively trainable when the data supports block continuity;
   theoretical pair-count reduction alone is not an efficiency result.

It does not solve the first-mode contamination problem. None of MLA, MoE, DSA, NSA, or Engram
enforces that a dynamic interaction branch has zero marginal effect. Our candidate novelty
must therefore live in the definition and supervision of the interaction, not merely in
low rank, top-k routing, or a separate memory branch.

## Decision on Fixed Rank

DeepSeek makes fixed latent width credible as a systems interface, not as a claim that every
example has the same intrinsic rank. MLA always allocates a fixed-size latent, but different
tokens can occupy different directions and use different effective singular spectra. The same
distinction motivated a testable variant of our model:

- use a fixed maximum rank `R_max` for tensor shapes and efficient kernels;
- predict non-negative pair- and mode-specific gates `g_ijr` from the dynamic task state;
- test whether the number of active modes can vary by pair without losing shape;
- construct categorical values from the stable state;
- apply the two-sided weighted gauge after assembling the low-rank field.

The tested pair field was

\[
G_{ij}=\mathcal P_{p_i,p_j}
\left[\sum_{r=1}^{R_{\max}}g_{ijr}u_{ir}v_{jr}^{\top}\right],
\qquad g_{ijr}\ge 0,
\]

with an effective rank measured from the projected field rather than equated to `R_max`.
A fixed rank-8 implementation is an experimental budget. It should not become the architectural
hypothesis unless held-out spectra show that eight active modes are consistently sufficient.

This also prevents a subtle failure: projecting each factor independently before multiplication
does not in general guarantee that the final, gated sum remains correctly normalized when the
marginals change across layers. The hard gauge belongs on the assembled value field, although
centered factors remain useful as a numerically stable parameterization.

### Empirical update: soft effective rank is not compute rank

The adaptive-rank mechanism was tested on the conditional-response task. An unregularized
dynamic gate reduced participation-ratio rank from 8 to about `5.10` while preserving shape MSE
and slightly increasing shape correlation. It did not improve CE, and its teacher-KL gain was
smaller than fixed rank-8 by `0.0000614` with paired 95% CI
`[-0.000165, -0.0000027]`.

The gates were almost parallel across pairs: mean cosine similarity to the seed-specific global
gate template was `0.993/0.990`, and one or two modes dominated the top-gate frequency. This is
mostly learned global mode reweighting, not pair-specific rank selection. Penalizing the gate
down to effective rank near 3 significantly damaged categorical shape. A true fixed rank-5
model was unstable across seeds and had a mean shape-correlation decrease of `0.052` relative
to rank-8.

The historical prototype therefore kept fixed rank 8 as its experimental budget. The current
theory-first architecture does not. Soft mode gating remains an optional diagnostic, not a
claimed efficiency mechanism. A participation ratio below 8
cannot be translated into runtime savings unless modes are actually skipped by a discrete
selector and an appropriate kernel.

The same separation principle now has direct empirical support in the content-tile model. On
three model seeds and 2,052 untouched test targets, retraining the dynamic router while retaining
the original categorical value is as good as or better than jointly adapting the value. Full
adaptation does not improve true-residue cross-entropy over the original-value model and has a
negative routed-minus-dense point estimate in all three seeds. This mirrors DeepSeek-V3's useful
distinction between semantic affinity and load-control variables: systems control should change
which computation executes without forcing the stable semantic value to absorb the routing
objective.

## Consequences for the Invention

The architecture should borrow four DeepSeek patterns without presenting them as novelty:

1. MLA-style latent caching for the stable content/value stream.
2. DSA-style dense warm-up followed by sparse top-k execution.
3. MoE-style separation between semantic affinity and systems-control variables.
4. NSA-style independent local, compressed-global, and selected-detail branches, but only if
   protein interaction targets show enough block concentration to justify block kernels.

The candidate new component is narrower and testable: a dynamic router chooses where
non-additive computation is needed, while a stable, separately parameterized value path is
forced to be marginal orthogonal and is supervised by conditional response. The value path is
not jointly adapted by default. A useful working name is a
**marginal-orthogonal sparse interaction operator**. It is not yet a novelty claim; adaptive
rank is no longer part of the supported claim.

Multi-token prediction is also relevant as a training-only idea. For proteins, an analogous
multi-mask conditional-response objective could supervise several target sites from one encoded
sequence and then be discarded at inference. This may amortize the expensive teacher queries,
but it should be tested only after the single-target message gain reproduces.

## Revised Candidate After the MSA-Teacher Result

The previous site-shared value student is now a rejected default. A four-target control shows
that it can memorize, and training-family index correlation becomes high, but its categorical
shape does not generalize to held-out families. Rank diagnostics also show why simply choosing
a larger global width is an incomplete response: individual blocks are mostly low rank, while
their preferred categorical subspaces vary across pairs.

The next operator should use a shared-plus-routed value parameterization:

\[
G_{ij}=\mathcal P_{p_i,p_j}
\left[
(U_i+\Delta U_{ij})D_{ij}(V_j+\Delta V_{ij})^\top
\right].
\]

Here `U_i, V_j in R^(q x r)` are stable site factors. The pair residuals
\(\Delta U_{ij},\Delta V_{ij}\) are produced only for pairs admitted by the top-k router.
For bounded cost, the first implementation should not emit unconstrained dense matrices from
a large MLP. Instead, it should mix a small bank of basis adapters:

\[
\Delta U_{ij}=\sum_{m=1}^{M}\pi^U_{ijm}A^U_m,
\qquad
\Delta V_{ij}=\sum_{m=1}^{M}\pi^V_{ijm}A^V_m,
\]

where the adapter coefficients are pair-conditioned and only the largest one or two adapters
need be active. This is a categorical analogue of shared experts plus routed experts, but the
claim is not MoE novelty. Its purpose is to let the interaction basis rotate between pair types
without paying for a separate full `q x q` matrix at every pair.

The execution path should be:

1. a local/background path predicts marginals `p_i` from the single sequence;
2. a cheap task-conditioned indexer selects nonlocal pairs;
3. selected pairs choose a tiny adapter mixture and assemble `G_ij`;
4. the exact weighted gauge is applied after assembly;
5. additive categorical messages are accumulated without a softmax competition across
   unrelated neighbors.

The DeepSeek precedents define the engineering rules around this operator:

- use a fixed maximum latent width for kernels, not as an intrinsic-rank assertion;
- keep router load-control variables out of semantic value weights;
- give the local/background and interaction paths independent values;
- warm up the router densely, then train with the actual sparse execution pattern;
- measure wall-clock behavior with block-friendly candidate batches, not FLOPs alone;
- treat MSA supervision like a training-only auxiliary module, discarded at inference.

### Immediate falsification sequence

1. Reproduce the four-target memorization control with pair-conditioned rank 8.
2. Compare site-shared rank 8, site-shared rank 16, and pair-conditioned rank 8 under matched
   routed-pair count and matched parameter/FLOP accounting.
3. Require improvement in held-out-family categorical shape, not only training loss or index
   recall.
4. Verify the two weighted gauge errors numerically after every assembled pair field.
5. Run one formal seed before any multi-seed or larger-backbone expansion.

If pair-conditioned rank 8 cannot beat global rank 16 on held-out families, the shared-plus-
routed basis hypothesis is rejected and the next investigation should move to equivariant or
family-conditioned representations rather than further router tuning.

### Empirical update: the simple adapter bank is rejected

The falsification experiment is now complete for one seed. On four memorized targets,
pair-residual rank 8 improves shape correlation from 0.726 to 0.776, while shared rank 16
reaches 0.860. On held-out families after 300+300 updates, the corresponding correlations are
0.034, 0.056, and 0.060. After 2,000+1,000 updates, pair-residual rank 8 falls to 0.064 versus
0.087 for shared rank 8.

The unbalanced adapter selects one adapter on 89.9% of directed pair sides. A DeepSeek-V3-style
load bias, used only for top-k selection and not semantic weighting, increases the effective
adapter count from 1.39 to 6.72. The model then suppresses the residual/base RMS ratio from
0.155 to 0.044 and still does not improve held-out reconstruction. Load separation works as an
optimization mechanism, but it does not create missing pair information.

The next architecture should therefore maintain a persistent state on routed relations. A
sparse node-edge-node update can gather pair context before `G_ij` is decoded, while retaining
single-sequence inference and work linear in the selected edge count. This is a stronger change
than another factor adapter and is now the primary invention hypothesis.

## Candidate Architecture: Marginal-Orthogonal Interaction Network

The name is provisional. The architecture has three functional paths.

### 1. Background path

Use a causal/local convolution, state-space operator, or optional hashed motif memory to
produce background logits:

\[
b_i(a), \qquad p_i(a)=\operatorname{softmax}_a b_i(a).
\]

This path absorbs local syntax, amino-acid frequency, conservation, and other effects that
can be predicted without a specific nonlocal pair.

### 2. Interaction indexer

Predict a cheap pair score \(s_{ij}\), but do not train it against attention weights. Train it
against interaction-only task effects, for example

\[
w_{ij}\propto \left\|\Delta_{ij}\right\|,
\]

where

\[
\Delta_{ij}^{ab}
=L(x_{i\to a,j\to b})-L(x_{i\to a})-L(x_{j\to b})+L(x).
\]

At small sequence lengths, all pairs can provide a dense teacher. After warm-up, convert
the indexer to top-k routing. For long sequences, the indexer itself should be factorized or
hierarchical so that candidate generation costs \(O(Lr+Lk)\) or \(O(L\log L+Lk)\), rather
than DSA's formally quadratic lightweight scan.

### 3. Categorical interaction value

For a routed pair, predict a low-rank categorical field \(G_{ij}\in\mathbb{R}^{q\times q}\).
Define

\[
P_i=I-\mathbf 1p_i^\top,
\qquad
\widetilde G_{ij}=P_iG_{ij}P_j^\top.
\]

Then

\[
p_i^\top\widetilde G_{ij}=0,
\qquad
\widetilde G_{ij}p_j=0.
\]

This is a two-sided weighted ANOVA/Hoeffding projection. It removes both site marginals
from the pair field, so the pair branch cannot improve the objective merely by duplicating
a site-only residual predictor.

Given observed residue \(x_j\), the interaction message to site \(i\) is

\[
r_i(a)=\sum_{j\in\mathcal N(i)}\alpha_{ij}\widetilde G_{ij}(a,x_j).
\]

The final logits are \(b_i+r_i\), with an optional site-only orthogonal branch retained as an
explicit control rather than silently folded into the pair branch.

## Why This Is Not Just DSA or MLA

| Component | What is sparse or compressed | Training target | Identifiability |
|---|---|---|---|
| MLA | KV representation | language modeling | none for interactions |
| DSA | attended token set | dense attention mass | inherits attention modes |
| NSA | compressed/local/selected token blocks | language modeling | functional separation only |
| DeepSeekMoE | per-token FFN experts | language modeling plus load control | expert specialization only |
| Proposed | non-additive categorical pair effects | mixed mutation/task effect | two-sided marginal zero-sum |

The decisive distinction is not the top-k mechanism. It is that both routing and values are
defined using a non-additive task effect after marginal removal.

## Minimal Falsification Experiment

1. Freeze a trained local or Transformer teacher.
2. Sample held-out protein families, residue pairs, and a small mutation alphabet.
3. Compute single- and double-mutation losses and form \(\Delta_{ij}^{ab}\).
4. Compare equal-budget models:
   - site-only marginal-orthogonal residual;
   - unprojected pair model;
   - two-sided projected pair model.
5. First use dense pair evaluation to test identifiability.
6. Only if the projected pair model wins, train an indexer on \(\|\Delta_{ij}\|\), warm it up
   densely, and convert it to top-k.
7. Report held-out-family reconstruction error, ranking AP for strong epistasis, calibration,
   active pairs per token, and measured runtime.

A failure of the projected pair model to beat the site-only control would reject the present
interaction parameterization before any large-scale architecture or kernel work.

## Sources

- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- [Engram](https://github.com/deepseek-ai/Engram)
