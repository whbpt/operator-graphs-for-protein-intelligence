# Typed Relational Workspace Language Model

Date: 2026-08-08

Status: architecture proposal; no implementation or empirical claim.

## 1. Research claim

Many language tasks contain several persistent object types: tokens, spans, entities, events,
relations, evidence sources, tool states, and task goals. A dense Transformer stores all of them in
one token-indexed state and repeatedly reconstructs their relationships through attention.

The proposed Typed Relational Workspace Language Model (TRW-LM) assigns these objects separate
state spaces and lets computation follow declared local, hierarchical, and relational supports. Its
central hypothesis is:

> A language model with typed multiscale state can match a dense Transformer's open-ended language
> capacity while using less activation memory and learning long-range relational tasks from fewer
> examples.

The proposal transfers a design principle from protein models: align the organization of the
function with stable organization in the data. Language has no canonical geometry comparable to a
molecule, so every transferred bias requires its own intervention test.

## 2. Motivation and target bottleneck

For a length-$L$ sequence, standard dense attention forms token-token routes with quadratic support.
The conceptual state is

\[
H\in\mathbb R^{L\times d},
\qquad
A^{(h)}\in\mathbb R^{L\times L}.
\]

Embedding lookup already avoids materializing one-hot zeros. Conventional attention implementations
may materialize quadratic score and probability tensors; fused kernels such as FlashAttention avoid
writing the full matrix to high-bandwidth memory. Dense attention still performs quadratic pair
computation, while hidden activations, query/key/value projections, feed-forward activations, and
the KV cache grow with sequence length. Masked or numerically tiny weights create an execution gain
only when the kernel and data layout exploit their structure.

TRW-LM targets three repeated costs:

1. recovering the same entity and event identities in many layers;
2. reconstructing persistent relations from transient token states;
3. carrying long source documents after their relevant claims and provenance have been identified.

The project will treat wall-clock time and peak memory as primary outcomes. A sparse diagram without
measured sparse execution does not count as an efficiency result.

## 3. Transfer from protein architecture

The transfer is at the level of design principles:

| Protein-system pattern | Language-model analogue | Intended effect |
| --- | --- | --- |
| atom/residue/global hierarchy | token/span/entity/document hierarchy | local computation with compact global communication |
| persistent pair representation | persistent typed relation state | avoid reconstructing long-range relations in every layer |
| local atom attention plus global residue track | local token attention plus global entity/event workspace | reduce dense token-token routing |
| MSA rows kept as a separate axis | evidence sources kept as a separate axis | preserve provenance and expose source conflict |
| triangle updates | path and relation-composition checks | maintain multi-hop consistency |
| recycling | iterative workspace refinement | allocate more computation to unresolved states |
| geometric and permutation symmetry | source-order and identifier equivariance | remove arbitrary presentation choices from the learned problem |
| negative design against unwanted states | explicit penalties for unsupported and contradictory outputs | control failure states as well as target answers |

Entity boundaries, event types, discourse structure, and relevance have no fixed physical
counterpart. Their uncertainty and task dependence require creation, merging, splitting, and
deletion of learned workspace objects.

### 3.1 Prior-art boundary

Most ingredients already have strong precedents:

- [Longformer](https://arxiv.org/abs/2004.05150) and
  [BigBird](https://arxiv.org/abs/2007.14062) establish local, global, and sparse attention patterns;
- [FlashAttention](https://arxiv.org/abs/2205.14135) establishes memory-efficient exact dense
  attention kernels;
- [Perceiver](https://arxiv.org/abs/2103.03206) establishes a compact latent workspace for large
  inputs;
- [Transformer-XL](https://arxiv.org/abs/1901.02860) establishes recurrent segment memory;
- [Universal Transformer](https://arxiv.org/abs/1807.03819) establishes recurrent refinement and
  adaptive computation;
- [RETRO](https://arxiv.org/abs/2112.04426) establishes retrieval-conditioned language modeling;
- [Entities as Experts](https://arxiv.org/abs/2004.07202) establishes sparse access to explicit
  entity memories.

TRW-LM combines typed persistent relations, source-separated evidence, inspectable provenance,
negative design, and adaptive sparse execution in one state model. Novelty at the component level
is not claimed. A dedicated prior-art review and controlled experiments are required before making
an integrated-architecture novelty claim.

## 4. Typed state

For an input containing $L$ tokens, $M$ segments, $K$ active entities or events, $P$ relation edges,
and $R$ evidence sources, maintain

\[
\begin{aligned}
H&\in\mathbb R^{L\times d_t} &&\text{token state},\\
S&\in\mathbb R^{M\times d_s} &&\text{segment state},\\
U&\in\mathbb R^{K\times d_e} &&\text{entity and event state},\\
Z&\in\mathbb R^{P\times d_r} &&\text{sparse typed relation state},\\
V&\in\mathbb R^{R\times d_v} &&\text{evidence-source state},\\
g&\in\mathbb R^{d_g} &&\text{task state}.
\end{aligned}
\]

Every non-token object carries provenance: source identity and a compact set of supporting token
spans. Provenance is part of the computational state and can be inspected or supervised.

### 4.1 Token and segment tracks

Tokens use causal local attention with window width $w$ and optional global anchors. Segment states
pool contiguous token spans and return messages to their member tokens. Initial segments can follow
document boundaries, paragraphs, code blocks, tool calls, or deterministic fixed-size blocks.
Learned boundary updates may merge or split them later.

### 4.2 Entity and event track

Entity/event slots aggregate mentions across segments. Slots remain typed but open-vocabulary: the
model predicts a type embedding instead of selecting from one fixed ontology. A slot may be created,
merged, split, or retired. Each structural action records its token and source support.

### 4.3 Relation track

The relation graph contains at most $k_r$ outgoing candidates per entity/event. A router proposes
support using local token evidence, current entity states, task state, and retrieved sources. The
edge state records relation type, temporal scope, confidence, and provenance.

The default design excludes all-pairs $K^2$ storage. Dense relation computation remains a small-$K$
control and carries no efficiency claim.

### 4.4 Evidence-source track

Documents, retrieved chunks, observations, and tool outputs retain separate source identities.
Within-source processing and cross-source comparison use different parameters. Permuting independent
sources should permute their source states and leave the final answer distribution unchanged.

This axis supports explicit operations for agreement, contradiction, recency, authority, and
dependency between sources. Repeated copies can be detected before their content is counted as
independent evidence.

### 4.5 Task state

The task state controls which entities, relations, and sources remain active. It also supplies the
halting signal for iterative refinement. The initial prototype will keep $g$ small and auditable;
an unrestricted global vector could become a dense information bypass.

## 5. One refinement cycle

A cycle performs the following updates:

\[
\begin{aligned}
H' &= \operatorname{LocalAttn}(H)
      +\operatorname{Msg}_{S\to H}(S)
      +\operatorname{Msg}_{U\to H}(U),\\
S' &= \operatorname{SegmentUpdate}(S,\operatorname{Pool}(H')),\\
U' &= \operatorname{EntityUpdate}
      (U,\operatorname{Mentions}(H',S'),\operatorname{Msg}_{Z\to U}(Z)),\\
Z' &= \operatorname{RelationUpdate}
      (Z,U',V,g),\\
V' &= \operatorname{SourceUpdate}(V,S',U',Z'),\\
g' &= \operatorname{TaskUpdate}(g,U',Z',V').
\end{aligned}
\]

Relation composition runs only on typed paths admitted by the sparse graph. Examples include
temporal transitivity, ownership chains, reference resolution, data-flow dependencies, and
agreement or contradiction across sources. Each operation is learned under a declared type
signature. An unrestricted triangle update serves as an ablation.

After each cycle, an uncertainty head marks unresolved objects and relations. Later cycles update
those neighborhoods and their provenance closure. Adaptive halting stops when the predicted value
of another cycle falls below its measured compute cost.

## 6. Decoder and persistent memory

The causal decoder reads:

1. recent local token states;
2. selected segment summaries;
3. task-relevant entity and relation states;
4. source states and supporting spans;
5. tool or environment observations represented as typed events.

Long-term memory stores workspace objects and provenance. Raw token KV states remain in a bounded
recent-context cache. A memory record has an explicit schema:

\[
(\text{entity/event},\text{relation},\text{time},\text{source},\text{confidence},\text{support}).
\]

Eviction uses task relevance, recency, confidence, and graph connectivity. Deleted records retain a
compact tombstone when later updates may refer to them. The prototype must compare this memory with
ordinary KV caching and retrieval over raw chunks.

## 7. Inductive biases and invariance tests

TRW-LM introduces six testable biases:

1. **Locality:** most token interactions are local.
2. **Hierarchy:** long-range communication passes through a smaller set of spans, entities, and
   events.
3. **Typed persistence:** important relations survive across layers and generation steps.
4. **Source separation:** evidence aggregation preserves provenance and conflict.
5. **Sparse composition:** multi-hop updates follow selected typed paths.
6. **Adaptive computation:** unresolved states receive additional refinement cycles.

Three invariance or equivariance tests are required:

\[
\begin{aligned}
\text{source permutation:}&\quad
p(y\mid \pi\mathcal D)=p(y\mid\mathcal D),\\
\text{identifier renaming:}&\quad
F(\rho x)=\rho F(x),\\
\text{irrelevant-source insertion:}&\quad
p(y\mid\mathcal D\cup\mathcal N)\approx p(y\mid\mathcal D).
\end{aligned}
\]

Here $\pi$ permutes independent sources, $\rho$ consistently renames controlled opaque entity labels
or program identifiers, and $\mathcal N$ contains controlled irrelevant evidence. These properties
will be measured under interventions and will not be inferred from attention maps.

## 8. Complexity budget

Ignoring channel constants, the persistent state target is

\[
O(Ld_t+Md_s+Kd_e+Pd_r+Rd_v),
\qquad P\le Kk_r.
\]

One refinement cycle targets

\[
O(Lwd_t)+O(Lk_xd_e)+O(Kk_rd_r)+O(Rk_sd_v),
\]

where $k_x$ bounds token-to-workspace messages and $k_s$ bounds source comparisons. These bounds
depend on executed sparse kernels. Routing, sorting, indexing, and irregular memory access are part
of the measured cost.

The proposal succeeds computationally only when it improves at least one Pareto frontier:

- quality at fixed peak memory;
- quality at fixed training FLOPs;
- supported context length at fixed hardware;
- latency at fixed answer quality.

## 9. Training objective

The first integrated objective is

\[
\begin{aligned}
\mathcal L={}&\mathcal L_{\mathrm{LM}}
+\lambda_p\mathcal L_{\mathrm{provenance}}
+\lambda_r\mathcal L_{\mathrm{relation}}
+\lambda_c\mathcal L_{\mathrm{cycle}}\\
&+\lambda_n\mathcal L_{\mathrm{negative}}
+\lambda_b\mathcal L_{\mathrm{budget}}
+\lambda_h\mathcal L_{\mathrm{halting}}.
\end{aligned}
\]

- `LM` preserves general language modeling.
- `provenance` requires claims to identify supporting sources and spans.
- `relation` supervises synthetic and observed entity/event relations.
- `cycle` tests whether a later refinement resolves earlier uncertainty without erasing supported
  facts.
- `negative` penalizes unsupported claims, unresolved contradictions, identity swaps, stale facts,
  and fabricated citations.
- `budget` prices active tokens, relation edges, source comparisons, and refinement cycles.
- `halting` calibrates the expected gain from further computation.

Training begins with deterministic segments and supervised synthetic relations. Learned structural
actions are introduced only after the fixed-structure model clears the quality and efficiency gates.

## 10. Minimal experiment

### 10.1 Relational Recall and Revision benchmark

Generate multi-source episodes containing:

- persistent entities with aliases;
- timestamped events and revisions;
- multi-hop relations;
- duplicated sources;
- controlled contradictions;
- irrelevant but lexically similar distractors;
- questions requiring an answer and exact supporting provenance.

Episode length, source count, relation depth, contradiction rate, and update frequency vary
independently. The generator supplies the true entity-event graph and every answer's support set.

### 10.2 Models

Compare parameter- and training-FLOP-matched systems:

1. dense causal Transformer using FlashAttention and a matched KV-cache policy;
2. sliding-window Transformer with global tokens;
3. recurrent Transformer with adaptive depth;
4. frozen dense Transformer plus an external graph workspace;
5. integrated TRW-LM with fixed structural actions;
6. integrated TRW-LM with learned sparse routing.

The first implementation should use a 150M--350M parameter scale. This range is large enough to
expose language-model tradeoffs and small enough for controlled ablations.

### 10.3 Metrics

Report:

- next-token loss on a general held-out corpus;
- exact answer and calibrated answer probability;
- provenance precision and recall;
- contradiction detection;
- entity identity retention after long delays;
- source-permutation and identifier-renaming defects;
- peak training memory, peak inference memory, throughput, latency, and executed FLOPs;
- active workspace size, relation degree, and number of refinement cycles.

## 11. Promotion gates

### Gate A: representation value

- General-corpus validation loss remains within 2% of the matched baseline.
- Relational Recall and Revision accuracy improves on at least three held-out difficulty axes.
- Provenance recall improves without increasing fabricated support.

### Gate B: structural semantics

- Source permutation and consistent identifier renaming produce small declared defects.
- Removing the relation track selectively damages multi-hop tasks.
- Removing source identity selectively damages conflict and provenance tasks.
- Random relation graphs fail to reproduce the gain.

### Gate C: computational value

At a context length of at least 32k tokens, the integrated model must achieve one of:

- at least 40% lower peak inference memory at matched quality;
- at least 1.7x longer supported context at matched peak memory;
- at least 25% lower executed FLOPs at matched quality.

Wall-clock latency may degrade by at most 20% at the first gate. Later optimization must remove that
gap. A theoretical sparse count with slower dense fallback does not pass.

### Gate D: external validity

After the synthetic benchmark is frozen, evaluate without structural retuning on:

- multi-document question answering with citations;
- repository-level code reasoning under variable renaming;
- long-running agent traces with tool-state updates;
- document revision and temporal consistency tasks.

At least two task families must retain the efficiency gain and improve a structural metric.

## 12. Work plan

### Phase 0: measurement and controls

- instrument peak memory, activation memory, KV cache, FLOPs, and kernel utilization;
- implement the synthetic generator and invariance tests;
- establish dense, local-attention, and recurrent baselines.

### Phase 1: external workspace

- attach a typed workspace to a frozen small language model;
- test whether explicit entity, relation, and source states improve provenance and revision tasks;
- use the result to freeze the first state schema.

This phase tests representational value. It cannot establish end-to-end memory efficiency because
the dense backbone remains active.

### Phase 2: integrated fixed-structure model

- replace dense token attention with local attention and deterministic segment communication;
- integrate persistent relation and source tracks;
- train at 150M--350M scale;
- run track-removal and random-graph controls.

### Phase 3: learned sparse structure

- introduce entity actions and top-$k$ relation routing;
- train with explicit execution prices;
- use kernel-aware block layouts and measure realized utilization;
- add adaptive refinement and calibrated halting.

### Phase 4: scaling decision

Scale beyond the prototype only after Gates A--C pass. Failure analysis should identify whether the
limiting factor is representation error, routing error, sparse-kernel overhead, relation bottleneck,
or loss of open-ended token interaction.

## 13. Main risks

1. **Wrong structural bias.** Language may require relations that the active schema suppresses.
2. **Parser dependence.** Early entity errors can persist and contaminate later reasoning.
3. **Workspace bottleneck.** Compression may erase lexical detail needed for generation.
4. **Sparse execution overhead.** Indexing and routing may cost more than dense matrix operations.
5. **Global bypass.** Task or segment states may silently carry all information and make typed
   relations decorative.
6. **Unstable structural actions.** Slot creation and merging may produce discontinuous training.
7. **Benchmark leakage.** Synthetic graph regularities may reward the proposed representation by
   construction.

Each risk has a corresponding control in the promotion gates. A failed gate stops the project or
narrows its claim; increasing model size does not override the failure.

## 14. Immediate next artifact

The next deliverable is a benchmark specification. Full model implementation begins after the
baseline and intervention suite are frozen. The specification should define the episode grammar,
provenance ground truth, baseline configurations, hardware reporting format, and the exact pass/fail
thresholds from this proposal.
