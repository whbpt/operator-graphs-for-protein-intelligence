# Prior-Art Matrix for Marginal/Sparse Interaction Separation

Date: 2026-07-12

| Literature | Mathematical object | Already solves | Does not by itself solve | Consequence for this project |
|---|---|---|---|---|
| Hoeffding decomposition / functional ANOVA | Orthogonal components under a fixed or exogenously conditioned product reference measure | Identifiable one-body, pairwise, and higher-order function spaces | Dynamic networks whose reference, router, or score observes the decomposed variables | Weighted-zero-sum interaction is prior art; our theorem needs an endpoint-excluding context |
| Hierarchical log-linear models | Main effects and interaction terms in contingency tables | Parameter gauge and interaction order | Single-sequence amortization; sparse hardware execution | ``Main effect plus pair effect'' is not novel |
| Marginal models for categorical data (Bergsma--Rudas 2002) | Marginal log-linear parameters and compatibility conditions | Rich separation of marginal and association parameters | A sparse single-sequence neural operator | Our parameter-separation story must be compared directly with marginal-model theory |
| Lancaster/Bahadur representations | Expansion of joint distributions relative to product marginals | Dependence components after removing marginal effects | Modern sequence backbone and conditional computation | Density-ratio interaction has classical statistical roots |
| Correspondence analysis | SVD of standardized contingency-table residuals | Marginal-normalized association modes and principal inertia | Learning which sequence pairs should be evaluated | Standardized residual rank is a better reference than raw-block rank |
| Potts/DCA/PLM | Pairwise energy model and zero-sum gauge | Protein coevolution, gauge-fixed categorical couplings | Optional-MSA single-sequence inference; exact separation of final predictive entropy | Gauge and categorical blocks cannot carry novelty |
| GREMLIN-LH | Spectral regularization of entropy contamination | Reduces entropy-related leading modes during Potts fitting | Non-symmetric Transformer dynamics; exact probability coupling | Motivates the problem, but our method must differ from spectral regularization |
| IPF / Sinkhorn matrix scaling | Positive matrix projected to prescribed row and column marginals | Exact bivariate marginal preservation; nonlinear row/column gauge correction | Global consistency of many pair couplings; causal interpretation | A coupling decoder is mathematically natural but algorithmically established |
| Discrete copulas / copula graphical models | Dependence with fixed univariate marginals | Separates marginal distributions from dependence parameters | Unique copula for arbitrary discrete variables; sparse sequence computation | ``Marginals plus dependence'' is a known modeling program |
| Minimum information dependence modeling (Sei--Yano 2024) | Multi-marginal exponential dependence model with a unique sum of one-body adjustments and potential after gauge fixing | Fixed marginals, Fisher-orthogonal marginal/dependence parameters, conditional inference, entropic-OT connection, and finite categorical/log-linear case | Endpoint-removable neural amortization; sparse single-sequence execution; optional-MSA protocol | This is high-risk close prior art; coupling, adjustment, or orthogonality cannot carry novelty |
| Mutual information / chi-squared association | Divergence from product marginals | Exact non-negative dependence strength; local quadratic approximation | Direction, causality, or recoverability from one sequence | MI can score an edge but is not a structural ground truth |
| Bayes risk / Fano entropy-error bounds | Posterior classification error and conditional entropy | Irreducible prediction limits from a stated observation | Which teacher is biologically correct; how to build the cavity mixer | Use to bound single-sequence support recovery, not as a novelty claim |
| Information-theoretic dynamic feature selection | Sequential acquisition by conditional information gain | Learned or greedy policies for querying informative features | Querying a coordinate here forbids emitting it as an interaction endpoint | Information gain is a heuristic; the exact certificate objective includes source opportunity cost |
| Mixture transition distribution models | Convex mixtures of lower-order conditional distributions | Valid conditional pooling and interpretable lag weights | A globally consistent joint posterior with prescribed site marginals | Multi-neighbor conditional pooling is established and is not pointwise marginal-preserving |
| Interaction Networks / Graph Networks | Persistent edge states with node-edge-node message passing | Relation-centered computation | Marginal categorical identifiability | Sparse relation state is edge-GNN prior art |
| AlphaFold2 Evoformer / AlphaFold3 Pairformer | Dense pair tracks and triangle updates | Rich pair context and geometric consistency | Linear sparse execution; marginal-gauge categorical semantics | Pair track and triangle update cannot be novelty |
| ESMFold | Single-sequence language state plus dense pair/folding trunk | Pair representation without an inference MSA | Explicit entropy/interaction gauge; sparse pair execution | Single-sequence pair state is already demonstrated |
| MSA Transformer | Axial processing of alignment rows and columns | Learned evolutionary structure from MSA input | MSA-optional inference contract | Our base forward must not require this data shape |
| Pure-attention rank-collapse theory | Row-stochastic token mixing and convergence toward token uniformity | Explains a token-space common mode | Categorical one-body/pair gauge | Token Perron projection and categorical projection must remain distinct |
| Deep Sets / additive sufficient statistics | Permutation-invariant sums of token-local features | Algebraically exact deletion of token-local terms | Rich ordered contextual representation; bitwise invariance after rounded subtraction | Useful implementation basis for cavity summaries, not a novelty source |
| Linear attention / Performer | Additive key-value sufficient statistics | Fast global aggregation and algebraic direct endpoint subtraction for token-local keys/values | Exclusion after ordinary contextual stacking; finite-precision residuals; categorical identifiability | A separate cavity stream may reuse this algebra, but contextual leakage and arithmetic path must be audited |
| S5 / GateLoop / Mamba and associative scans | Token-local affine or data-controlled recurrence products | Efficient ordered composition; GateLoop explicitly uses associative cumulative products | Deletion-aware pair cavities; lossless fixed-width context | Replacing deleted maps by identities plus segment queries is our candidate implementation, while recurrence and scan algebra are prior art |
| Boolean subcube partition complexity | Partitions of a Boolean product domain into cells with free coordinates | Formal language for functions constant under changes to free variables; can exceed decision-tree efficiency | The present q-ary label-constrained router, learnable neural routing, categorical gauge, protein contact coverage | Strict dynamic support borrows this language but is a more specialized object, not ordinary top-k |
| Hard attention / stochastic computation graphs | Sampled discrete actions trained with score-function gradients | A hard training forward and unbiased gradient estimators under standard conditions | The leaf-output coordinates need not be disjoint from every queried coordinate | Reuse the optimization machinery; the certificate semantics are separate |
| Neural decision forests / adaptive neural trees | Differentiable or stochastic tree routing and conditional single-path inference | Trainable hierarchical conditional computation | Routers may consume full representations; soft leaf mixtures do not preserve cavity support | Tree routing itself is established and cannot carry novelty |
| Active feature acquisition | Sequentially query features based on values already acquired | A trainable query policy with explicit acquisition budgets | The final action is prediction, not selection of unqueried interaction endpoints | Closest procedural template for a certificate policy, but a different semantic objective |
| Gumbel-Softmax / straight-through gates | Relaxed or biased gradients for discrete choices | Practical categorical optimization | A soft forward does not become endpoint-invariant by annealing alone | Gradient estimators must not be confused with the forward theorem |
| BigBird / Exphormer / Diffuser | Fixed local, random, global, expander, or diffused sparse connectivity | Linear or subquadratic global information flow and strong expressivity results | Sequence-specific direct contact edges; marginal/dependence semantics | Fixed sparse support is defensible engineering prior art but does not solve direct interaction support |
| DeepSeek MLA / DSA / MoE / NSA | Latent caching, routing, sparse blocks, load control | Efficient implementation and role-separated optimization | Statistical interaction identifiability | Borrow engineering patterns, never claim them as the invention |

## Closest mathematical neighbors

The closest prior art is not sparse attention. It is the intersection of:

1. functional ANOVA and log-linear interaction gauges;
2. correspondence analysis of standardized residuals;
3. fixed-marginal couplings produced by IPF/Sinkhorn;
4. discrete copula or dependence modeling;
5. sparse edge-state neural networks;
6. affine recurrent sequence models and associative range products.

Any eventual novelty argument must explain why the proposed operator is more than these six
components placed side by side.

## Narrow unresolved question

The current candidate question is:

> Can endpoint-deleted affine recurrence products retain enough ordered single-sequence context to
> amortize a fixed or hard-certificate-routed sparse set of directed common-gauge categorical
> fields, while an optional leave-query-out MSA supplies dependence targets in the identical
> model-defined gauge?

The potentially new part is the interface and training contract, not the component algorithms.

The new mathematical bottleneck is stronger than the coupling itself: a shared marginal used on
all active edges must exclude all active neighbor residues, while strict routing must also exclude
the candidate endpoints. Efficient endpoint-removable ordered context is the unresolved value
primitive. For support, a hard-forward certificate policy gives an exact training contract and
Bayes decision theory defines a budgeted posterior teacher-mass envelope. The unresolved questions
are the endpoint-deletion Bayes gap, gradient variance, direct coverage, and hardware-compatible
panel generation.

A globally endpoint-invariant, spanning undirected support is not the target: if every vertex is
active, invariance fixes the graph over the whole product domain. We choose a targetwise directed
operator to keep its conditional semantics distinct from a globally identified graph. Reciprocal
targetwise routing can remain nontrivial; symmetric protein-contact outputs are auxiliary readouts.

## Claims to avoid

- ``We invented a pair representation.''
- ``Sinkhorn removes entropy.''
- ``Low rank is the sparse signal.''
- ``Mutual information is causal interaction.''
- ``A zero-sum logit field exactly preserves final marginals.''
- ``An MSA log-odds field is structural ground truth.''
- ``The first attention eigenvector is the same object as categorical entropy.''
- ``Double-centering a table proves global ANOVA separation even when its network observes both
  endpoint residues.''
- ``Pairwise fixed-marginal couplings define one globally consistent joint distribution.''
- ``Every edge must have an adaptive intrinsic rank.''
- ``A fixed-width endpoint-removable summary is lossless for arbitrary sequence length.''
- ``Associative scan or an SSM is itself the proposed novelty.''
- ``Leave-two-out edge scores followed by top-k produce a jointly cavity-measurable support.''
- ``A fixed expander graph identifies arbitrary sequence-specific protein contacts.''
- ``Reciprocal targetwise routing is the same as a globally invariant graph decomposition.''
- ``DeepSeek-style latent compression shows that categorical interaction has fixed low rank.''
- ``Gumbel-Softmax or a straight-through estimator proves endpoint-invariant routing.''
- ``A strict router can use an endpoint category to decide whether that same endpoint is selected.''
- ``Optional MSA supervision makes an exact family interaction graph recoverable from one sequence.''
- ``Conditional mutual information is itself a structural interaction score.''

## Literature additions still needed

- neural contingency-table and differentiable IPF models;
- sparse optimal-transport layers and their kernel cost;
- global marginal-polytope consistency versus locally consistent pair couplings;
- correspondence-analysis regularization in modern neural categorical models;
- deletion-aware sequence models or dynamic-range-product neural precedents closer than the
  current SSM/segment-tree combination;
- approximation theory for low-dimensional affine recurrence products as conditional pair
  contexts;
- approximation and optimization theory for hard certificate routers with hardware-compatible
  panel generation;
- lower bounds for contact coverage under fixed cavity-measurable sparse supports.
