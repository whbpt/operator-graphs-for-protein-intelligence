# Evolutionary Pair-Basis Experiment: Round 1

Date: 2026-07-12

## Question

Can a basis residual generated only for routed pairs repair the failure of a site-shared
rank-8 categorical value student to learn leave-query-out MSA log-odds fields?

The tested field is

\[
G_{ij}=\mathcal P_{p_i,p_j}
\left[(U_i+\Delta U_{ij})D_{ij}(U_j+\Delta U_{ji})^\top\right].
\]

Each residual mixes the top two members of an eight-adapter bank. The adapter is evaluated
only for sampled teacher pairs or top-k inference pairs. The complete categorical block is
then projected through the exact two-sided weighted gauge.

## Results

### Four-target memorization

| Value model | Parameters | Shape MSE | Shape correlation |
|---|---:|---:|---:|
| Site-shared rank 8 | 67,022 | 0.763 | 0.726 |
| Pair-residual rank 8 | 71,959 | 0.599 | 0.776 |
| Site-shared rank 16 | 77,942 | 0.395 | 0.860 |

The pair residual improves expressivity at the same rank, but does not match simply doubling
the shared rank on four memorized targets.

### Held-out families, 300 dense plus 300 sparse updates

CE gain is background CE minus interaction-model CE, so positive is better.

| Value model | Shape MSE | Shape correlation | CE gain |
|---|---:|---:|---:|
| Site-shared rank 8 | 1.985 | 0.034 | -0.000510 |
| Pair-residual rank 8 | 1.976 | 0.056 | -0.000626 |
| Balanced pair-residual rank 8 | 1.977 | 0.052 | -0.000593 |
| Site-shared rank 16 | 1.984 | 0.060 | +0.000791 |

Rank 16 and the pair residual produce similar small shape correlations. The rank-16 CE point
estimate is positive in one seed, but its teacher KL is worse than background, so it is not
evidence that the MSA interaction message was reconstructed.

### Held-out families, 2,000 dense plus 1,000 sparse updates

| Value model | Shape MSE | Shape correlation | CE gain | Interaction RMS |
|---|---:|---:|---:|---:|
| Site-shared rank 8 | 1.949 | 0.087 | -0.001300 | 0.02245 |
| Pair-residual rank 8 | 1.966 | 0.064 | -0.000061 | 0.00250 |

Longer training does not rescue the pair residual. It suppresses the interaction amplitude and
returns close to the background rather than learning a useful message.

## Collapse Diagnostic

Without load control, 89.9% of directed pair sides select adapter 0. The effective number of
top adapters is 1.39 out of 8, mean cosine to a global adapter-weight template is 0.908, and
the residual/base RMS ratio is 0.155. The mechanism therefore degenerates toward a global
basis correction.

Following DeepSeek-V3, a non-gradient routing bias was then used only for adapter selection,
while semantic mixture weights remained functions of the unmodified logits. This raises the
effective adapter count to 6.72 and lowers template cosine to 0.474. However, the model shrinks
the residual/base RMS ratio to 0.044 and does not improve held-out reconstruction.

## Decision

The simple global adapter bank is rejected as the next default architecture. It can increase
memorization capacity, and load control can prevent routing collapse, but neither produces a
transferable pair field from the current independent site states.

The next hypothesis is an explicit sparse relation-state stream. After the router selects an
edge set, each selected pair receives a persistent state `r_ij`. Node-to-edge and edge-to-node
updates allow the relation to gather context before decoding a categorical field:

\[
m_i=\sum_{j\in\mathcal N(i)}\phi(h_j,r_{ij}),
\qquad
r'_{ij}=r_{ij}+F(r_{ij},h_i,h_j,m_i,m_j).
\]

The field `G_ij` is decoded from `r_ij` and projected through the hard gauge. MSA log odds
supervise only the training-time relation target; inference still accepts one sequence. This
keeps work proportional to the routed edge count while removing the assumption that a pair
field is a one-step product of two independent site factors.

## Next Falsification

Freeze the validated router and compare, on identical pairs and training examples:

1. site-shared rank 8;
2. site-shared rank 16;
3. one-step pair adapter rank 8;
4. rank-8 sparse relation state with one node-edge-node update.

The relation-state model must improve held-out categorical shape and teacher KL before any
larger backbone, more seeds, or kernel work is justified.
