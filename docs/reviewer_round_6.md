# Independent Theory Review, Round 6

Date: 2026-07-12

Scope:

- `docs/trainable_certificate_router_theory.md`
- `docs/cavity_support_routing_theory.md`
- the cavity-routing section of `paper/theory_main.tex`
- `docs/prior_art_matrix.md`
- `docs/mathematical_foundations_and_prior_art.md`

Experiments remained paused. The review tested the hard certificate theorem, blindness results,
coverage bounds, soft relaxations, complexity accounting, and prior-art boundaries.

## Verdict

No P0 issue was found. The hard-forward theorem is valid: for a fixed independent random seed and
deterministic tie rule, a leaf that emits only coordinates never queried on its realized path is
pathwise targetwise invariant. Target blindness, self-membership blindness, and the endpoint-only
support limitation are also valid.

The initial draft needed narrower quantifiers for stochastic coverage, a complete soft-union
cavity condition, an explicit terminal stop gradient, and full layer-cost accounting. These were
corrected before accepting the section.

## Accepted theorem

For a target `i`, let the hard policy sequentially observe

\[
H_t=((a_1,x_{a_1}),\ldots,(a_t,x_{a_t}))
\]

using an independent fixed seed. If the terminal leaf satisfies

\[
N_i(x;\omega)\cap(\{i\}\cup Q_{leaf})=\varnothing,
\]

then changing only `x_i` and the selected source categories leaves the transcript, stop action,
and source set unchanged.

The observation at a query node must be token-local. A nonlocal observation is structurally
admissible only if its complete upstream dependency set is disjoint from the target and from the
union of all coordinates any descendant leaf can emit. An ordinary contextual token state does
not satisfy this condition merely because its nominal index is unselected.

## Blindness consequences

Pathwise strict support implies:

\[
N_i(x_i,x_{-i})=N_i(x_i',x_{-i}),
\]

and for every candidate `j`,

\[
\mathbf1\{j\in N_i(x)\}
\]

is invariant to changes in `x_j`. Thus the router cannot selectively open an arc using a
nonconstant rule of the two realized endpoint categories alone. It may open the arc using position
or third-party context and then represent the endpoint-specific signed effect in
`G_{i<-j}(:,x_j)`.

## Coverage bounds

For a deterministic tree or one fixed-seed realization, if each of `B` queries reveals one raw
q-ary category and every leaf emits at most `k` sources, then

\[
\left|\bigcup_xN_i(x)\right|\le kq^B.
\]

Full direct positional coverage requires `k q^B >= L-1`. Realizing every size-`k` source set
requires

\[
q^B\ge {L-1\choose k}.
\]

For outcome alphabet sizes `m_t`, replace `q^B` by `product_t m_t`. A finite seed bank of size
`S` changes the bounds to `S k q^B` and `S q^B >= choose(L-1,k)`. Continuous observations or
seeds have no finite cross-seed counting bound without quantization or another finite-policy
assumption. Random coverage is not sequence evidence for selecting the correct source.

## Training gradient

The likelihood-ratio sum now includes the terminal stop action. If `tau` is the number of query
actions, then `A_{tau+1}=STOP` and

\[
\nabla_\theta J
=\mathbb E\left[
\sum_{t=0}^{\tau}
(C-b_t(H_t))\nabla_\theta
\log\pi_\theta(A_{t+1}\mid i,H_t)
\right]
+\text{pathwise terms}.
\]

The baseline must be conditionally independent of the current sampled action, and standard
integrability conditions are required. Hard Gumbel-max preserves the forward theorem with fixed
noise and deterministic ties. Straight-through estimation leaves the hard forward intact but has
a biased gradient. Soft Gumbel-Softmax does not itself provide certificate semantics.

## Soft-union correction

Deleting the union of positive-mass leaves from the background is not enough. A structurally
auditable soft construction also requires:

- the union support itself is targetwise invariant;
- all mixture weights and active values are measurable with respect to the same union cavity;
- both directional references and all gauge-dependent teacher, norm, and spectrum quantities use
  that same cavity;
- the true upstream dependency closure, not just nominal gate indices, excludes the target and
  active union.

Other exact cancellations are possible in principle but require a direct functional proof.

## Complexity correction

`O(L(B+k))` is router-only and holds only with bounded controller actions, `O(k)` panel generation,
duplicate handling, bounds handling, and disjointness enforcement, without scoring all positions.
The complete layer also includes:

\[
O(LC_\circ)
\]

for segment-tree construction,

\[
O(L(k+1)\log L\,C_\circ)
\]

for target-cavity queries, plus background, `k+1` reference, and `k` field decoders per target.
Subquadratic scaling requires the total per-target cost to be `o(L)`; bounded widths and degrees
are one sufficient regime.

## Prior-art verdict

The current boundary is accurate and appropriately narrow:

- hard attention and stochastic computation graphs already supply score-function training;
- neural decision forests and adaptive neural trees already supply trainable tree routing;
- active feature acquisition already supplies sequential query policies;
- Gumbel-Softmax, straight-through gates, and learned subset selection are established.

The present claim is limited to the access-control contract that an output leaf selects only
unqueried coordinates and its integration with a whole-neighborhood categorical cavity. The
literature search is explicitly non-exhaustive, so this is not yet a novelty claim.

## Research-direction check

The certificate router is aligned with the project, but it solves only endpoint leakage through
support choice. It does not prove that the selected support is biologically real, that third-party
single-sequence context is sufficient, that the hard policy is easy to optimize, or that entropy
and interaction strength are statistically independent. Training-time function-space separation
still comes from the common cavity and directional marginal gauge; MSA remains an optional
teacher rather than an inference input.
