# Independent Reviewer Report: Round 1

Date: 2026-07-12

## Verdict

The project contains a sound algebraic idea, but the empirical draft had two P0 validity
problems and had moved too far into routing and kernel details before establishing the central
scientific object. The current evidence does not yet support a new Transformer claim.

## P0 findings

### Target-example alias

`torch.from_numpy(sequence)` shared storage with the NumPy query sequence. Masking one target
therefore modified later examples from the same family. The current builder could accumulate
multiple masks or select token 21 as an amino-acid context.

Action taken:

- target tensors are now copied before masking;
- invariant tests require the source sequence to remain unchanged;
- only the current target may be masked;
- context tokens must remain inside the amino-acid alphabet.

### Teacher/student gauge mismatch

The MSA teacher was projected under `p_MSA`, while the student was projected under the frozen
background `p_model`. The shape loss compared matrices in different marginal-orthogonal
subspaces. A reviewer diagnostic found large residual gauge error and substantial distortion
after re-projection.

Action taken:

- the MSA log-odds block is reprojected into the background/student gauge before computing
  strength, shape, or the executable message;
- marginal KL to the MSA is now reported by the evolutionary examples;
- teacher and student comparison in a common gauge is a formal model invariant.

The corrected teacher calibration remained strong and selected the same message scale, but its
numerical result is not used to resume architecture experiments because the user paused them.

## P1 findings

1. The draft had demonstrated orthogonal residual fitting around a frozen marginal reference,
   not joint training-time separation of entropy and sparse signal.
2. The phrase "value basis is the bottleneck" exceeded the evidence. Encoder information,
   gauge mismatch, data scale, optimization, and split leakage had not been excluded.
3. The MSA field is a strong evolutionary conditional teacher, not automatically sparse,
   structural, or causal interaction ground truth.
4. Sampling repetitions were treated too much like independent seeds. Family must be the
   highest bootstrap cluster.
5. Query-only sequence clustering does not prove MSA/profile independence between splits.
6. A persistent relation state is close to AlphaFold pair tracks, ESMFold, Interaction Networks,
   Graph Networks, and edge-state GNNs.
7. The current convolution-GRU prototype is not a Transformer replacement.

## Scope correction

The project had accumulated too many router-side branches: hashing, segment hierarchy,
content tiles, nonlinear aggregation, teacher filtering, consensus, adaptive rank, and adapter
balancing. These are useful negative results, but they do not answer the central question:

> In one common marginal gauge, does a single-sequence state contain recoverable information
> about an MSA-derived categorical interaction field?

Further experiments are paused. The research focus is now:

- functional-ANOVA identifiability;
- token-space versus categorical common modes;
- marginal-weighted geometry;
- pair-support sparsity versus categorical rank;
- optional-MSA supervision in the model gauge;
- prior-art and novelty boundaries.

## Fixed-rank decision

A universal intrinsic rank is rejected. The maximum nontrivial rank after marginal removal is
`q-1`. A fixed `R_max` may be used for tensor layouts and kernels, but active rank must be
pair-dependent and tied to an approximation error. Soft gates that compute every mode do not
provide adaptive rank or compute savings.

## Prior-art boundary

The following cannot be claimed as novel:

- zero-sum or weighted gauges;
- Hoeffding/functional ANOVA;
- sparse attention or top-k routing;
- low-rank categorical values;
- pair tracks, triangle updates, or node-edge-node message passing;
- optional MSA distillation by itself.

The narrow candidate hypothesis is:

> A single-sequence sparse relation operator whose categorical values lie in the tangent-product
> space defined by the model's own marginals, with optional leave-query-out MSA supervision
> reprojected into that same gauge.

This remains a hypothesis. It is not yet a demonstrated novelty or architecture win.

## Required gates before experiments resume

1. Prove common-gauge teacher/student comparison in code and mathematics.
2. Specify joint marginal/relation training and gradient ownership.
3. Audit profile/MSA overlap in the data split.
4. Pre-register baselines: ungauged decoder, ordinary edge GNN, dense pair track, and shuffled
   neighbor control.
5. Report pair-support sparsity independently from categorical rank.
6. Define what makes the final operator Transformer-like: token mixer, layer composition,
   complexity, and standard baselines.
