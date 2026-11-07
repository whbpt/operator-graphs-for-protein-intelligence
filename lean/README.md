# Lean 4 companion: machine-checked certificates

This directory formalizes the monograph's finite counterexamples in Lean 4,
so that "audit" means *verified by the proof kernel*, not *asserted by the
author*. It is a companion, not a translation: only statements whose
content is exact enter here; empirical correspondences and estimand
discussions stay in the book.

## Layout

| File | Certifies |
|---|---|
| `OperatorGraph/PairLaw.lean` | Ch. 6, "Three valid pair laws with no joint law": the normalized 2×2 tables (positive half) and the incompatibility of `X₁ = X₂`, `X₂ = X₃`, `X₁ ≠ X₃` as normalized tables on `{0,1}³`, in marginal form, and for measurable coordinates on an arbitrary probability space. |
| `OperatorGraph/Anova.lean` | Ch. 5, "A three-variable ANOVA calculation": orthonormality of `1, x_j, x_k, x_jx_k`, coefficient extraction, the response audit `G(x_k) = A + Cx_k` with dynamic variance exactly `‖C‖²` in an arbitrary real inner product space, and the correlated-law cross term `‖A‖² + ‖B‖² + 2ρ⟨A,B⟩`. All kernel-checked. |
| `OperatorGraph/NoRecovery.lean` | Ch. 8/10 (printed), "Refinement reorganizes; it does not recover": processing cannot increase entropy (`entropy_processed_le`), pushforward composition, and the no-recovery inequality for the deterministic-composition setting. Self-built discrete-entropy layer (mathlib v4.35 has none). |
| `OperatorGraph/Shadow.lean` | Ch. 5, "Two operators with the same scalar shadow": both Frobenius norms = 10, shadow identity `M_A = M_B = √10/3`, zero row/column means, spectral witnesses (G_AᵀG_A eigenvalues {9,1,0}, BᵀB eigenvalues {3,3,0}), eigenvector independence, differing actions on the same source contrast. |
| `OperatorGraph/Bayes.lean` | Ch. 13, "A binary Bayes-error primer": pointwise optimality of `min{η,1-η}` and the convex-combination bound for randomized rules. |
| `OperatorGraph/Rem.lean` | Ch. 5, "Random energy" order-profile identity: `Σ_k C(L,k)(q-1)^k = q^L` (binomial theorem, the combinatorial spine that normalizes the profile). |
| `OperatorGraph/ThreeState.lean` | Ch. 3, "A three-state table, taken apart": the decomposition T = c + u + v + G and the fiber-operator action Kf = (1, -2/3, -1/3)ᵀ. |
| `OperatorGraph/MeanSpectrum.lean` | Ch. 10/11, "When averaging graphs creates a mode": eigenvalue witnesses for context spectra {0,0,2} and mean spectrum {0,1/2,3/2}, plus independence. |
| `OperatorGraph/Epistasis.lean` | Ch. 14, four-genotype epistasis primer: additive null, log-coordinate identity, multiplicative null. |
| `OperatorGraph/TwoBackground.lean` | Ch. 4, two-background calculation: orthogonality, norms, dynamic-norm identity. |

## Build

Requires [elan](https://elan.lean-lang.org); the toolchain is pinned in
`lean-toolchain` and mathlib in `lakefile.toml`.

```sh
cd lean
lake update          # once; writes lake-manifest.json (needs git + network)
lake exe cache get   # fetch prebuilt mathlib .olean files
lake build
```

On a throttled link (direct GitHub here measured at ~1.4 KB/s), route the
fetches through the local proxy (~4.6 MB/s):

```sh
export https_proxy=http://127.0.0.1:10808 http_proxy=http://127.0.0.1:10808
```

Trust structure, per `#print axioms`: **every theorem in this companion**
except the six `native_decide` exhibiting-table facts in `PairLaw.lean`
depends only on `propext`, `Classical.choice`, `Quot.sound` — kernel-checked,
same axiomatic base as mathlib. The six exceptions use `native_decide`
(compiled evaluation; kernel `decide` cannot reduce `Rat` arithmetic), as
marked in that file.

Adding a certificate: one file per book example, named after the chapter's
counterexample, with a module comment quoting the book passage it certifies.
