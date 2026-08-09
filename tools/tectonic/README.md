# Local Tectonic runtime

Build a TeX entry point from its source directory with:

```sh
sh ../tools/tectonic/build.sh main.tex
```

In the author's workspace, the wrapper uses a pinned Tectonic 0.16.9 binary,
`default_bundle.zip`, and a local format cache for an offline build. These large runtime files are
excluded from the public repository. On another machine the same wrapper uses the `tectonic`
executable available on `PATH`.

The pinned bundle includes the Computer Modern sans-serif font used by the current manuscript and
complete glyph-to-Unicode entries for its CMSY/CMEX subsets. Keep those entries when refreshing the
bundle: without them, xdvipdfmx can omit an entire `/ToUnicode` map and PDF text extraction will
silently drop mathematical minus signs and operators.

Pinned SHA-256 values:

```text
e62304878074c889e7f96d169698632c4fe695b525fb54a3473d7b2128f54512  tectonic-bin
52bc0392bffbee30a35cd7d95de3014393c88d256e6f5eb5a741082902457526  default_bundle.zip
a86ffcac335474fb9fae47cd9986b929719dc3ddf29bfb31123ecc1790ef6bbb  cache/formats/d06dc529270d63296ae382551930b086aba07d654c089d882ec1b6db5689c658-latex-33.fmt
```
