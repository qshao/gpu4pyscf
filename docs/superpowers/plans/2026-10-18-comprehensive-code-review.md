# Comprehensive GPU Code Review and Repair Plan

**Goal:** Repair confirmed issues and measure code-only acceleration while
preserving quantum settings, FP64, backend dispatch and scientific gates.

**Scope:** All working-tree production changes against original HEAD, new disk
storage and affected caller interfaces; build/packaging and qualification code.
Existing branch work is preserved. Source-only entry snapshot and hashes are in
`benchmarks/single_gpu/code_review_results/before/` and `before_manifest.json`.
Historical benchmark snapshots/reports and unknown `record` remain untouched.
The user explicitly authorizes both review and repair; no separate approval is
needed for routine fixes. One direct-launched GPU process at a time.

## Review

- [x] Independent read-only review of DF storage, gradients, contractions.
- [x] Inspect build policy, source packaging and existing regression coverage.
- [x] Run pre-change regression set: 110 selected checks pass.

## Reproduce and repair

- [x] Add failing tests in `benchmarks/single_gpu/test_code_review_regressions.py`
  for singleton Coulomb blocks in both directions, CD/ED metric application to
  F-order and strided outputs, and default/explicit-wheel package naming.
- [x] Fix Coulomb layout selection using the full stride predicate and require
  integral element pitches before entering raw BLAS paths in `df/df_jk.py`.
- [x] Make default metric output C-contiguous; use explicit writeback for supplied
  noncontiguous outputs in `df/grad/rhf.py`, retaining the current C-order fast path.
- [x] Initialize the normal package name in `setup.py`; preserve explicit wheel
  suffix and CUDA dependency behavior. No install or shared environment changes.

## Code-only acceleration

- [x] Test output-buffer reuse for exact-dtype real/complex matrix contractions
  (`alpha=1,beta=0`, C-contiguous, no alias). Add the minimal direct `matmul(out=)`
  path in `lib/cutensor.py`; all other cases keep the existing implementation.
- [x] Test that each grid derivative batch is released before constructing its
  successor; benchmark the binary weighted reduction, reject it for regression, and
  weight the freshly owned derivative in place before its original unary sum in
  `grad/rks.py`. Compare full-response CPU gradients and forced multiple batches.
- [x] Benchmark old/new component calls from the hash-pinned entry snapshot,
  alternating repeated samples with synchronization. Measure requested allocator
  bytes separately. Promote measured improvements; do not infer workflow ratios.

## Qualification and reporting

- [x] Run affected regression suite plus open-shell/range-separated controls.
- [x] Run medium and large fresh atomic-guess SCF/full forces against existing
  independent CPU references, with unchanged settings and original gates.
- [x] Recheck source identities and independent results/code review. Document
  findings, fixes, measured component gains and unmeasured opportunities in
  `benchmarks/single_gpu/code_review_results/REPORT.md` and benchmark README.
  Prior campaign reports remain historical source identities after any fix.

## Final evidence

178 affected regression checks and 30 audit checks pass. Fresh Tamoxifen and
Taxol workflows pass independent CPU gates; maximum gradient errors are
1.058e-8 and 4.691e-9 Eh/Bohr. The CPU-only results audit passes, rehashing
the 549-file entry snapshot and verifying exactly five changed production files.
Direct matrix output reuse observes1.104x at64x64 and about1.006x at the tested
larger sizes, with no additional result tensor. The large weighted-grid sample
reduces additional tracked temporary allocation88,869,529 to2713bytes at
approximately2.6% more component time. No molecular speed ratio is inferred.
The slower binary-grid candidate remains archived with its source/helper bytes.
Final reports and read-only review are recorded under code_review_results/.
