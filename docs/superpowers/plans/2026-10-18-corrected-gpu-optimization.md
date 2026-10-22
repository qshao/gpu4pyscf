# Corrected GPU Optimization Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans.

**Goal:** Improve large-system throughput and gradient memory while preserving
the corrected factor-based auxiliary metric and original CPU accuracy gates.

**Architecture:** Reversible benchmark candidates run against a snapshot of the
corrected production sources. Use identical saved orbitals for gradient timing,
fresh processes for allocation measurements, and matched full SCF/force workflows
before promotion. No CPU-reference jobs during final GPU timing; one GPU job at
a time. Never overwrite historical reports or touch the unknown `record` file.

**Tech Stack:** Python, NumPy, CuPy FP64/cuBLAS, PySCF reference, pytest on GB10.

## Task 1: Establish corrected baseline

- [x] Snapshot affected current production/candidate/profile sources under
  `benchmarks/single_gpu/runs/2026-10-18-pre-next-optimization/` with SHA256.
- [x] Create `benchmarks/single_gpu/corrected_optimization.py` for provenance,
  qualified CPU references, saved-orbital validation, strict comparisons and
  immutable report writes into `optimization_results/`.
- [x] Add `test_corrected_profile.py`: full level3 Taxol SCF/force profile from
  saved orbitals used only as initial density. Record actual exchange block
  dimensions and stable metric application time separately. Original independent
  CPU gates remain1e-8Eh/1e-6Eh/Bohr. Inclusive profile scopes are not additive.

## Task 2: Test actual short exchange blocks

- [x] Add `test_short_exchange.py` comparing production against strided GEMM at
 1542AOs/300occupied columns for32/64/128aux rows, tight and padded storage.
  Include partial tails, dense independent contraction checks, two warmups and
  seven alternating timing samples; output each raw observation and identities.
- [x] Reject slower shape regions. Keep the existing qualified256-row path.
  Any threshold change requires regression tests and paired molecular SCF/force
  validation at a matched, explicit memory budget. Component ratios alone do
  not justify promotion or whole-workflow speed claims.
  **Protocol correction:** initial full resident attempts bypassed NWChem grid
  pruning through factory ordering, including after exchange withdrawal. They
  are invalid comparisons, not evidence of a candidate accuracy failure.
  Preserve both reports and sources; the new regression reproduces the bug.
  Corrected full acceptance asserts actual scientific factory settings.

## Task 3: Requalify packed gradients

- [x] Add `test_corrected_packing.py` with fresh baseline/packed allocation runs
  at identical Taxol orbitals and independently qualified CPU gates. First
  establish a corrected full-gradient oracle, then require candidate difference
 <=1e-9Eh/Bohr. Preserve historical V1/V2 failures unchanged.
- [x] If scientific gates pass, run both warmups and three alternating
  gradient-only timing pairs. Report medians and paired ratios; do not present
  gradient speedup as a complete SCF/workflow speedup.
  **Not run:** packing V3 fails the strict gate; retain rejection.
- [x] Promote only if measured memory reduction and repeated timing justify it.
  Before changing production preserve a callable corrected baseline so the
  acceptance benchmark cannot accidentally compare the candidate with itself.
  **Decision:** no packing promotion.

## Task 4: Acceptance and reporting

- [x] Run affected mathematical, streams/layout, RHF/UHF/ED/range-separated,
  disk/cleanup and independent CPU comparison tests after any production edit.
  Final corrected protocol run:56 focused tests plus12 configuration/reporting
  tests pass; `qualified_regressions.xml` and `configuration.xml` retain evidence.
- [x] Run matched full resident/disk workflows if a candidate is promoted,
  maintaining the original scientific settings and convergence requirements.
- [x] Obtain independent read-only review using requesting-code-review; fix
  material findings. Recompute source/native/geometry/reference identities and
  numerical gates in a separate summary. Document rejected experiments and
  remaining limits. Do not commit, merge or publish in this stage.

Success means a concrete, verified improvement or a documented rejection based
on measured evidence. An additional20% whole-workflow gain is a target, not an
acceptance assumption. Stable metric solver block tuning and broader XC changes
are follow-up options only when the corrected profile supports them.

## Evidence-driven scope adjustment

The corrected profile passes CPU gates and measures30.72s metric application
within98.63s gradient J/K. Short32/64/128-row exchange transforms win1.32-1.35x
on both tight/padded storage; molecular streamed J/K timing follows. Packed V3
still fails the unchanged1e-9 differential (2.46e-9); retain this failed experiment
without timing/promotion. Do not spend further work on this numerical route in
this stage or loosen its gate.

Fresh corrected gradient allocation observation is7,951,961,177B overall versus
6,473,769,120B inside J/K. Inspecting full-response XC reveals an available-memory
fraction heuristic for grid-weight derivative blocks that can use multiple GiB.
Add `xc_workspace.py` and `test_xc_workspace.py` to test a512MiB target for that
temporary block, preserving all grids, weights and response terms. Measure XC
alone in fresh processes, then full gradients with CPU and corrected-GPU gates.
Only promote if measured memory improves and paired timing shows no meaningful
regression. This addresses the observed allocation peak; a block target is not a
claim to bound total GPU/process RAM.

XC measurements pass: component7,887,651,993→1,124,073,384B, full gradient
7,951,961,177→6,556,902,612B (17.54% lower). Full-force differential3.23e-13,
independent CPU6.98e-9. Three alternating XC timing pairs give38.806s baseline
versus38.732s capped; treat as unchanged speed. The original function is
hash-pinned and remains callable for comparison. Current-production acceptance
uses new `production_{storage}_qualified_acceptance.json` outputs; both invalid
protocol reports remain unchanged. Final56-focused-test rerun passes, including
the configuration regression. Corrected full resident and disk acceptance both
pass, each with three SCF iterations. Maximum independent CPU errors are
2.225e-9 Eh and6.991e-9 Eh/Bohr. The independent summary recomputes numerical
gates, timing medians and source/native/reference/input identities and passes.
Both production optimizations are retained; packing remains benchmark-only.
