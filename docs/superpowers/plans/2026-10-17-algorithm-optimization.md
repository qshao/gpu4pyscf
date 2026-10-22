# Symmetric exchange and adaptive density implementation plan

**Goal:** Reduce ground-state SCF arithmetic without changing scientific settings.

**Architecture:** First implement reversible benchmark candidates. Symmetric
exchange uses SYRK for real float64 identical left/right factors, with full
matrix restoration after each tested update. Adaptive GGA density chooses
between existing MO and DM evaluations based on active AO/occupied dimensions;
include DM formation and gathering costs. Preserve general, response, complex
and other functional paths. Promote only candidates that improve whole workflows.

**Tech stack:** CuPy/cuBLAS, existing native GB10 build, pytest, cached matched
CPU references with fresh GPU controls. Execute inline in the existing branch.

- [x] Add failing correctness tests for symmetric Gram accumulation and adaptive
  density in `benchmarks/single_gpu/test_algorithm_candidates.py`; implement
  candidates in `benchmarks/single_gpu/algorithm_candidates.py` after observing
  failures. Compare with NumPy/independent density formulas, test aliasing,
  layouts, zero dimensions, fractional occupations, scratch and output reuse.
- [x] Run component timings on actual exchange shapes (537 AOs, 100/169 occupied
  factors, blocks 256/66) and density crossovers, including all extra work.
  Reject unhelpful candidates before expensive workflow timing.
- [x] Add reversible per-algorithm workflow contexts and run three alternating
  original/candidate pairs after both warmups on Vitamin C and tamoxifen;
  PBE for density, B3LYP for exchange. Require CPU energy/force gates, record
  iteration counts and exact source/native identities, restore bindings.
- [x] Promote qualifying candidates surgically to the corresponding production
  module, retain original routes for ineligible cases, and run relevant upstream
  numint/DF J/K plus molecular energy/force tests. Preserve prior uncommitted work.
- [x] Request independent review, audit artifacts/hashes and report component and
  complete-workflow gains separately. Record rejected experiments honestly.

Commands use `/home/qshao/miniconda3/envs/dft_oracle/bin/python -m pytest`.
No simultaneous GPU benchmarks, numerical threshold relaxation, unsolicited
commits, external publication or shared-environment changes.

Completed: isolated benchmark3tests passed648.82sec; production target
checks50passed8.18sec(one previously diagnosed baseline fingerprint excluded);
final original-GPU acceptance passed615.45sec; fresh molecular qualification
8passed5.89sec. Promoted adaptive realGGA density and CuPy-fallback symmetric
exchange at>=512AOs. Large-case cumulative speedups1.108xPBE/1.208xB3LYP.
All CPU gates and restoration checks passed. No precision/grid/tolerance
relaxation. Detailed raw/summary reports and limitations are in the benchmark
README. Independent code and baseline-isolation reviews found no material
blocker. Additional local-DM workspace is documented; no memory reduction claim.
