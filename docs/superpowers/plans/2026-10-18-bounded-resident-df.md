# Bounded Resident DF Implementation Plan

**Goal:** Identify current SCF/gradient allocation peaks and reduce resident DF
construction scratch while retaining FP64 accuracy and measuring any speed cost.

**Architecture:** Work on the existing development branch, preserving the prior
immutable campaign snapshot as the control. Add benchmark-only phase allocation
instrumentation; do not edit the frozen runner, configuration or memory profiler.
Use the original atomic density, quantum settings and independent CPU references.
One GPU process runs at a time. No commits, staging or changes to unknown `record`.

**Tech stack:** Python, CuPy, PySCF, pytest, existing native SM121 libraries.

## 1. Measure allocation phases and current contractions

- [x] Add `benchmarks/single_gpu/phase_memory.py`: a context manager wrapping the
  existing allocator with retained owners. `phase(name)` records entry live bytes
  and peak global live bytes while active, including worker-thread allocations.
  Nested stage peaks are inclusive and must not be summed. Restore the allocator
  on success and exceptions. Unit tests must check cross-phase lifetime, nesting,
  threaded accounting and restoration before running a GPU experiment.
  Eight CPU/GPU tests pass. Independent specification/quality review passed.
- [x] Add `test_resident_workspace_run.py`: run current Taxol from atomic density
  at the frozen original-comparison settings, with coarse/fine synchronized
  diagnostic scopes and phase allocation instrumentation. Assert actual GPU grid
  pruning and full gradient response. Record source/helper/native/reference hashes
  and CPU energy/force comparisons under `resident_workspace_results/` without
  overwriting prior reports. Add explicit timing of optimized exchange helpers.
- [x] Run one diagnostic Taxol workflow. Check whether resident build drives the
  30.1GB peak; report diagnostic timing separately from performance controls.
  The build peak is 30.100 GB; gradient peak is 16.669 GB. Exchange transform and
  Gram helpers take 184.367 s and 100.057 s in the synchronized diagnostic.

## 2. Bound resident build scratch

- [x] Write failing planner tests for a new `_resident_build_batch_size` in
  `gpu4pyscf/df/df.py`, including low-memory inputs, optional transformed buffers,
  minimum batch size and invalid budgets. Test generally contracted bases for
  bypass/equivalence. Persistent CDERI and metric/native memory are separate.
- [x] Implement the planner and apply a 4GiB default resident-build workspace
  target (configurable through existing `__config__` conventions). Preserve the
  available-memory limit, evaluator shell-pair minimum, CD/ED selection and
  thresholds. Verify actual evaluator batch sizes before allocating scratch;
  do not silently claim that indivisible shell batches fit an impossible budget.
  Inspection found that generally contracted bases require entire angular-pair
  groups in the existing recontractor. Apply the new cap only to the splittable
  non-general-contraction path; test the general path against the control and
  retain its existing allocation policy. For automatic/mapped-host storage,
  account conservatively for both input and transformed output buffers.
  Initial planner/native equivalence tests pass; the additional actual native minimum-shell-batch rejection
  test passes. The combined focused/native/disk group has 20 passing checks.
- [x] Preserve the pre-change source in the existing immutable snapshot. Run
  focused GPU tests for ordinary and generally contracted bases and small
  independent CPU energy/gradient controls before large acceptance.

## 3. Qualify the candidate

- [x] Run candidate Taxol fresh-process phase/memory observation. Require original
  CPU gates (1e-8Eh energy, 1e-6Eh/Bohr max gradient component), identical atomic
  input and unchanged settings. Compare stage peaks to the control.
- [x] Run three alternating fresh-process control/candidate timing pairs using
  the hash-pinned pre-change build function. Memory instrumentation must be off.
  Do not infer speed from diagnostic timings or old intermediate baselines.
- [x] Check disk mode remains unchanged, temporary storage is cleaned, and no
  complete host integral copy is introduced. Leave unqualified speed candidates
  experimental; record failures without relaxing any gate.

## 4. Audit and report

- [x] Independently recompute CPU comparisons, input/source identities, medians
  and peak reductions; request read-only code/results review.
- [x] Document measured improvements and limitations. Distinguish GPU requested
  storage, host RSS, OS cache and native allocations. Preserve prior results as
  the earlier source snapshot, not as results for the modified working tree.

The next exchange change is chosen only after the diagnostic profile identifies
its dominant helper; no simultaneous speculative convergence or metric changes.

## Final qualification

Ten accepted workflow observations pass independent CPU gates. Three alternating
fresh-process timing pairs give 565.263834 s control and 566.881035 s candidate
medians (0.286% more time). Tracked CuPy peak falls 30.100129212 to
20.050847372 GB (33.386% lower); host RSS remains approximately 1.476 GB.
Maximum CPU energy/force errors across all observations are 2.2283e-9 Eh and
2.1083e-8 Eh/Bohr. All large observations record the same atomic density and
20 cycles. Production change in this stage is limited to df/df.py.

20 focused/native/disk checks, 13 bounded-workspace audit checks and 26 exchange
audit checks pass. All 22 fixed-density exchange calls pass the 1e-9 J/K gate;
128/512/1024 blocks are slower than 256 and remain experimental. Final read-only
source/results review found no blockers. The preliminary memory observation and
its exact runner are retained, with explicit exclusion/legacy handling; the
subprocess-launcher failure is retained separately. See resident_workspace_results/
REPORT.md, summary.json and exchange_REPORT.md under benchmarks/single_gpu/.
