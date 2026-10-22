# Exchange layout optimization

**Goal:** Measure the remaining bottlenecks and reduce exchange transformation
cost without changing the quantum method, precision, grids or convergence criteria.

**Approved design:** Reprofile current optimized code; exploit the existing
auxiliary-index-contiguous DF block layout using strided batched GEMM. Compare
against the current GPU implementation, retaining native/custom contraction
backends and unsupported layouts. Promote only after whole-workflow benefit.

**Execution:** Continue inline in the existing `improve-single-gpu` working tree;
preserve all previous results and unrelated files. No commit or publication.

- [x] Profile current tamoxifen PBE/B3LYP, including the promoted SYRK helper.
  Reuse validated CPU references after checking geometry, settings, library and
  native binary identities. Record synchronized inclusive timings separately
  from an uninstrumented control.
- [x] Write and run failing tests for the layout-aware transformation. Compare
  odd dimensions, batches, empty dimensions and stream behavior against NumPy.
  Implement a benchmark-only helper and verify those tests.
- [x] Benchmark actual DF memory strides at 208, 512 and 537 AOs, with full/partial
  auxiliary blocks and both converged and initial density ranks. Record copies
  and medians from alternating samples.
- [x] If components improve, compare reversible substitution against current
  optimized defaults for B3LYP energies plus gradients, warming both variants
  and alternating three pairs. Gate every run against the matched CPU reference.
- [x] Promote only a measured winner with dispatch tests and preserved backends;
  otherwise retain the experiment without changing production. Run relevant
  regression checks, review evidence and document results and remaining limits.

**Result:** Guarded strided GEMM promoted. Experimental large-case speedup
1.168x; separately measured production speedup 1.179x against the previous
optimized GPU baseline (49.539 to 42.005 seconds, three pairs). Profiled
transformation time 16.464 to 9.149 seconds, including fallback blocks. All 31
before/experiment/acceptance energy-gradient records passed unchanged CPU
gates. Targeted tests: 72 passed, known baseline fingerprint excluded. Molecular
qualification: 8 passed. Source/native/report hashes and saved baseline replay
verified; independent review found no material issue. Work remains uncommitted.
