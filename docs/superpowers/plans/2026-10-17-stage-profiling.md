# SCF/gradient stage profiling and scaling

The user approved proceeding from complete-workflow benchmarks to profiling and
larger system/basis tests. Execute inline without production changes.

1. Reuse matched molecular construction/settings and CPU comparisons.
2. Measure PBE/B3LYP on Vitamin C def2-SVP (208 AOs), Vitamin C def2-TZVP
   (420 AOs), and tamoxifen def2-SVP (537 AOs).
3. For each case run CPU reference, GPU warmup, three plain controls, and one
   instrumented workflow. Retain all results and strict numerical gates.
4. Temporarily wrap stable SCF J/K, XC, eigensolver and DF build entry points;
   wrap gradient one-electron, XC full response and DF J/K total/core stages.
   Restore all bindings on exit. Inclusive nested timings must not be summed.
5. Report instrumented/control elapsed ratio, control spread and iteration counts.
   Instrumentation is diagnostic, not a production speedup measurement.
6. Review findings, save source-fingerprinted raw data, and select the next
   optimization from observed costs. Do not change numerical approximations,
   grid response, precision, or default block size to manufacture a speedup.

Completed: the six-case matrix passed in 1056.25 seconds. All 24 timed GPU
calculations met the preset CPU energy/force gates; all warmups passed too.
Raw data and derived summary are in `benchmarks/single_gpu/stage_profile_results.json`
and `stage_profile_summary.json`, with interpretation and priorities in that
directory's README. Source hashes were checked against the measured files.
XC is the primary PBE target; DF J/K is the primary larger-hybrid target.
No production implementation was changed during this profiling campaign.
