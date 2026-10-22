# Fine stage profiling implementation plan

**Goal:** Resolve the measured PBE XC and large-hybrid DF J/K costs into
actionable sub-stages without changing production algorithms.

**Architecture:** Add a benchmark-only context manager around the existing
coarse profiler. Attribute XC work using thread-local scopes so worker-thread
SCF calls and main-thread gradient calls remain distinct. Time generator
advancement rather than generator creation for DF block retrieval. Restore
all patched bindings. Store raw timings, call counts, contraction shapes,
source fingerprints, and controls separately from prior campaigns.

**Tech stack:** Existing pytest/CuPy/PySCF molecular harness on one GB10.

- [x] Add `benchmarks/single_gpu/test_fine_stage_profile.py`: synchronize around
  AO, density, functional, assembly helpers; DF block retrieval/unpacking;
  expression-specific SCF and gradient contractions; derivative integrals and
  metric solver construction/application. Treat nested timings as inclusive.
- [x] Reuse the saved, geometry-hashed tamoxifen def2-SVP CPU references for
  PBE/B3LYP only after checking geometry, settings, source and numerical identity.
  Run a GPU warmup, three controls and one instrumented run per method. Compare
  every run with its reference using unchanged tolerances. Print phase progress.
- [x] Execute with
  `/home/qshao/miniconda3/envs/dft_oracle/bin/python -m pytest benchmarks/single_gpu/test_fine_stage_profile.py -q -s --tb=short`.
  Require convergence, matching iterations, finite stage durations, observed
  targets and restored bindings; record overhead against the control median.
- [x] Audit dominant stages and document results in the existing benchmark
  README. Select the next specific experiment from the evidence. Do not change
  default block sizes, precision, screening or scientific approximations.

The user authorized proceeding with finer profiling; execute inline in the
existing development branch and preserve all unrelated working changes.

Follow-up experiment selected from the PBE result: MO-density construction
occupied 4.64 seconds, versus 0.034 seconds for XC functional evaluation. Test
four batched AO-to-MO products and the existing fused GGA density-reduction
kernel in `benchmarks/single_gpu/test_mo_density_experiment.py`. Keep the
candidate benchmark-only. Check numerical/buffer/stride/empty-occupation cases,
then five alternating original/candidate whole-workflow pairs for Vitamin C
and tamoxifen PBE, including CPU gates and identical iteration counts.
Command after profiling finishes:
`/home/qshao/miniconda3/envs/dft_oracle/bin/python -m pytest benchmarks/single_gpu/test_mo_density_experiment.py -q -s --tb=short`.
An unsuccessful candidate remains an explicit measured rejection; no default
or production path changes follow without demonstrated benefit.

Completed: fine profiling passed in 399.49 seconds; candidate component and
paired workflow tests passed separately in 1.27 and 269.08 seconds. The fine
profile identifies two exchange contractions totaling 28.462 seconds. The
GGA candidate regressed Vitamin C by 2.90% and improved tamoxifen by 0.82%;
retain it benchmark-only. All numerical gates and restoration checks passed.
Raw and derived reports plus detailed interpretation are in the existing
benchmark directory/README. Source/report fingerprints were verified.
No production implementation was changed during this work.
