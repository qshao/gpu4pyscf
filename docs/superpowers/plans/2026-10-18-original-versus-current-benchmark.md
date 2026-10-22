# Original versus Current GPU Benchmark Plan

**Goal:** Measure complete B3LYP SCF plus full-response nuclear gradients on
small (vitamin C,208AO), medium (tamoxifen,537AO), and large (Taxol,1123AO)
systems, comparing original GPU algorithms at9af748a with current code.

**Architecture:** Freeze both complete Python/source trees, share identical
SM121-compatible native binaries (native numerical kernels are unchanged), and
run one fresh GPU process at a time. Assert actual NWChem pruning/level3,
def2-SVP/universal-JKfit,1e-10/1e-7SCF thresholds, full grid/auxiliary response,
four CPU threads, and identical atomic initial densities. Timing covers molecule
setup/initial guess/fresh DF construction/SCF/full gradient, excluding imports and
report provenance. Three uninstrumented fresh-process timing samples per variant;
separate fresh memory runs capture complete-workflow CuPy allocations and VmHWM.
No converged seed, CDERI reuse, CPU reference calculation during GPU timing, or
component-derived overall speed ratios. Check saved independent CPU references.

Additional large-case current disk mode quantifies the capacity/time tradeoff.
Original code has no disk mode. Report host RSS separately from requested CuPy
allocations; GB10 unified memory prohibits summing them. Native/preexisting
allocations and OS file cache are outside the CuPy ledger. Preserve failed
accuracy/capacity runs explicitly and do not equate nonconverged runs with speed.
Do not touch unknown `record`, commit, or modify production implementations.

- [x] Prepare immutable source/native/geometry/reference manifests and a common
  fresh-process runner under `benchmarks/single_gpu/original_comparison/`.
- [x] Verify runner settings, baseline import isolation, initial-density identity,
  result comparators and memory accounting on the small workload.
  Actual factory tests and audit negative controls:11pass. Small/medium all
  initial-density hashes match; CPU gates pass. Setup-only CPU/GPU pruning
  function binding mismatch was caught before SCF and is archived separately.
- [x] Run original/current small, medium and large resident modes: three timing
  repetitions in alternating order, plus one memory observation each.
  Small and medium complete. Medians5.490→5.278s and70.541→48.070s. Whole-workflow
  CuPy peaks1.357→.526GB and21.105→4.343GB; hostRSS essentially unchanged.
  All both-version SCF histories have18iterations on these two cases.
  Large timing repetitions complete: original median1103.110s, current566.023s;
  all have20iterations. Original forces fail the CPU gate in all three runs;
  current forces pass. The1.949x observed runtime ratio is accuracy unqualified.
  Separate large resident memory observations complete: tracked CuPy peak
  65.620→30.100GB (54.1%lower); hostRSS1.475GB essentially unchanged.
- [x] Run current large disk mode with4GiB staging and zero cache: three timing
  samples and one separate memory observation; verify scratch cleanup.
  All three timing samples pass CPU gates and cleanup; median582.270s.
  Separate memory observation also passes: tracked CuPy peak6.567GB,
  hostRSS3.623GB. This is78.2%less tracked GPU storage than current resident
  mode, with2.87%longer median runtime. OS-cache/RAM-pressure limits remain explicit.
- [x] Independently recompute numerical gates, medians, speed/time ratios,
  allocation/RSS changes and all input/source/native identities. Review the
  harness/results and produce a concise three-size comparison with limitations.
  All28 records audited against immutable inputs;11 harness checks pass.
  Independent review confirms all CPU comparisons, medians, reductions and
  qualification text. Final REPORT.md and summary.json are generated.
  Audit passes; scientific status explicitly retains original Taxol failures.
