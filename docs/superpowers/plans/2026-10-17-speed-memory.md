# Speed and memory for larger systems

**Goal:** Reduce peak memory as well as elapsed time without changing the
quantum method, precision, grid, screening or convergence thresholds.

**Approved scope:** Continue from the qualified exchange-layout implementation.
Test direct Coulomb GEMV and gradient GEMM, then bounded gradient workspaces.
Large-system memory takes priority over increasing auxiliary block sizes.
Retain existing backends and unsupported paths. Do not cache additional full
integral tensors. Keep prior results and the unrelated `record` file intact.

- [x] Save the current optimized baseline sources and fingerprints.
- [x] Write failing mathematical/alias/alpha-beta tests; implement benchmark-only
  Coulomb GEMV and gradient direct-GEMM candidates; compare actual-stride timings.
- [x] Measure current peak live/reserved CuPy memory and host process RSS in a
  fresh process. On unified-memory GB10 these quantities must not be summed.
- [x] Test bounded gradient auxiliary work blocks and Hermitian-only smaller
  scratch storage, retaining all auxiliary/grid response terms. Compare memory
  in fresh processes and speed separately without memory instrumentation.
- [x] Benchmark promising combinations against the current approximately
  42-second B3LYP baseline using alternating pairs and unchanged CPU gates.
  Report an observed 20% gain only if achieved; reject speed/memory regressions.
- [x] Promote qualified changes with regression coverage, independent review
  and evidence hashes. Document limits for system size and hardware.

The intended large-system atom/AO count and host/GPU memory budget were requested
as optional clarification. Work on the existing GB10 reference continues while
that workload is specified. Do not treat the current small-grid benchmark as
proof that an arbitrary large calculation fits in memory.

Production outcome: large B3LYP median 42.475 to 38.867 seconds over three
alternating pairs (1.093x; 8.49% less time), below the requested extra 20%.
Fresh-process tracked live CuPy allocations fell from 19.64 to 4.03 GiB
(79.48% less); process RSS was essentially unchanged. No precision, method,
grid, screening or convergence settings changed. Peak allocations exclude
native/library storage; the configured workspace budget is not a total cap.

Validation: 98 targeted tests passed (one established upstream sparse-index
fingerprint deselected), eight fresh molecular tests passed, production memory
and performance qualification passed. An additional three baseline/compact/
production diagnostic replays preserve the strict legacy gradient fingerprint.
Independent code review found no material issues. Evidence and source hashes
are checked by benchmarks/single_gpu/summarize_speed_memory.py and retained in
speed_memory_summary.json. All changes remain local and uncommitted.

Remaining capacity work: stream or recompute persistent DF tensors; current
host fallback uses RAM, not disk. Remaining speed work: exchange transforms and
three-center derivatives, retaining the smaller scratch buffers. Real larger
chemical workloads and production grids remain to be qualified once specified.
