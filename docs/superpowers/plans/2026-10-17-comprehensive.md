# Comprehensive single-GPU qualification and occupied-pair gradients

**Goal:** Broaden molecular size/basis/grid qualification and test a substantial
exact-symmetry gradient optimization, preserving FP64 and scientific gates.

**Approved design:** The user approved broad benchmarks followed by targeted
kernel/algorithm work. Execute in the existing development branch, preserve all
previous reports, keep experimental disk caching off, and do not commit/publish.
Use the planning, execution, TDD and verification workflows without another
permission round. GPU tests run one at a time. Independent CPU-reference jobs
may overlap each other, but not final GPU timing measurements.

## Benchmark campaign

Create `comprehensive.py` and `test_comprehensive.py` under
`benchmarks/single_gpu/`. Inputs: Vitamin C SVP PBE/B3LYP, tamoxifen SVP
PBE/B3LYP, tamoxifen TZVP B3LYP, Taxol SVP B3LYP, valinomycin SVP PBE/B3LYP.
These span208–1542 spherical AOs. Also inventory Taxol TZVP2228 and
valinomycin TZVP2958 AOs as capacity estimates only, not completed calculations.
Use level3 NWChem-pruned grids, conv_tol1e-10/conv_tol_grad1e-7, full grid and
auxiliary response, same geometry/functional/basis between CPU and GPU, 4CPU
threads and max_cycle100. CPU/GPU energy/force gates stay1e-8Eh/1e-6Eh/Bohr.

- [x] Snapshot current Python/native identities and all affected full sources.
- [x] RED/GREEN tests for case registry, exact configured grid/pruning, unknown
  case/mode rejection, and fingerprints. Never mutate historical molecular.py.
- [ ] Generate isolated CPU references once per case, recording failures and
  preserving completed records if interrupted. CPU timing is not a speed target.
- [ ] Run GPU controls and instrumented profiles with no CPU-reference contention.
  Keep timing, allocation instrumentation and inclusive profile scopes separate.
  Force GPU-resident CDERI when exchange requires it, fail explicitly if it
  cannot fit rather than silently allocating a full host tensor.
- [ ] Independently validate geometry/settings/native/source identities, CPU
  comparisons and scaling. Benchmark level3 is not proof of grid convergence.

### Scope adjustment after measured reference cost

The first four CPU oracles completed. Taxol's CPU reference was still in SCF
after734.69seconds and was deliberately interrupted and archived, with no
completed data lost. The tamoxifen TZVP CPU oracle continues as a representative
case above1000 AOs. A scope question offered full CPU references versus
representative CPU plus original-GPU differential comparisons; after a reasonable
reply interval, proceed with the latter stated default. No full CPU qualification
is claimed for Taxol/valinomycin unless those references subsequently complete.

Use `capacity.py` for four larger cases, comparing unchanged original GPU
gradients to the benchmark-only packed candidate. Full-workflow differential
gates are1e-10Eh/1e-9Eh/Bohr plus equal SCF cycles; independent CPU gates remain
separate. Every allocation observation uses a fresh process. Taxol/tamoxifen
TZVP initially use resident CDERI; valinomycin uses disk CDERI with an explicit
4GiB staging budget and no cache. Pure PBE may bypass CDERI construction entirely;
report actual storage rather than implying that disk streaming was used.
Keep CPU-oracle matrix status partial while those optional oracles are deferred.

## Exact occupied-pair candidate

Create a benchmark-only candidate and mathematical tests before promotion.
Eligibility is realFP64/hermi1 with identical factor objects (`left is right`),
with a large-matrix threshold to preserve existing small fingerprints. Store
`B[L,i,j]` for `i>=j` only. Metric multiplication applies independently to each
packed column. Unpack only a bounded auxiliary block for AO transformations.
Compute diagonal traces from packed diagonal indices. The auxiliary-response
Frobenius Gram is exactly `2*P@P.T - D@D.T`, where P holds the lower triangle
and D its diagonal columns; use raw FP64 SYRK on C-order buffers to avoid a
full packed transpose/copy. Retain generic paths for nonsymmetric factors.

- [x] RED/GREEN NumPy-oracle tests for packing, trace, metric application,
  weighted Gram alpha/beta, aliases/layout/type guards and restored bindings.
- [x] Replay saved full gradient against candidate on the qualified537-AO case,
  then production-grid and larger cases. Compare forces/iterations and memory
  in fresh processes; do not retain tensor owners in instrumentation.
- [x] If unchanged gates pass and memory/speed results justify it, promote
  narrowly, including all auxiliary/grid responses and open-shell fallbacks.
  Otherwise retain the candidate as a documented rejected experiment.
- [ ] Run relevant regressions, paired workflow acceptance, independent review
  and report/source hash QA. Document remaining limits and rejected candidates.

Alternatives: fused exchange kernels may improve memory traffic, but the
remaining GEMM/SYRK arithmetic may dominate on GB10. Do not implement a large
custom kernel before profiling larger cases. Finer integral shell-pair batching
is a separate capacity project if build-workspace minima dominate.

## Additional measured candidate: coalesced deterministic recontraction

Inspection while larger CPU oracles run found that `_create_pair_recontractor`
uses atomicAdd with one block per primitive contribution. The native source also
contains an unused gather kernel, but its warp lanes read different input rows
at a fixed auxiliary column. Test a benchmark-only CuPy RawKernel with adjacent
threads reading adjacent auxiliary columns; each thread sums one output element
in a fixed CSR contribution order. Keep FP64 and disable FMA contraction to
match separate multiplication/addition semantics. No screening/precision change.

- [x] Independent NumPy oracle, repeated-index/sign/zero/empty tests, explicit
  nondefault-stream and repeatability checks before component timing.
- [x] Compare atomic and coalesced kernels using actual recontraction layouts;
  warm both and record seven alternating timings and source/native identities.
- [ ] Only if worthwhile, compare full molecular workflows and original CPU
  oracles before production promotion. Determinism alone does not diagnose the
  previous experimental-cache iteration regression; do not assert that cause.

Actual factory tests confirm cc-pVDZ uses recontraction, while the tested
def2-SVP basis uses the existing no-recontraction path. This candidate therefore
cannot explain the earlier def2-SVP cache iteration regression. Candidate CSR
metadata is resident on the GPU for all batches; baseline tables remain on CPU
and transfer per batch. Account for both before making any RAM claim.

## Measured outcomes and promotion decision

Five independent CPU/GPU full energy/force comparisons pass at level3:
Vitamin C SVP PBE/B3LYP, tamoxifen SVP PBE/B3LYP and tamoxifen TZVP B3LYP.
The first four also have profiles and fresh-process memory observations.
The1042-AO TZVP case has a fresh-process original-GPU allocation run and saved
orbitals; its force error is4.57e-7Eh/Bohr. No repeated speed acceptance is claimed
for the new large cases.

The independent Taxol CPU reference completed from GPU-seeded initial density,
with CPU recomputation, three SCF cycles and full-response forces. It took
408.03s SCF and782.01s gradients using16CPU threads. Both resident baseline and
disk/packed GPU energies pass1e-8Eh; forces fail1e-6Eh/Bohr at2.54e-5 and2.48e-5.
This is a preexisting-baseline qualification limit as well as a candidate limit.
Same-orbital full-grid XC diagnostics agree within1.84e-11Eh/Bohr, narrowing the
next checks to DF J/K derivatives, metric application and SCF-state sensitivity.
Original-gradient profiling matches its saved replay within5.67e-13 and records
81.58s J/K plus39.12s XC (inclusive diagnostic timings). Tighter SCF diagnostics,
if needed, must use separate honest settings; do not waive failed full workflows.

Packing V1 reduced537-AO new gradient-J/K storage10.4% and whole allocations1.25%.
The1123-AO combined disk/packing run failed the strict full-workflow gate at
1.77e-6Eh/Bohr. Original full-gradient replay at identical saved orbitals differs
by2.37e-9, while original J/K repeats within4.52e-13. Averaging orientations in
V2 still fails the extra1e-9 matched-orbital target (2.64e-9). Keep both versions
benchmark-only; preserve failed runs and identify archivedV1 sources explicitly.
No production promotion or new whole-workflow20% speed claim is justified.

The1542-AO valinomycin bounded disk J/K probe passed: actual CDERI31.52GB,
CuPy tracked peak4.52GB, staging within4GiB, selected NumPy J/K errors below
1.4e-12, temporary-file cleanup confirmed. This is capacity/contraction evidence,
not converged large-system SCF/force qualification. Full valinomycin CPU oracles,
full force qualification, repeated candidate timings and the optional Taxol
full-host probe remain deferred. The eight-case matrix stays partial.

The coalesced recontraction and large short-block exchange candidates remain
unpromoted. Their component ratios do not establish molecular speed gains.
Next speed work should target dominant J/K and XC scopes using these profiles;
next capacity work should bound gradient occupied-pair storage without violating
the numerical gates. Independent read-only review confirmed holding promotion
and required raw baseline-hash checks in the summary validator.

Final focused verification:43 GPU candidate/disk/memory/snapshot tests and13
CPU configuration/provenance tests passed. Small and1542-AO disk J/K probes,
1042-AO baseline, independent seeded Taxol CPU reference, original-gradient
replay/profile and both matched-density XC captures also completed. These passes
do not change the failed Taxol full-force comparisons or strict packed-gradient
experiments. `summarize_comprehensive.py` verifies current/archived source and raw
baseline identities, recomputes numerical comparisons and retains partial matrix
status. No candidate was promoted during this comprehensive stage.

## Taxol accuracy resolution (2026-10-18)

The user subsequently requested resolution of the force discrepancy. Independent
CPU derivatives at the identical saved GPU orbitals AND orbital energies locate
the2.48e-5Eh/Bohr error in DF J/K. CPU forces at that state differ from the
separately converged CPU reference by only3.72e-9; full-grid XC matches1.84e-11.
The auxiliary metric has condition estimate3.41e11. Forming the projected
inverse-like metric before applying occupied-pair right-hand sides amplifies
roundoff. Direct factor-based application reduces J/K error to1.82e-9.

- [x] Preserve pre-fix sources and immutable failed reports.
- [x] Reproduce the large error against matched-state independent CPU components.
- [x] Verify factor-based metric application against the CPU derivative oracle.
- [x] Apply blocked `C @ solve(J, C.T @ B)` to RHF and both UHF spin slices,
  retaining FP64, existing CD/ED choice/cutoff, basis and screening.
- [x] Add independent70-digit Decimal regression for an exactly factorable,
  nearly dependent metric. Confirm the former inverse approach fails it while
  the correction passes1e-12; test both decompositions beyond256 RHS columns.
- [x] Full resident SCF/gradient acceptance at original thresholds passes:
  energy2.2246e-9Eh, force6.9793e-9Eh/Bohr.
- [x] Full disk SCF/gradient acceptance at original thresholds passes:
  energy2.2228e-9Eh, force6.9954e-9Eh/Bohr. Disk staging remains within4GiB,
  cache zero, cleanup verified. Resident/disk force difference9.79e-10.
- [x] Independent read-only review; raw/source/seed/native fingerprints and
  scientific gates recomputed by `summarize_accuracy.py`.

These are full SCF and full-response gradient calculations from saved orbitals
used only as initial density, with fresh integral/Fock/grid setup and mandatory
SCF convergence. Timings are not comparable to previous cold20-cycle runs.
No tighter-SCF workaround is used: a separate stationarity/refinement diagnostic
was interrupted once matched-state CPU derivatives established the J/K cause;
its failed/KeyboardInterrupt record is preserved and is not a passing test.
Occupied-pair V3 is adapted to the corrected production solver but remains an
unpromoted benchmark candidate. The historical eight-case matrix remains partial;
the dedicated Taxol accuracy summary passes both storage modes.

Final correction verification:52 GPU regression tests (including restricted and
unrestricted derivatives, finite differences, range separation, Cartesian basis,
disk round trips/Hessian/cleanup, memory ownership, orbital replay and packed
candidate compatibility) plus13 CPU configuration/provenance tests pass.
Both independent full Taxol storage-mode acceptances pass separately. An
additional negative-control replay confirms the former inverse approach fails
the new high-precision regression. No acceptance tolerance was relaxed.
