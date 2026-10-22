# Single-GPU Qualification

## Comprehensive code review and repair (2026-10-18)

The [review report](code_review_results/REPORT.md),
[source findings](code_review_results/REVIEW.md) and
[independent audit](code_review_results/summary.json) cover the current repairs.
They fix default package naming, singleton Coulomb BLAS layout, noncontiguous
metric writeback and raw-BLAS pitch eligibility. Eligible matrix products reuse
their output; grid-response weighting consumes and releases its fresh batch.

The matrix component ratios are 1.104x at 64×64 and approximately 1.006x for the
tested larger real/complex matrices, with no extra result tensor. Large grid
weighting reduces additional requested temporary allocation from 88.9 MB to
2.7 KB at an observed 2.6% component time increase. No overall workflow speedup
is inferred. A slower binary-einsum alternative is archived and rejected.

178 regression checks and 30 audit checks pass. Fresh medium/Taxol SCF and full
forces pass independent CPU gates at unchanged scientific settings. Component
timings compare against the source-only optimized entry snapshot, not original
HEAD. Earlier campaigns below remain historical source identities; their
live-tree freshness checks can reject the subsequently repaired working tree.

    python -m pytest benchmarks/single_gpu/test_code_review_regressions.py benchmarks/single_gpu/test_code_review_acceleration.py -q
    python -m pytest benchmarks/single_gpu/test_code_review_audit.py -q
    python benchmarks/single_gpu/summarize_code_review.py

## Bounded resident DF build workspace (2026-10-18)

Resident density-fitting construction now limits its splittable build buffers
with `gpu4pyscf.__config__.df_cderi_build_workspace_bytes` (default 4 GiB).
Set the value before constructing DF integrals, for example:

```python
from gpu4pyscf import __config__
__config__.df_cderi_build_workspace_bytes = 4 * 1024**3
```

This budget applies to bases without general contractions. It bounds each
worker's integral-input buffer and, when automatic storage can select mapped
host memory, its additional transformed-output buffer. Forced GPU storage writes
transformed values directly into persistent CDERI. The budget excludes persistent
CDERI, metric/factors, other arrays, native-library workspaces and allocator
reservation. Generally contracted bases retain their existing angular-group
batching; disk CDERI uses its separate staging budget. Worker budgets do not form
a total multi-GPU or process RAM limit. An indivisible shell-pair batch that
exceeds the configured budget raises an error.

The large Taxol workspace-cap, memory and timing campaign explicitly uses
`with_df.use_gpu_memory=True`, forcing resident GPU CDERI. Automatic storage
(`None`) conservatively budgets two buffers and can select mapped host storage;
its full-workflow timings are outside this campaign.

The single-GB10 Taxol memory qualification reduced whole-workflow peak tracked
requested CuPy storage from 30.100129212 GB to 20.050847372 GB (decimal GB).
The gradient peak remained 16.669080260 GB. The candidate passes the independent
CPU energy/force gates, with maximum force error 4.9844e-9 Eh/Bohr; full small
and medium workflow validations also pass. Physical settings, FP64 arithmetic,
screening, convergence thresholds and full grid/auxiliary response are unchanged.
The [resident workspace report](resident_workspace_results/REPORT.md) and
[audited data](resident_workspace_results/summary.json) cover ten accepted
workflow observations. Three alternating fresh-process timing pairs, separate
from memory instrumentation, give control/candidate medians of
565.264/566.881 s: 0.286% more time for the lower-memory candidate. The control
is the earlier optimized, accuracy-corrected GPU implementation before the cap.
Host address-space peak RSS remains approximately 1.476 GB. Across all workflow
observations, maximum CPU energy/force errors are 2.2283e-9 Eh and
2.1083e-8 Eh/Bohr, within unchanged 1e-8/1e-6 gates. All large runs record
20 SCF cycles from the identical atomic density. Twenty focused/native/disk
checks and thirteen bounded-workspace audit checks pass.

The preliminary candidate memory run is retained but excluded from the final
statistics because its runner predated the explicit runtime-budget assertion.
Its exact runner is accepted only for the individually pinned pre-change control
profile. All remaining accepted observations assert and record the 4 GiB target.
The subprocess-launcher CUDA initialization failure is retained separately;
directly launched workflows supply the accepted results.

The allocation ledger measures requested live storage allocated through the
wrapped CuPy allocator, including retained arrays across phase boundaries and
worker-thread allocations. It excludes native and preexisting allocations.
Named phase peaks are inclusive and must not be summed. Requested CuPy storage,
pool reservation and process RSS are distinct metrics; on GB10 unified memory,
do not add them or interpret any one as total physical RAM.

The [resident exchange sweep](resident_workspace_results/exchange_REPORT.md)
compared auxiliary block sizes 128, 256, 512 and 1024 at fixed density. All tested
alternatives were slower than 256, which remains unchanged. Those J/K component
timings and workspace estimates do not establish molecular workflow gains or a
hardware-wide memory bound. All 22 calls pass the 1e-9 full-matrix J/K gate;
26 independent sweep-audit checks pass.

Next kernel candidates are a packed-CDERI occupied transform that skips existing
screened zero entries (about 45% for Taxol), preserving the current screening,
and an alternative `B[i,L,j]` layout for larger GEMMs. Both are unmeasured and
have no production promotion. In the diagnostic baseline, occupied transforms
and Gram operations together take about 284 s of a 569 s workflow. A 20%
workflow time reduction would therefore require roughly 40% less combined
transform/Gram time if all other costs remained fixed.

## Original-source complete-workflow comparison (2026-10-18)

Completed results: [comparison report](original_comparison/REPORT.md) and
[audited data](original_comparison/summary.json), covering 28 observations and
11 harness checks. All tested current calculations pass CPU accuracy gates;
original Taxol forces fail and its runtime ratio is explicitly qualified.

The campaign in `original_comparison/` compares the original GPU Python
implementation at `9af748a71ca8f4881c604332c2f0cc1632e033ac` with an immutable
earlier working-tree snapshot labeled `current` within that campaign. This
snapshot predates the bounded resident DF build changes now in `df.py` and does
not represent today's working tree. Both snapshots use the same
GB10-compatible native libraries. This baseline differs from the intermediate
development baselines described below.

Vitamin C (20 atoms, 208 AOs), tamoxifen (57 atoms, 537 AOs), and Taxol
(113 atoms, 1123 AOs) use B3LYP/def2-SVP, universal-JKfit, level-3 NWChem grids,
unchanged convergence thresholds, and full grid/auxiliary gradient response.
Each fresh process starts from the same CPU atomic initial density and rebuilds
DF integrals. Timings include setup, SCF and gradients. Three timing processes
per variant are separate from allocator-instrumented memory observations; all
GPU jobs run sequentially. Taxol additionally exercises disk storage with a
4 GiB staging budget and zero integral cache.

`summarize_original_comparison.py` requires all 28 observations, verifies frozen
source/native/geometry/reference identities, and independently recomputes CPU
accuracy gates and timing/memory statistics before producing `REPORT.md` and
`summary.json`. Failed original accuracy checks remain visible and disqualify
an equivalent-accuracy speedup claim. Requested CuPy allocations and host VmHWM
are reported separately; neither is total physical RAM on this unified-memory
device. Disk staging limits also exclude OS file cache and other workspaces.

The original campaign validator requires its earlier live-tree source identity;
its freshness check rejects the post-cap working tree. Its frozen reports remain
the historical comparison. Use `summarize_resident_workspace.py` to audit the
new cap measurements against the current tree.

    python -m pytest benchmarks/single_gpu/test_original_comparison_config.py benchmarks/single_gpu/test_original_comparison_summary.py -q
    python benchmarks/single_gpu/summarize_original_comparison.py

## Taxol force accuracy correction (2026-10-18)

The Taxol force discrepancy was caused by explicitly forming the projected
inverse-like auxiliary metric before applying it to the occupied-pair tensor.
Its condition number is approximately3.4e11. At identical orbitals and orbital
energies, independent CPU J/K derivatives exposed a2.48e-5Eh/Bohr error; full-grid
XC agreed within1.84e-11 and the CPU SCF-state effect was only3.72e-9.

Restricted and unrestricted production gradients now apply
`C @ solve(J, C.T @ B)` in blocks of256 right-hand-side columns. This preserves
FP64, native GPU integrals, CD/ED selection and linear-dependence cutoff, basis,
screening and full auxiliary/grid response. It uses the existing GPU solvers,
with bounded projection temporaries. No CPU metric fallback or looser tolerance
is introduced. The matched-orbital J/K error falls to1.82e-9Eh/Bohr.

Full resident and disk SCF/forces pass the independent CPU reference with maximum
force errors6.98e-9 and7.00e-9Eh/Bohr, respectively, and energy errors2.22e-9Eh,
at the original SCF thresholds
1e-10Eh/1e-7 and original acceptance gates1e-8Eh/1e-6Eh/Bohr. Saved orbitals are
only the initial density; native DF integrals, Fock and grids are rebuilt and SCF
must converge. Resident/disk forces differ by9.79e-10Eh/Bohr. Disk staging stays
within4GiB, experimental cache remains zero, and temporary files are removed.
See `accuracy_results/summary.json`
and run `python benchmarks/single_gpu/summarize_accuracy.py` to independently
recompute comparisons and verify source, seed, native and raw-report identities.

    python -m pytest benchmarks/single_gpu/test_metric_application.py -q
    python -m pytest 'benchmarks/single_gpu/test_taxol_accuracy_acceptance.py::test_taxol_accuracy[resident]' -q -s
    python -m pytest 'benchmarks/single_gpu/test_taxol_accuracy_acceptance.py::test_taxol_accuracy[disk]' -q -s

Reports refuse overwrite; archive them explicitly before rerunning. The new
metric regression includes an independent70-digit Decimal oracle for a nearly
dependent matrix and CD/ED cases crossing the256-column boundary. Previous
production sources and occupied-pair V2 are retained under
`runs/2026-10-18-pre-accuracy/`. Failed historical reports remain failed; current
occupied-pair V3 uses the corrected solver but remains benchmark-only. The
eight-case scaling matrix is still partial. These checks establish accuracy,
not a new speedup or hardware-wide RAM bound.

## Production-grid expansion and occupied-pair experiment

The new `comprehensive.py` matrix uses level-3 NWChem-pruned grids, four CPU
threads, full grid/auxiliary response and unchanged energy/force gates. The
eight cases span208–1542 AOs; larger2228/2958-AO basis inventories are estimates,
not qualified calculations. Level3 is a benchmark setting, not a grid-convergence
study. Experimental CDERI caching is disabled and exchange CDERI is forced to
GPU memory so capacity failures cannot silently become full-host allocations.

Run explicit node IDs, one GPU job at a time. Run every memory node in its own
fresh Python process. `-k gpu` also matches the directory name and therefore
does not select only the GPU backend. Examples from the repository root:

    python -m pytest 'benchmarks/single_gpu/test_comprehensive.py::test_case[cpu-tamoxifen_svp_b3lyp]' -q -s
    python -m pytest 'benchmarks/single_gpu/test_comprehensive.py::test_case[gpu-tamoxifen_svp_b3lyp]' -q -s
    python -m pytest 'benchmarks/single_gpu/test_comprehensive.py::test_case[profile-tamoxifen_svp_b3lyp]' -q -s
    python -m pytest 'benchmarks/single_gpu/test_comprehensive.py::test_case[memory-tamoxifen_svp_b3lyp]' -q -s

Reports under `comprehensive_results/` are never implicitly reused or overwritten.
Archive an existing report explicitly before a rerun. CPU references can overlap
other CPU references when RAM permits, but must finish before final GPU timing.
Profiles contain inclusive synchronized scopes; do not sum nested timings.

`occupied_pair.py` is a benchmark-only exact-symmetry gradient candidate. It
stores one triangle of symmetric occupied-pair tensors, applies the unchanged
metric to each packed column, and restores `2*P@P.T-D@D.T` for auxiliary response.
It unpacks only a bounded auxiliary block. Eligibility is real FP64, hermi1,
identical factor objects, CuPy fallback, at least512 AOs and64 occupied columns.
Generic response densities and open-shell code retain their existing paths.
Mathematical equivalence does not imply bitwise equality; unchanged scientific
gates are required. Persistent occupied-pair storage is approximately halved;
whole-workflow memory depends on other tensors and must be measured separately.

    python -m pytest benchmarks/single_gpu/test_occupied_pair.py -q
    python -m pytest 'benchmarks/single_gpu/test_occupied_pair_workflow.py::test_workflow[memory-packed-tamoxifen_svp_b3lyp]' -q -s
    python -m pytest 'benchmarks/single_gpu/test_occupied_pair_acceptance.py::test_paired[tamoxifen_svp_b3lyp]' -q -s

Workflow memory reports measure both whole-workflow requested CuPy storage and
new allocations inside gradient J/K. They exclude preexisting/native storage;
process RSS is separate. Paired acceptance warms both versions and alternates
three pairs with matching iteration counts. A single control or an allocation
instrumented run cannot establish a speedup. Do not promote the candidate until
scientific, memory and repeated timing qualification is complete.

The initial four CPU controls have completed; an independent CPU reference also
completed for tamoxifen/def2-TZVP(1042 AOs). A separate Taxol CPU reference uses
saved GPU orbitals only as its initial density, then independently recomputes
CPU integrals, converges SCF and evaluates full-response gradients. It uses16
CPU threads; comparisons exclude only that execution setting and retain all
physical settings and original tolerances. Its status is recorded separately in
`taxol_svp_b3lyp_seeded_cpu.json`. Full valinomycin CPU references remain deferred.
Taxol's interrupted cold CPU record is preserved under
`runs/2026-10-17-comprehensive-cpu-deferred/`. The original eight-case independent
CPU matrix therefore remains partial; do not describe it as fully qualified.

`capacity.py` adds fresh-process larger-case comparisons against the full saved
pre-comprehensive GPU implementation. Here “original GPU” means that archived
development-stage baseline, including earlier optimizations; it is not a newly
built upstream release. It compares unchanged quantum settings, convergence
cycles, energy and all force components. CPU comparisons are recorded separately
when available. `test_capacity.py::test_taxol_combined_disk_packed` compares
resident baseline against disk plus packed gradients, so any whole-workflow
memory reduction combines the two changes. The gradient J/K allocation scopes
isolate the occupied-pair change after CDERI has been released.

Historical occupied-pair variants V1/V2 remain benchmark-only. The first Taxol combined
disk/packing full run failed its strict force-preservation gate, despite close
energies and equal cycle counts. Original full-gradient replay at the identical
saved orbitals reduces the difference to approximately2.4e-9Eh/Bohr. Separately
converged GPU states differed much more; the subsequent CPU decomposition located
the instability in metric application, as described above.
That replay does not erase the failed workflow result. The second variant
averages both pair orientations before packing, but still fails the additional
1e-9 fixed-orbital gate. Failed raw reports retain their failed status. Earlier
537-AO memory measurements used the archived lower-triangle V1 source, not the
archived averaging V2. `summarize_comprehensive.py` verifies the source identities,
labels them separately and recomputes CPU comparisons when references exist.

A streaming-buffer budget bounds staging buffers, not total process or GPU RAM.
AO workspaces, occupied-pair response tensors, density matrices, native allocations
and global OS file cache have separate costs. On GB10's unified memory, do not add
the CuPy allocation ledger to process RSS or treat either as hardware-wide RAM.

The completed Taxol seeded CPU reference exposed a pre-correction scientific limit:
both archived resident GPU baseline and disk/packed workflow exceeded the original
1e-6 force gate (maximum errors approximately2.54e-5 and2.48e-5Eh/Bohr), while
energies passed1e-8Eh. Neither archived Taxol workflow was independently force-qualified.
The corrected production acceptance is reported above. Do not attribute the original discrepancy solely to packing or claim
accuracy preservation from energy agreement. The1042-AO tamoxifen TZVP baseline
does pass independently (force error4.57e-7Eh/Bohr). Its timings and allocations
were instrumented and partly overlapped CPU work; no new speed ratio is claimed.

The1542-AO valinomycin disk J/K probe passes selected NumPy contractions with
maximum errors1.31e-12(J) and1.67e-13(K). Actual stored CDERI is31,524,357,312bytes;
tracked CuPy peak is4,521,525,304bytes, process VmHWM3,480,666,112bytes, and
maximum device staging3,934,435,200bytes within the4GiB staging budget. These are
separate memory measures for one J/K evaluation, not converged forces or a total
RAM cap. Temporary files are verified removed. The full eight-case qualification
matrix remains partial. See `comprehensive_summary.json` for independently
recomputed comparisons and immutable raw-report hashes.

`test_taxol_xc_diagnostic.py` compares only the full-grid XC force contribution
on CPU/GPU at the identical saved density, independently of SCF and DF J/K.
A discrepancy there locates the combined XC/grid/assembly path, not necessarily
the functional evaluation alone. The completed Taxol comparison agrees within
1.84e-11Eh/Bohr, making XC arithmetic at identical density an unlikely cause of
the full-force failure. Subsequent matched-state CPU J/K derivatives isolated
metric application as the cause. This component agreement does not qualify
either failed full workflow. `test_taxol_original_profile.py` replays the
original full gradient and requires agreement with its saved original replay
within1e-9. Its inclusive diagnostic profile gives81.58s DF J/K and39.12s XC;
these are gradient-only scopes, not a complete SCF speed benchmark. The full
resident Taxol baseline spent433.42s in SCF and119.26s in gradients, so an overall
20% reduction requires SCF work as well as gradient optimization.

    python -m pytest 'benchmarks/single_gpu/test_taxol_xc_diagnostic.py::test_fixed_orbital_xc[cpu]' -q -s
    python -m pytest 'benchmarks/single_gpu/test_taxol_xc_diagnostic.py::test_fixed_orbital_xc[gpu]' -q -s
    python -m pytest benchmarks/single_gpu/test_taxol_original_profile.py -q -s

After the metric correction, qualify candidate speed/memory gains separately
from the historical baseline. Further performance opportunities include SCF J/K
and XC, derivative integrals and exchange block-size
selection under an explicit workspace budget. The1542-AO disk probe transferred
7518 auxiliary rows in235 calls, so short streaming blocks also need measurement;
the existing128/214-row component timings are not enough to tune that workload.

`gradient_snapshot.py` saves converged FP64 orbitals, occupations and energies.
Its replay test requires reproducing the original full gradient within1e-9.
`test_occupied_gradient.py` uses these verified orbital states for separate
fresh-process allocation observations and three alternating gradient-only timing
pairs after both warmups. It does not time/repeat SCF, and its ratios must not
be reported as whole-workflow gains. Saved converged energy is used only for
unchanged CPU energy checks; gradients are recomputed with full responses.

`test_large_jk_capacity.py` exercises bounded disk storage on a1542-AO system,
with a small disk smoke case and a Taxol full-host comparison. This intentionally
performs one J/K evaluation at a non-self-consistent one-electron orbital guess;
it does not qualify converged large-system SCF energies or forces. The independent
NumPy contraction oracle checks selected J/K matrix elements using the same
GPU-generated DF integrals, with a raw row buffer capped at256MiB. It is not
an independent CPU integral implementation. Its supplementary density, factor,
lookup and selected-row arrays and BLAS workspace are outside that row-buffer cap.

`recontract_gather.py` is a separate, unpromoted prototype for generally contracted
bases. Actual cc-pVDZ density-fitting checks pass; tested def2-SVP bypasses this
path. Its diagnostic timing report was measured during CPU-reference contention
and is not final speed qualification. It cannot explain the earlier def2-SVP
experimental-cache iteration regression. Candidate GPU CSR mappings for all
batches must be counted before making any memory claim.

`test_large_stream_exchange.py` measures another unpromoted opportunity: the
existing strided GEMM on1542-AO/300-occupied-column blocks shorter than256
auxiliaries. Component ratios are not molecular speedups. Changing production
thresholds requires matched molecular checks; the256-row same-algorithm control
provides a timing sanity check.

These tests deliberately separate array/contraction qualification from the
native integral package. They require one visible GPU. Multi-GPU work is excluded.

Run from the repository root with a Python environment containing CuPy and pytest:

    python -m pytest benchmarks/single_gpu/test_environment.py -k gpu_environment -s
    python -m pytest benchmarks/single_gpu/test_contraction.py benchmarks/single_gpu/test_cuda_architectures.py -q
    python -m pytest benchmarks/single_gpu/test_benchmark.py -q -s

The local usable test interpreter is
/home/qshao/miniconda3/envs/dft_oracle/bin/python.
The default interpreter lacks PySCF. Do not assume it is interchangeable.

The contraction fixture loads the real source file with a scoped stub for its
unused logger import. It does not stub CuPy, mathematical operations, or device
execution. This permits component testing before native integral compilation.

The benchmark compares the current source to pinned upstream commit
9af748a71ca8f4881c604332c2f0cc1632e033ac, and writes contraction_results.json.
Both versions use exactly the same fallback inputs, output buffers and dtype.
Seven alternating batches of 30 calls follow five warmup calls.
Each measurement synchronizes the GPU and records wall time and CUDA events.
Reported event durations include launch gaps; they are not individual kernel
execution times. Pool usage fields are retained allocations, not peak VRAM.
First-call values share process/library caches and are not fair cold-start
comparisons; speedups use warm medians only. No speedup assertion is used because
GPU contention makes a hard timing threshold unsuitable for correctness tests.
The report includes source hashes, software versions and raw timing samples.

After compiling the native source, run:

    python -m pytest benchmarks/single_gpu/test_environment.py -k native_source_import -s
    python -m pytest gpu4pyscf/lib/tests/test_cutensor.py -q
    python -m pytest gpu4pyscf/mp/tests/test_dfmp2.py -q

Molecular end-to-end qualification has passed on the local GB10.
Component speedups do not establish complete SCF/gradient/Hessian speedups.
See docs/single-gpu-support.md for the capability matrix and native build commands.


Molecular CPU references can run before native GPU libraries are compiled:

    python -m pytest benchmarks/single_gpu/test_molecular.py -m "not special" -q

This records HF/PBE/B3LYP/WB97X energies and gradients, an HF Hessian and
DF-MP2 correlation energy, with convergence information and component timings.
Results go to molecular_results/. DFT gradients include grid response and DF
gradients include auxiliary-basis response. The small unpruned level-1 grid is
a matched regression setting, not a recommendation for production accuracy.
DF setup is observed without forcing extra tensor construction; setup times
are part of SCF elapsed time, not additive independent timings.

After a native build, execute matched GPU comparisons:

    python -m pytest benchmarks/single_gpu/test_molecular.py -m special -q

These tests fail if native libraries are missing; they do not skip or substitute
an installed GPU4PySCF package. Energy, gradient and Hessian gates are 1e-8 Eh,
1e-6 Eh/Bohr and 1e-5 Eh/Bohr squared. CPU/GPU scientific settings must match.

The observed component dependency pins are in requirements-cuda13-components.txt.
To reproduce them without modifying a shared scientific environment:

    python3.11 -m venv .venv-single-gpu
    .venv-single-gpu/bin/python -m pip install -r benchmarks/single_gpu/requirements-cuda13-components.txt

A usable CUDA driver/runtime and CMake are also required. These commands describe
the observed component environment; installing it does not replace compiling the
native libraries or qualify the whole current package.


Molecular artifacts include source revision/dirty state and SHA256 fingerprints
of the benchmark, comparison tests, contraction implementation and build policy.
GPU artifacts additionally identify the CUDA driver/runtime, GPU and loaded
native library binaries. MP2 timing includes object construction, auxiliary
setup and kernel on both backends; imports are outside the timed region.
The validator requires symmetric Hessian presence, matching scientific settings
and shapes, convergence, and the stated numerical tolerances.

Run the DF memory-path profile and complete workflow benchmark after building:

    python -m pytest benchmarks/single_gpu/test_df_profile.py benchmarks/single_gpu/test_workflow_benchmark.py -q -s

The DF profile records resident and mapped-host tensors at block sizes 128,
256 and 512, with CPU J/K comparisons and seven alternating paired timings.
The workflow benchmark records three alternating paired HF/B3LYP SCF+gradient
samples. Both compare the pinned baseline contraction against the modified
implementation in the same process and restore imported bindings afterward.
Reports are df_profile_results.json and workflow_results.json. GPU pool values
exclude direct allocations and are not peak memory. Shared-GPU timing variation
prevents a reliable complete-workflow speedup claim.

Combined correctness command (benchmark cases excluded):

    python -m pytest benchmarks/single_gpu gpu4pyscf/lib/tests/test_cutensor.py gpu4pyscf/lib/tests/test_cupy_helper.py gpu4pyscf/mp/tests/test_dfmp2.py -m 'not benchmark' -q

## Larger whole-workflow benchmark

    python -m pytest benchmarks/single_gpu/test_medium_workflow_benchmark.py -q -s

This uses the tracked 20-atom Vitamin C geometry with def2-SVP and an explicit
JKFIT auxiliary basis for HF/PBE/B3LYP/WB97X. Both contraction variants receive a
method warmup, followed by seven alternating paired GPU calculations. Every
result must agree with its matched CPU reference within 1e-8 Eh energy and 1e-6
Eh/Bohr gradient. SCF convergence is 1e-10 Eh and 1e-7 orbital gradient; grid level
1 has pruning disabled. These are regression/performance settings, not a
production grid-convergence study.

The baseline swaps only the pinned contraction implementation into the same
current native build. This isolates the contraction change; it does not compare
independently built GPU4PySCF releases. Timings cover mf.kernel and gradient.kernel,
including DF initialization inside SCF. Object construction, initial-guess
generation, imports and metadata collection are outside those intervals. CPU
reference times are not a tuned CPU performance comparison.

medium_workflow_results.json contains raw runs, iteration counts, correctness
errors, phase medians and paired-ratio spreads. Use those spreads when evaluating
small changes on this shared GPU; do not aggregate method-specific ratios into
a blanket speedup. Earlier component/DF/water reports are archived under
runs/2026-10-17-before-rerun/ before replacement by this campaign.

### Measured campaign, 2026-10-17

The component/DF/water campaign passed 5 tests; the larger workflow benchmark
passed separately in 360.13 seconds. All 56 timed Vitamin C GPU calculations
passed their CPU energy/gradient gates, and baseline/current iteration counts
matched for each method. Maximum errors were 4.58e-11 Eh and 6.09e-8 Eh/Bohr.

| Vitamin C method | Baseline SCF + gradient | Current SCF + gradient | Median-time ratio | Paired ratio range |
| --- | ---: | ---: | ---: | ---: |
| HF | 2.081 s | 2.065 s | 1.008x | 1.000–1.023x |
| PBE | 3.287 s | 3.295 s | 0.998x | 0.984–1.017x |
| B3LYP | 3.534 s | 3.513 s | 1.006x | 1.001–1.011x |
| WB97X | 5.004 s | 4.991 s | 1.003x | 0.999–1.012x |

Ratios above one mean faster current code. The headline ratio divides the two
timing medians; paired ratios compare corresponding repetitions and need not
have the same median. There is no substantial whole-workflow gain on this case.
The seven pairs and shared GPU do not establish broad statistical significance.

Component gains reproduced at 2.50x (small matrix) and 6.98x (DF-J accumulation).
Vitamin C block-256 J/K ratios were 1.009x resident and 1.066x host. Whole-water
HF/B3LYP ratios were 1.032x/0.953x, illustrating sensitivity of small workflows.

Further optimization should start with profiling the unaccelerated time and
checking system/basis-size scaling. Gradients account for about 33–40% of current
HF/hybrid/range-separated workflow times; PBE spends about 84% in SCF. These are
phase timings, not identification of individual kernel bottlenecks. Profile DF
gradient assembly and XC/SCF stages before choosing a kernel change or memory
heuristic. No production optimization or global block-size change was made in
this benchmark campaign. cuEST and neural Skala execution remain unqualified.

benchmark_campaign_summary.json retains raw-report hashes, baseline-switch helper
hash, paired spreads, iteration counts, phase shares and numerical-error maxima.

## Stage profiling and size scaling

    python -m pytest benchmarks/single_gpu/test_stage_profile.py -q -s

This current-code campaign covers PBE and B3LYP for Vitamin C with def2-SVP
(208 AOs), Vitamin C with def2-TZVP (420 AOs), and tamoxifen with def2-SVP
(537 AOs). Each case has a matched four-thread CPU reference, GPU warmup,
three controls, and one separately instrumented run. Controls and instrumented
runs must meet the same 1e-8 Eh and 1e-6 Eh/Bohr CPU-reference tolerances.
The grid, response and convergence settings match the larger workflow benchmark.

Stage wrappers synchronize the GPU before and after each call and restore all
bindings on exit. Their times include nested calls: DF build is inside SCF J/K,
SCF J/K and XC are inside effective potential, and gradient J/K core is inside
gradient J/K total. Do not sum parent and child labels. Within-phase stage
shares use the instrumented phase duration; combined XC and J/K shares below
use the instrumented SCF-plus-gradient duration. These are synchronized wall-time
shares, not pure GPU-kernel measurements or achievable speedups. Performance
totals use the three control timings. Controls retain the molecular helper's
per-iteration synchronization.
The instrumented/control ratio measures additional stage-wrapper perturbation;
it does not measure overhead against a fully asynchronous application.

stage_profile_results.json retains controls, instrumented stages, full CPU/GPU
energies and gradients, iteration counts, numerical errors, source/binary hashes
and geometry hashes. These shared-GB10 timings describe this local build and
these settings; changing molecule, basis and SCF cycle count does not establish
an asymptotic scaling exponent. No production code changes are part of profiling.

### Measured stage campaign, 2026-10-17

The matrix passed as one benchmark test in 1056.25 seconds, with all 24 timed GPU
calculations passing the CPU gates. Maximum errors were 7.69e-11 Eh and
6.68e-7 Eh/Bohr. Warmups also passed their CPU comparisons. Control iteration
counts matched the instrumented run for every case.

| Molecule / basis | AOs | Method | Control median SCF + gradient | Control range | XC share | J/K share |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Vitamin C / def2-SVP | 208 | PBE | 3.334 s | 3.324–3.345 s | 56.0% | 36.2% |
| Vitamin C / def2-SVP | 208 | B3LYP | 3.510 s | 3.502–3.520 s | 41.5% | 51.6% |
| Vitamin C / def2-TZVP | 420 | PBE | 7.053 s | 7.052–7.103 s | 59.0% | 33.8% |
| Vitamin C / def2-TZVP | 420 | B3LYP | 12.499 s | 12.483–12.737 s | 28.1% | 68.3% |
| Tamoxifen / def2-SVP | 537 | PBE | 18.753 s | 18.710–18.820 s | 65.7% | 30.3% |
| Tamoxifen / def2-SVP | 537 | B3LYP | 59.497 s | 59.465–60.585 s | 18.8% | 80.0% |

XC share adds SCF quadrature and gradient full XC response. J/K share adds SCF
J/K and gradient DF J/K total. Each percentage divides those disjoint stage times
by instrumented SCF-plus-gradient elapsed time. Parent stages and nested DF build
or J/K core timings are excluded from these sums. Instrumented/control ratios
ranged from 0.999 to 1.018, with only one instrumented sample per case; that range
is not an uncertainty estimate.

The next optimization priorities are:

1. **PBE XC:** separately profile AO evaluation, density construction, functional
   evaluation and potential/grid-response assembly in `dft/numint.py` and
   `grad/rks.py`. XC accounts for 56–66% of combined time in these cases.
   Investigate buffer reuse, repeated AO work and synchronization only after
   that finer split identifies the responsible work.
2. **Larger hybrid DF J/K:** separate integral generation, unpacking/contractions,
   metric operations and derivative kernels in `df/df_jk.py` and `df/grad/rhf.py`.
   J/K grows from 52% to 80% of combined time across these hybrid cases; prioritize
   the SCF path as well as gradients. The current coarse profile does not resolve
   individual kernels or justify a default block-size change.
3. **Acceptance benchmark:** test any proposed change with alternating baseline
   and candidate whole workflows on this matrix, unchanged scientific settings,
   CPU energy/force gates and iteration counts. Validate production grid settings
   separately before drawing conclusions for production calculations.

`stage_profile_summary.json` records the derived shares, phase medians, control
spreads, error maxima and raw-report hash. This campaign identifies optimization
targets; it introduces no additional acceleration or external-library support.

## Finer XC and DF J/K profiling

    python -m pytest benchmarks/single_gpu/test_fine_stage_profile.py -q -s

This narrows the investigation to tamoxifen/def2-SVP PBE and B3LYP. The saved
CPU references are reused after checking geometry, scientific settings, CPU
library versions and recorded source/native build fingerprints. Each method
gets a fresh GPU warmup, three controls and one instrumented run. Thread-local
scopes distinguish SCF XC worker calls from gradient XC calls. DF generator
advancement is timed around `next`, excluding work done by its consumer.

The campaign passed in 399.49 seconds. All eight timed calculations and both
warmups passed CPU checks. Maximum errors were 7.73e-11 Eh and 7.03e-7 Eh/Bohr.
Control medians were 18.725 seconds (PBE) and 60.643 seconds (B3LYP); corresponding
instrumented/control ratios were 1.013 and 0.992. A ratio below one reflects
run variation; one instrumented sample does not establish an overhead bound.

| Measured helper | Method | Instrumented time | Interpretation |
| --- | --- | ---: | --- |
| SCF MO-density construction | PBE | 4.643 s | 52.3% of SCF XC stage |
| SCF functional evaluation | PBE | 0.034 s | 0.38% of SCF XC stage |
| SCF AO evaluation, both passes | PBE | 0.748 s | Smaller than density construction |
| Gradient GGA matrix assembly | PBE | 1.851 s | Includes its nested AO scaling |
| Exchange transform `Lij,njk->nikL` | B3LYP | 16.457 s | 209 calls |
| Exchange accumulation `nikL,njkL->nij` | B3LYP | 12.005 s | 209 calls |
| DF block retrieval, including unpack | B3LYP | 0.980 s | Includes nested unpack time |
| Three-center derivative kernel calls | B3LYP | 3.866 s | 12 synchronized calls |

The two exchange contractions together occupy 74.1% of instrumented B3LYP SCF
J/K time and 47.3% of the combined SCF-plus-gradient time. Recorded transform
shapes have 537 AOs, occupied-factor dimensions 100 or 169, and auxiliary block
sizes 256 or 66. Prioritize layout-aware GEMM and contraction/temporary-memory
experiments for those exact shapes before changing integral or unpack kernels.
For PBE, prioritize density construction and potential/gradient matrix assembly;
functional evaluation is a small part of the measured time.

These are synchronized helper wall times. Internal layout conversion, allocation
and BLAS/kernel work remain combined within contractions. SCF AO-dot matrix
products are not separately wrapped, so scaling/scatter helpers do not account
for all potential assembly. The PBE Coulomb gradient uses a different integral
path from the hybrid evaluator; a missing fine integral label means unmeasured
work, not zero cost. Fine timings are inclusive and must not be added to their
coarse parents. All patched bindings were restored and iteration counts matched.

Raw timings, expression-specific shapes and source hashes are saved in
`fine_stage_profile_results.json`; derived fractions and reporting limits are
in `fine_stage_profile_summary.json`.

### Reversible GGA density experiment

    python -m pytest benchmarks/single_gpu/test_mo_density_experiment.py -q -s

The benchmark-only candidate batches four GGA AO-to-MO products and applies the
existing fused GGA density-reduction kernel. Other functional classes and dtypes
delegate to the original implementation. The experiment checks sufficient and
insufficient scratch buffers, reused output, strided AO inputs, empty occupied
spaces and extra AO derivative components. All five component cases passed.

Component timings are exploratory sequential samples. The acceptance benchmark
uses PBE Vitamin C and tamoxifen with five alternating original/candidate pairs
after both variants warm up. Every result gets the unchanged CPU energy/force
checks; paired SCF iteration counts must match. Scientific settings and native
build are identical, and no production binding persists after the experiment.

The component test passed in 1.27 seconds and the paired workflow test passed in
269.08 seconds. All 20 timed workflow calculations and four warmups met the CPU
gates, with maximum errors 2.86e-11 Eh and 3.25e-8 Eh/Bohr. Iteration counts matched
within every pair. These measurements compare only the local original function
and benchmark-only candidate in the same build.

| PBE system | Original median | Candidate median | Candidate time change | Paired original/candidate ratio range |
| --- | ---: | ---: | ---: | ---: |
| Vitamin C / def2-SVP | 3.334 s | 3.431 s | +2.90% | 0.9710–0.9746 |
| Tamoxifen / def2-SVP | 18.740 s | 18.587 s | −0.82% | 1.0070–1.0096 |

The candidate remains **benchmark-only**. The small tamoxifen improvement does
not justify a general change that regresses Vitamin C. Component results were
also mixed: the large 537-AO block improved from 6.061 to 5.576 ms, while the
208-AO block regressed from 0.853 to 1.104 ms. Five pairs on a shared GPU are a
local comparison, not broad performance qualification.

This campaign motivated experiments on the two measured B3LYP exchange patterns
at the recorded shapes, including algorithmic use of symmetry. Layout conversion,
temporary allocation and multiplication must be assessed together, followed by
alternating whole-workflow comparisons on smaller and larger cases. Configured
cuTENSOR/custom-engine behavior and scientific settings must be preserved.

`mo_density_component_results.json` and `mo_density_workflow_results.json`
contain raw data. `mo_density_experiment_summary.json` retains paired ratios,
numerical maxima, raw-report hashes and the decision to reject a global change.

## Algorithm optimization: symmetry and adaptive density

    python -m pytest benchmarks/single_gpu/test_algorithm_candidates.py -q
    python -m pytest benchmarks/single_gpu/test_algorithm_benchmark.py -q -s

These experiments change arithmetic rather than numerical approximations.
The exchange candidate computes real self-Gram updates with cuBLAS SYRK and
mirrors the computed triangle using the existing native helper. Its input/output
eligibility requires a symmetric accumulator, identical left/right factors,
float64 contiguous arrays and a measured size guard. Mixed factors, supported
complex paths and general response contractions retain the original route. Components
showed SYRK slower at 208, 256 and 384 AOs, but faster at 512 and 537 AOs; the
candidate therefore starts at 512 AOs. A lower flop count alone is insufficient
to choose the GPU implementation.

Adaptive GGA density construction selects the DM route when active AOs are fewer
than four times the occupied MO count, based on the leading multiplication
costs. The local DM is formed from the same occupation-weighted coefficients;
its construction, scratch allocation and output copying are included in timings.
Other functional classes and dtypes retain the original calculation. No grid,
screening, precision, occupation or SCF tolerance is relaxed.

The isolated workflow comparisons use three alternating pairs after both
warmups, with PBE for density and B3LYP for exchange on Vitamin C and tamoxifen.
Every energy and gradient is checked against the previously qualified CPU
reference, with matching geometry/settings/library versions/native binaries.
These comparisons retain the earlier matrix-contraction improvements on both
sides and isolate each additional algorithm change.

`algorithm_baseline.json` retains pre-algorithm function source and file hashes;
the saved function ASTs were independently checked against revision
`9af748a71ca8f4881c604332c2f0cc1632e033ac`. The final production acceptance test
also restores that revision's contraction implementation, so its baseline is
the original GPU Python algorithms and contractions in the same native build.
This measures cumulative changes, not independently built package releases.

The isolated campaign passed three benchmark tests in 648.82 seconds. Component
timings include SYRK triangle reconstruction and local-DM formation. SYRK took
23.355 ms versus 60.127 ms for the original 537-AO, 100-occupied, block-256 Gram
update. Adaptive density took 0.399/1.501/3.207 ms versus 1.474/2.638/4.268 ms
at 128/256/384 active AOs and 100 occupied MOs. These are component measurements,
not complete-calculation speedups.

| Isolated additional algorithm | System / method | Original median | Candidate median | Speedup | Paired ratio range |
| --- | --- | ---: | ---: | ---: | ---: |
| Adaptive density | Vitamin C PBE | 3.243 s | 3.186 s | 1.018x | 1.018–1.022 |
| Adaptive density | Tamoxifen PBE | 18.767 s | 16.932 s | 1.108x | 1.106–1.116 |
| Symmetric exchange | Vitamin C B3LYP | 3.499 s | 3.503 s | 0.999x | 0.998–1.000 |
| Symmetric exchange | Tamoxifen B3LYP | 59.559 s | 51.245 s | 1.162x | 1.146–1.163 |

Both algorithms were promoted after these comparisons. Production SYRK is
limited to the CuPy fallback, real float64 ground-state self-Gram updates and
at least 512 AOs. Configured cuTENSOR/custom engines and mixed-factor paths keep
their original contractions. Adaptive density is limited to real float64 GGA
inputs; occupation weights, response terms and mathematical definitions remain
unchanged. It allocates an additional local AO density matrix, so these speed
results do not establish a reduction in peak memory.

Production tests observed the expected failures before implementation. The
post-promotion targeted run passed 50 tests in 8.18 seconds, including production
and candidate invariants, DF J/K and numerical integration. The one excluded
`sparse_index` fingerprint failure was previously reproduced on unmodified
baseline code; no numerical tolerances were relaxed for the new algorithms.
`algorithm_isolated_summary.json` records paired spreads, error maxima and raw
report hashes. The final production-versus-original-GPU comparison is separate
so the isolated and cumulative ratios are not conflated.

For reruns after promotion, the isolated harness restores saved pre-algorithm
functions for both variants, while retaining earlier contraction improvements.
Live module globals preserve the candidate's nested contract substitutions.
Baseline identity and component reproducibility checks passed after promotion;
their fresh component report is retained in
`runs/2026-10-17-after-promotion-components.json`. Original campaign artifacts
remain unchanged so their recorded hashes stay valid.

### Algorithm-stage production versus original GPU acceptance

    python -m pytest benchmarks/single_gpu/test_algorithm_acceptance.py -q -s

This campaign, before the subsequent exchange-layout promotion below, passed
in 615.45 seconds. Both variants received warmups; the
large cases used three alternating pairs each. Small cases used one pair each
and are timing smoke checks, not robust small-system speedup estimates.

| System / def2-SVP | Method | Original GPU SCF + gradient | Production SCF + gradient | Speedup | Paired ratio range |
| --- | --- | ---: | ---: | ---: | ---: |
| Vitamin C, 208 AOs | PBE | 3.359 s | 3.273 s | 1.026x | one pair |
| Vitamin C, 208 AOs | B3LYP | 3.542 s | 3.485 s | 1.016x | one pair |
| Tamoxifen, 537 AOs | PBE | 18.717 s | 16.885 s | **1.108x** | 1.104–1.117 |
| Tamoxifen, 537 AOs | B3LYP | 59.867 s | 49.572 s | **1.208x** | 1.183–1.230 |

The larger cases reduce measured time by 9.79% (PBE) and 17.20% (B3LYP).
Ratios measure the combined earlier contraction improvements and new algorithm
changes against original GPU Python paths, on the same GB10/CuPy-fallback native
build. These are shared-GPU, unpruned level-1-grid regression settings, not a
production grid-convergence study or a general hardware speedup claim.

All 16 timed calculations and eight warmups passed the matched CPU energy/force
gates, with maximum CPU differences 7.64e-11 Eh and 6.59e-7 Eh/Bohr. Maximum
paired original/production GPU differences were 9.09e-13 Eh and 5.69e-8 Eh/Bohr.
Paired SCF iteration counts matched, and temporary baseline bindings were
restored. Production source hashes were checked against the measured files.

A fresh post-change molecular qualification passed eight tests in 5.89 seconds:
matched HF/PBE/B3LYP/WB97X energies and gradients, HF Hessian, DF-MP2, and result
validation checks. Five standalone CPU-reference benchmark cases were excluded
from that command. Earlier molecular reports are preserved under
`runs/2026-10-17-before-algorithm-molecular/`.

`algorithm_acceptance_results.json` contains the raw comparisons;
`algorithm_acceptance_summary.json` records medians, paired spreads, CPU/GPU
error maxima, measured source hashes and the raw-report hash. Numerical
approximations and precision remain unchanged. The previously rejected batched
GGA candidate remains benchmark-only and is not part of these production defaults.

## Exchange layout experiment after algorithm promotion

    python -m pytest benchmarks/single_gpu/test_optimized_stage_profile.py -q -s
    python -m pytest benchmarks/single_gpu/test_exchange_layout.py -q
    python -m pytest benchmarks/single_gpu/test_exchange_layout_benchmark.py -q -s
    python -m pytest benchmarks/single_gpu/test_exchange_layout_workflow.py -q -s

The fresh profile uses the optimized adaptive-density and SYRK defaults. It
includes one warmup, one uninstrumented control and one instrumented run for
tamoxifen PBE and B3LYP. Every run passes the matched CPU reference gates.
Synchronized helper times are inclusive and perturb overlap and scheduling.
The `density_mo` label now includes adaptive dispatch and nests `density_dm`;
the two must not be summed. SYRK has its own timer because it bypasses `contract`.

The B3LYP control took 49.497 seconds for SCF plus gradient; instrumentation took
51.288 seconds. The exchange transformation `Lij,njk->nikL` still consumed
16.464 seconds over 209 calls, versus 4.756 seconds for the promoted Gram update.
The gradient three-center derivative took 3.866 seconds. PBE SCF density dispatch
took 2.842 seconds (including 2.682 seconds inside DM evaluation), and the XC
functional evaluator took 0.034 seconds. These observations support targeting
the exchange layout next; they do not establish pure GPU kernel percentages.
Raw evidence is `optimized_stage_profile_results.json`.

Unpacked DF blocks are views of `(i,j,block_capacity)` storage, with the
auxiliary index contiguous. A partial block retains its original capacity as
padding. The candidate calls float64 strided batched GEMM directly on this
layout and writes `(n,i,k,L)` into the existing exchange buffer. It allocates
no tensor-sized temporary, preserves the current stream, and restores cuBLAS
pointer mode. The mathematical contraction, precision and scientific settings
are unchanged. Alias, dtype, shape, layout and backend guards retain the generic
contraction when required.

Component comparisons use seven alternating synchronized samples after warmup,
including actual DF strides. Direct GEMM is faster for the measured full
256-auxiliary blocks at 512/537 AOs but loses on the 66-auxiliary partial blocks.
The candidate is therefore enabled only for at least 512 AOs and at least 256
auxiliary functions in the block; smaller shapes retain the existing route.
CuPy's higher-level batched matmul did not reproduce the large full-block gain.
Raw timings are in `exchange_layout_components.json`.

`exchange_layout_baseline.json` saves the previous optimized `get_jk` source
and algorithm-file hashes; the complete pre-layout DF module is archived at
`runs/2026-10-17-pre-layout-df_jk.py`. Replaying the saved function with live
module globals preserves adaptive density and SYRK while allowing a reversible
layout substitution. The paired workflow test warms both variants and uses
three alternating pairs on each system, checking every energy and gradient
against its matched CPU reference. This baseline is the previous optimized
GPU implementation, distinct from the original-GPU acceptance campaign above.

The isolated experiment passed 21 tests in 409.93 seconds, including 19 numerical
and dispatch checks plus component and workflow comparisons. Tamoxifen B3LYP
SCF plus gradient improved from 49.526 to 42.413 seconds (1.168x, 14.36% less
time) across three alternating pairs. Paired ratios ranged from 1.167 to 1.186.
Vitamin C retained the existing route; its observed 1.009x ratio is a control
measurement, not evidence of acceleration from this guarded algorithm.
`exchange_layout_workflow_results.json` contains every timed and warmup result.

The candidate was promoted as `df_jk._exchange_transform`, replacing only the
five batched `Lij,njk->nikL` calls. The unbatched response transformations and
downstream mixed-factor contractions retain their original implementations.
Production dispatch/integration tests observed the expected failures before
implementation. A post-promotion targeted run passed 72 tests in 7.31 seconds,
covering the new helper, previous algorithm invariants, DF J/K and numerical
integration. The documented baseline `sparse_index` fingerprint remains the
single excluded test. Independent review found no material numerical, layout,
backend or response-path issue.

### Promoted layout acceptance and remaining costs

    python -m pytest benchmarks/single_gpu/test_exchange_layout_acceptance.py -q -s

The actual promoted production implementation passed its separate acceptance
campaign in 413.43 seconds. Both variants received warmups, followed by three
alternating pairs and a final instrumented production run. The baseline here
is the previous optimized GPU implementation, including adaptive density and
SYRK, in the same current native build.

| Tamoxifen B3LYP / def2-SVP, 537 AOs | Previous optimized GPU | New production |
| --- | ---: | ---: |
| Median SCF + gradient | 49.539 s | 42.005 s |
| Paired speedup | | **1.179x** |
| Elapsed time reduction | | **15.21%** |

The three paired ratios were 1.17898, 1.17853 and 1.17936. Production phase
medians were 29.154 seconds for SCF and 12.853 seconds for gradients; medians
of phases need not sum exactly to the median combined time. These remain local
shared-GB10/CuPy-fallback measurements at unpruned grid level 1. No cumulative
original-GPU ratio is inferred by multiplying separate benchmark campaigns.

The final profile measured 9.149 seconds over 209 exchange-helper calls, down
from 16.464 seconds for the old transformation. This is **inclusive helper wall
time**, including fallback blocks, not exclusively strided-GEMM kernel time.
Actual cuBLAS strided-batched GEMM executed 190 times. The 19 fallback
transformations took 0.417 seconds, nested inside the helper total. SYRK took
4.745 seconds. Instrumented SCF plus gradient took 44.435 seconds versus the
42.005-second uninstrumented median, so helper synchronization has a material
measurement cost.

Remaining costs include the 9.149-second exchange transform, DF gradient J/K
at 9.652 seconds (including the 3.867-second three-center derivative), SCF XC
quadrature at 6.226 seconds, and the two Coulomb contractions totaling 4.864
seconds. DF setup took 4.110 seconds nested in SCF J/K. These inclusive scopes
overlap and must not be added into a disjoint runtime budget. Gradient work is
now about 31% of the uninstrumented total. Further optimization should target
exchange memory/computation and DF gradient contractions, while preserving
auxiliary-basis and grid responses.

Across the fresh before-profile, isolated experiment and production acceptance,
31 energy/gradient records independently passed the unchanged CPU gates of
1e-8 Eh and 1e-6 Eh/Bohr. Maximum CPU differences were 7.73e-11 Eh and
7.03e-7 Eh/Bohr; maximum paired previous/new GPU differences were 1.36e-12 Eh
and 6.04e-8 Eh/Bohr. All paired iteration counts matched. Precision, mathematical
method, basis, grid, screening and convergence thresholds were unchanged.

A fresh production molecular qualification additionally passed eight tests in
5.60 seconds: HF/PBE/B3LYP/WB97X energies and gradients, HF Hessian, DF-MP2 and
comparison validation. Five standalone CPU-reference benchmark cases were
excluded from that command. Combined with the 72 targeted checks, this yields
80 passing tests; the previously established `sparse_index` fingerprint was
excluded from the targeted run. This is targeted qualification, not a claim
that the entire repository test suite passed. Previous molecular reports are
preserved under `runs/2026-10-17-before-layout-molecular/`.

`exchange_layout_acceptance_results.json` contains the final raw comparisons
and profile. `exchange_layout_summary.json` records medians, paired spreads,
accuracy maxima and report/source hashes. QA checked current source/native
identities, baseline source replay, unchanged density/SYRK helpers and the
reported numerical differences against the raw CPU/GPU records. Other GPU
architectures, production grid convergence and broader method performance
remain unqualified by this campaign.

### Bounded gradient workspaces and direct Coulomb contractions

The next production stage reduces transient density-fitting exchange-gradient storage
and avoids full tensor copies for padded auxiliary blocks. Restricted and
unrestricted gradient workspaces with exchange use auxiliary blocks capped at 128 and a
byte-budget planner. Hermitian gradients size scratch storage using occupied
orbitals. The planner preserves enough compressed storage for a complete
auxiliary shell and raises an explicit `MemoryError` if the minimum cannot fit.
All auxiliary-basis and grid response terms remain included.

The temporary-workspace budget defaults to 2 GiB, additionally limited to 80%
of available GPU memory. It can be set before a gradient calculation:

```python
from gpu4pyscf import __config__
__config__.df_gradient_workspace_bytes = 2 * 1024**3
```

This bounds planned compressed/dense work buffers and includes a conservative
packing/intermediate allowance. It is **not a total-memory limit**: persistent
DF integrals, auxiliary metrics, occupied-pair tensors and native/library
workspaces remain outside it. Planner tests at 2,000 and 4,000 AOs check sizing
only; they do not demonstrate real chemical calculations of those sizes.

For the real FP64 single-density CuPy path, direct cuBLAS GEMV accepts the actual
padded column-major DF blocks without packing a tensor-sized copy. Larger real
gradient transformations accumulate directly through GEMM. The latter path is
restricted to at least 512 Cartesian AOs; smaller matrices preserve the original
contraction order and strict legacy numerical fingerprints. Other contraction
engines and unsupported layouts/types use the existing generic contractions.

Fresh-process production B3LYP/def2-SVP memory qualification on tamoxifen
reduced peak tracked live CuPy allocations from **19.64 to 4.03 GiB** (**79.48%
less**). Initial pool allocations were zero in both runs. The allocation tracker
wraps the existing allocator across worker threads and includes allocations
that bypass the pool above 100 MiB. Pool-only accounting would substantially
underestimate this workload. These peaks exclude native/library allocations
outside CuPy and are not physical device-memory totals. Current-address-space
host peak RSS stayed approximately **1.30 GiB**; no host-RAM reduction is claimed.
GB10 uses unified memory, so host RSS and GPU allocation metrics must not be
summed. Instrumented memory-run timings are diagnostic, not speed evidence.

Reproduce the memory measurement in a fresh process and the separate paired
performance campaign with:

```bash
python -m pytest 'benchmarks/single_gpu/test_speed_memory_qualification.py::test_memory_qualification[current]' -q -s
python -m pytest benchmarks/single_gpu/test_speed_memory_acceptance.py -q -s
python benchmarks/single_gpu/summarize_speed_memory.py
```

The original memory baseline is archived under
`runs/2026-10-17-pre-promotion-speed-memory/`; do not overwrite it. The experimental
component/workflow reports remain separate from actual production acceptance.
The earlier combined prototype measured 42.063 to 38.706 seconds on the large
case (1.087x); that is below the requested additional 20% gain. Actual production
acceptance passed separately in 381.96 seconds, with both implementations
warmed before three alternating large-case pairs and one small smoke pair.

| B3LYP / def2-SVP SCF + gradient | Previous optimized GPU | Production |
| --- | ---: | ---: |
| Tamoxifen, 537 spherical AOs, median of 3 | 42.475 s | 38.867 s |
| Vitamin C, 208 AOs, 1 smoke pair | 3.477 s | 3.411 s |

The large-case speedup is **1.093x**, or **8.49% less elapsed time**. Paired
ratios were 1.09504, 1.09139 and 1.09283. Production phase medians were 26.275
seconds for SCF and 12.605 seconds for gradients. The comparator is the saved
approximately 42-second optimized GPU implementation in the same native build,
not CPU PySCF or the original upstream GPU release. No cumulative speedup is
inferred by multiplying separate campaigns. The small result is a smoke check,
not a repeated performance claim. Hardware is the shared GB10 with CuPy
contractions, using the existing unpruned level-1 regression grid; larger grids
and other architectures are not performance-qualified here.

The final synchronized profile observed 418 actual GEMV calls. Inclusive
Coulomb helper time fell to 0.421 seconds from the preceding profile's 4.864
seconds. Exchange transforms still take 9.136 seconds, SYRK 4.742 seconds,
SCF XC quadrature 6.268 seconds, and gradient DF J/K 9.126 seconds. Within the
last scope, three-center derivatives take 5.277 seconds. Smaller blocks trade
some derivative throughput for much lower memory. Direct gradient helper time
is 1.564 seconds, including any fallback calls. These instrumented, nested
scopes overlap and must not be added into a disjoint time budget. Further speed
work should target exchange and derivative integrals without restoring the
large scratch buffers.

Across the experimental workflow, production acceptance and before/after memory
runs, 31 saved energy/gradient records independently pass the unchanged CPU
gates: 1e-8 Eh and 1e-6 Eh/Bohr. Maximum observed CPU differences are 7.83e-11 Eh
and 6.91e-7 Eh/Bohr; maximum paired GPU differences are 1.37e-12 Eh and
8.93e-8 Eh/Bohr. Iteration counts match in all pairs and the final profile.
FP64, quantum method, basis, grids, screening and convergence tolerances are
unchanged. A strict small-matrix gradient fingerprint is preserved using the
original contraction path, without relaxing its tolerance.

Fresh production validation passed 98 targeted tests, with the previously
established upstream `sparse_index` fingerprint excluded, followed by eight
molecular tests (HF/PBE/B3LYP/WB97X energies and gradients, HF Hessian, DF-MP2
and comparison gates). Five standalone CPU benchmark cases were deselected in
the latter command. Restricted/unrestricted gradients, a converged open-shell
CPU comparison and tight workspace boundaries are covered. This is targeted
qualification rather than the full repository suite. Previous molecular
reports are preserved under `runs/2026-10-17-before-speed-memory-molecular/`.

`speed_memory_acceptance_results.json`, `speed_memory_usage_current.json` and
`speed_memory_summary.json` retain the raw comparisons, allocation accounting,
independently recomputed medians/errors and source/report hashes. The stdlib
summary script verifies current production and native identities, baseline
source replay, and unchanged previous-stage exchange/SYRK helpers. Independent
production code review found no material issues.

At that stage, the remaining capacity limit was persistent
integral storage, approximately proportional to auxiliary functions times AO
pairs, plus auxiliary metrics and occupied-pair intermediates. The default
host-storage fallback still uses RAM. The following stage adds opt-in disk
storage for CDERI; metric and occupied-pair storage remain capacity limits.

### Disk-backed DF and CPU–GPU communication

Molecular single-GPU DF now supports opt-in temporary disk storage. Integral
construction writes FP64 AO-pair blocks directly into the backing file; it
never allocates a full host CDERI tensor. Iteration reads auxiliary rows
directly into reusable pinned host buffers. Two GPU buffers and CUDA events
overlap H2D with consumer computation and protect both host and device reuse.
This extends capacity rather than replacing the default resident fast path.

```python
from gpu4pyscf import dft

mf = dft.RKS(mol).density_fit()
mf.xc = 'b3lyp'
mf.with_df.cderi_storage = 'disk'
mf.with_df.cderi_directory = '/path/to/disk/scratch'
mf.with_df.cderi_buffer_bytes = 1024**3
# Experimental only, after checking convergence and timing on your workload:
# mf.with_df.cderi_cache_bytes = 384 * 1024**2
mf.kernel()
```

Configure storage before building integrals. `cderi_storage=None` retains the
existing automatic GPU/host behavior. Disk mode requires one GPU and conflicts
with `use_gpu_memory=True`. Files are temporary and removed when the final
owning reference disappears; `with_df.reset()` releases that object's reference.
Copies and live iterators may retain their own reference. There is no persistent
checkpoint/restart format. Use a disk-backed scratch directory; a RAM filesystem
would consume system RAM. Qualification used ext4 on local NVMe, with OS file
cache uncontrolled.

The default CDERI staging budget is **128 MiB**. It bounds explicit build
buffers and, independently, streamed pinned-host and GPU staging. The integral
evaluator has indivisible AO-pair groups: the tamoxifen case requires
**995,840,640 bytes** of explicit build workspace, and its qualification uses
the configured **1 GiB** budget shown above. A smaller budget raises an
informative `MemoryError`; it does not silently exceed the cap. This is not
a total-process or total-GPU limit. Persistent metric/factor storage, unpacked
integrals, gradient/consumer workspaces, native/library allocations, pinned
allocator caches and the OS file cache are outside the staging cap.

`cderi_cache_bytes` is **experimental**, defaults to zero and is a separate GPU
storage budget. Auxiliary rows within that budget remain resident, filled directly on
GPU during construction. Later SCF iterations skip their disk reads and H2D
copies for complete contraction blocks. Partial cache tails remain allocated
but are not used when they would change the original block boundaries. A cache
smaller than one block has no hits. The rest of the tensor streams through the bounded buffers. This
reduces actual logical transfer requests compared with uncached disk mode;
the fully GPU-resident path already avoids those repeated transfers. Cache
capacity is a deliberate GPU-memory versus communication tradeoff.

Repeated small-case timing rejected the cache as a performance improvement:
it took29 rather than18 SCF iterations, despite passing unchanged convergence,
energy and gradient gates. Different floating-point execution can affect SCF
history even when integral values and final scientific results agree. Earlier
unaligned-cache timing also encountered a nonconverged run; it remains archived
as a failed experiment. Preserving block boundaries is a safeguard, not proof
that this was the sole cause. Cache is not promoted to a default or generally
speed-qualified option. Check end-to-end convergence and timing before enabling
it; per-pass transfer savings alone can be erased by extra iterations.

The storage object's counters record read/write bytes, H2D/D2H request bytes
and calls, skipped cache bytes, and explicit staging peaks. These are logical
array-copy counts, not measured physical bus traffic; GB10 has unified memory.
Host RSS and tracked CuPy peaks must not be added. Small tensors can have
similar or higher host RSS with disk mode because of staging/pinned-pool
overhead. Avoiding a full host tensor does not imply reduced RSS in every case.

Consumer work must run on the stream active when iteration starts, or manage
its own cross-stream dependencies. Yielded device buffers are borrowed until
the next iteration. Closing an iterator early synchronizes its in-flight
copies and consumer stream before releasing staging storage.

Regression coverage includes restricted/unrestricted hybrid and range-separated
SCF/gradients, general contractions, AO2MO and an HF Hessian. All scientific
settings and FP64 precision are retained. This storage mode is not implemented
for periodic or multi-GPU DF, and does not bound AO2MO output tensors or the
separate DF-MP2 integral driver's storage.

Fresh-process qualification compared all four storage modes on Vitamin C
(208 AOs) and tamoxifen (537 AOs), using identical B3LYP/def2-SVP scientific
settings. The large-case CDERI file is1,946,223,136 bytes. Its configured
1GiB staging budget covers an explicit995,840,640-byte build workspace;
streaming uses379,461,632 bytes each of pinned host and GPU buffers.

| Tamoxifen, fresh process | Peak tracked CuPy allocations | Host address-space peak RSS |
| --- | ---: | ---: |
| GPU-resident CDERI | 4.030 GiB | 1.296 GiB |
| Existing host-resident fallback | 4.030 GiB | 3.108 GiB |
| Disk, no GPU cache | 1.290 GiB | 2.296 GiB |
| Disk, experimental 384 MiB cache budget | 1.665 GiB | 2.297 GiB |

Disk-only reduces tracked CuPy peaks by **67.99%** versus GPU-resident mode
and process RSS by **26.13%** versus host fallback. It increases host RSS
versus GPU-resident mode. These are distinct accounting metrics, not total
physical memory. The CuPy ledger excludes native allocations outside its
allocator. Process RSS includes resident host allocations and library pages;
unmapped OS file cache is not included in either figure and is uncontrolled.
The small case does not show a host-RSS reduction; staging overhead matters.

The large cache allocates402,436,848 GPU bytes within384MiB, with379,461,632
bytes used per pass as complete contraction blocks. Over19 full SCF tensor
passes, it avoids7,209,771,008 bytes of disk reads and H2D requests:
**36.978 to29.768 GB**, or **19.50% fewer bytes**. H2D requests
fall from209 to171. Construction still writes the full1.946GB tensor once,
with the same logical D2H volume. In the fresh-process small-case run, a64MiB
cache budget reduces H2D from3.133 to2.374GB, **24.24% less**, over the same19
passes. Resident GPU CDERI has no repeated block H2D;
these transfer savings compare cached with uncached streaming storage.

Run each memory variant in its own process. For example:

```bash
python -m pytest 'benchmarks/single_gpu/test_disk_df_qualification.py::test_disk_memory[tamoxifen_svp-disk]' -q -s
python -m pytest 'benchmarks/single_gpu/test_disk_df_qualification.py::test_disk_memory[tamoxifen_svp-cached]' -q -s
python -m pytest benchmarks/single_gpu/test_disk_df_speed.py -q -s
python benchmarks/single_gpu/summarize_disk_df.py
```

Memory-hooked cold timings are diagnostic only. `disk_df_speed_results.json`
records a separate uninstrumented small-case campaign with one warmup per mode
and three rotated rounds, including fresh integral construction each time.
OS file cache is uncontrolled, so the campaign does not qualify cold-storage
performance or physical NVMe throughput. No large-case uninstrumented speed
claim follows from the memory campaign.

The three-round small-case medians are3.408s resident,3.549s host,3.693s disk,
and4.744s cached disk. Disk-only costs about8.36% more elapsed time than
resident storage here. Cached disk is **not performance-qualified**: all three
warm timed runs take29 rather than18 iterations. Its whole-SCF H2D rises to
3.748GB versus3.133GB uncached (19.62% more), despite24.24% fewer bytes per
pass. The fresh-process cached run takes18 iterations; that result does not
override the repeated warm-run regression. No cache speed benefit is claimed.

All24 saved memory/timing energy-gradient records pass unchanged1e-8Eh and
1e-6Eh/Bohr CPU gates. Maximum differences are7.64e-11Eh and6.74e-7Eh/Bohr.
Final observers retain weak references and metadata only, and verify that
gradients release CDERI normally. Earlier reports with retaining observers
are archived and excluded from the final resource comparison.

The broad production regression run passed115 tests, including17 initial disk tests,
with the previously established upstream `sparse_index` fingerprint excluded.
The tests include injected failed writes, insufficient general-contraction
budgets, exact short I/O, partial auxiliary blocks and early closure, skipped
cache bytes, queued consumers on nondefault streams, and CPU scientific gates.
After the final cache block-boundary change,18 disk tests plus a repeated-J/K
diagnostic passed. A fresh molecular run passed eight additional tests (five
standalone CPU benchmark cases deselected). The repeated-J/K diagnostic uses
the converged tagged density, checks both cache/file identity and unchanged
inputs, and compares repeated resident/disk/cached J/K under tight tolerances.
`disk_df_summary.json` independently checks source/native/report hashes,
recomputes errors and medians, and verifies transfer accounting. Pre-stage
sources are retained under `runs/2026-10-17-pre-disk-df/`; preliminary reports
remain in explicitly named archive directories.

## Corrected-code optimization campaign (2026-10-18)

This campaign uses the stable, factor-based auxiliary metric correction as its
baseline. The fresh Taxol/def2-SVP/B3LYP profile passes the independent CPU
energy/force gates at the original scientific settings. Saved orbitals provide
only the starting density for its fresh SCF; isolated gradient comparisons then
use the identical converged orbitals. Inclusive profile timings overlap.

The profile measures 98.63 s gradient J/K (including 30.72 s metric application) and
38.51 s full-response XC. Occupied-pair packing V3 still differs by 2.46e-9 Eh/Bohr,
exceeding the strict 1e-9 differential. Its failed raw report is retained and it
is excluded from promotion and timing. No threshold was relaxed.

Short auxiliary blocks 32/64/128 at 1542 AOs/300 occupied columns improve the exchange
transform by 1.32-1.35x on both tight and padded layouts. The separate molecular
Valinomycin experiment deliberately builds its 31.524 GB disk tensor with a 4 GiB
budget, then reduces the existing iteration buffer setting to 268,363,776 bytes
(two 32-row sparse buffers). This is a controlled low-workspace J/K experiment,
not a default whole-SCF benchmark. Three alternating pairs after warmups give
49.033 s baseline versus 44.425 s candidate median (1.1037x). Each pass requests the
same 31.524 GB H2D in 235 copies; no communication reduction is claimed. All saved
GPU differential and selected NumPy-contraction errors are finite and pass.
The selected CPU contraction uses the same GPU integrals, not an independent
CPU integral oracle. Scratch cleanup and zero cache are checked.

**Initial acceptance protocol error:** the new harness captured the factory
before `configured()`, bypassing NWChem grid pruning while reporting it enabled.
Both initial resident reports (`production_resident_acceptance.json` and
`production_resident_xc_acceptance.json`) fail CPU comparisons at a different
scientific configuration. They are excluded from candidate acceptance/rejection,
and remain unchanged. Their sources are retained in the historically named
`runs/2026-10-18-rejected-short-exchange/` and
`runs/2026-10-18-invalid-acceptance/` directories. The temporary exchange
withdrawal did not remove the offset. A new regression reproduces the protocol
bug before the fix; the corrected harness captures the configured factory and
asserts actual pruning, grid level, convergence thresholds and iteration limit.
Final production evidence uses only `production_{storage}_qualified_acceptance.json`.

The full-response XC grid-weight workspace experiment caps the estimated batch
target at 512 MiB and rounds down to 16 grid points. It retains all grids, weights,
FP64 calculations and response terms. Fresh-process XC allocation peaks fall
from 7,887,651,993 to 1,124,073,384 bytes, with a 2.22e-16 component difference.
Complete gradient new-allocation peaks fall from 7,951,961,177 to 6,556,902,612 bytes
(17.54%), with a 3.23e-13 force difference from the corrected GPU reference and
6.98e-9 from the independent CPU reference. These are requested CuPy allocations
within the measurement scope, excluding preexisting/native storage; process RSS
is separate. The 512 MiB target does not bound total memory. All fresh-process
memory timings are diagnostic, not speed qualification.

Raw reports are under `optimization_results/`; pre-stage sources and the
original streamed benchmark harness are retained under
`runs/2026-10-18-pre-next-optimization/`. `xc_workspace.py` recompiles a hash-pinned
original XC function to preserve the baseline after production changes. The
streamed benchmark's original source remains matched to its report; a subsequent
review added explicit finiteness checks, and the completed report was checked
independently for finite errors.

Paired XC timing is effectively unchanged: three alternating pairs give
38.806 s baseline versus 38.732 s capped median. Both production changes are
qualified by fresh resident and disk Taxol SCF/full-force runs, each taking three
SCF iterations at the original settings. Maximum independent CPU differences
are 2.225e-9 Eh and 6.991e-9 Eh/Bohr, below the unchanged 1e-8/1e-6 gates.
Both report verified actual factory settings; disk checks enforce zero cache,
bounded staging, weak-reference release and scratch cleanup. The final regression
run passes 56 focused tests plus 12 configuration/reporting tests. Independent
read-only review found no remaining material issues. The evidence validator
recomputes raw ratios, numerical errors and source/native/input hashes:

```bash
python benchmarks/single_gpu/summarize_corrected_optimization.py
```

`optimization_results/summary.json` records the passing result and explicitly
excludes the two unpruned protocol attempts. Component speed ratios must not be
multiplied or presented as a whole-workflow improvement. Host RSS is essentially
unchanged in this gradient experiment; the measured reduction is in requested
CuPy allocations, not total physical RAM.
