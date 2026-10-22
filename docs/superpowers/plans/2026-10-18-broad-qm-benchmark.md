# Broad QM Benchmark Campaign Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This document authorizes no benchmark execution by itself; the current request is planning only.

**Goal:** Establish which QM tasks benefit from the optimized GPU4PySCF, by how much, at what memory and communication cost, and with what independently verified accuracy.

**Architecture:** Freeze the original GPU code and the final reviewed optimized code, then run a stratified, capability-gated suite through a new benchmark-only driver. Keep independent CPU references, uninstrumented workflow timings, allocation observations, and communication profiles separate. Preserve the existing campaign records and manuscript as historical evidence.

**Tech stack:** Existing PySCF/CuPy/GPU4PySCF environment; Python/JSON/NumPy; existing phase allocation ledger; pytest for harness validation; Matplotlib for final figures; optional Nsight Systems if available and usable on GB10. Initial execution platform: one NVIDIA GB10, four CPU threads per timed GPU job, FP64. Multi-GPU and non-NVIDIA porting remain outside the initial campaign.

**Status:** Proposed experiment and implementation plan; no new QM calculations or GPU benchmarks executed to prepare it. Budget assumption pending user preference: begin with a bounded pilot and estimate the full campaign from measurements.

---

## 1. Recommended scope and alternatives

Use a balanced suite: **60 core energy-plus-gradient configurations**, followed by **56 targeted extension configurations**, plus an adaptive capacity study. A configuration specifies molecule/cell, charge/spin, method, basis/auxiliary basis, task, response settings, and any environment. It is not a timing repetition. Extensions enter performance testing only after their capability and accuracy gates pass.

Alternatives considered:

| Approach | Benefit | Limitation |
|---|---|---|
| Balanced suite, recommended | Covers the main cost regimes and application types with explicit controls | Requires several task adapters and method-specific comparisons |
| Full Cartesian product | Broad combinatorial coverage | Multiplies costly Hessians/correlated calculations, includes unsupported combinations, and overweights easy methods |
| Ground-state-only expansion | Fastest path to a larger timing table | Cannot establish effects on response, correlation, solvent, excited states, or real workflows |

The experiment must answer five questions separately:

1. Does the optimized code preserve the requested observable at the original scientific settings?
2. Does it reduce complete-workflow elapsed time versus a qualified GPU control?
3. Does it reduce requested GPU storage, host residency, or scratch requirements?
4. Which operations and transfers dominate each task family after optimization?
5. Which combinations are unsupported, fail accuracy, fail convergence, or exceed available resources?

No claim that every method should accelerate: changes aimed at density-fitted exchange may have little effect on pure DFT, direct SCF, periodic algorithms, or a correlation solver.

## 2. Freeze controls before collecting results

### Source identities

| ID | Definition | Permitted interpretation |
|---|---|---|
| G0 | Untouched GPU Python source at `9af748a71ca8f4881c604332c2f0cc1632e033ac`, using the same compatible native numerical libraries as G1 where applicable | Historical original-GPU comparator; equal-accuracy ratios only for tasks whose observable passes |
| G1 | A new immutable snapshot of the **final reviewed working tree**, including the resident build cap and final review repairs | Primary optimized implementation for the new campaign; do not substitute historical campaign A's `current` snapshot |
| G0a, conditional | G0 with only independently audited correctness repairs needed for a failing task | Optional equal-accuracy GPU control; always label its changes and report separately from G0 |
| CPU-ref | Independently run CPU PySCF with matching physical model and numerical settings | Accuracy oracle; not a CPU performance denominator unless separately benchmarked |

The original Taxol gradient failed the previous CPU gate. Reconfirm the failure at qualification, retain it, and exclude that G0 **gradient/workflow** result from equal-accuracy speed aggregates. Its energy/SCF observable can be separately qualified. If a correction-only G0a can be isolated, verify its patch diff, keep contraction/storage optimizations out of it, and run a distinct comparison. If isolation is not defensible, retain the unqualified historical runtime ratio and do not manufacture an equal-accuracy speedup.

A missing disk-backed representation in G0 is an unsupported capability, not an infinite speedup. Storage comparisons are primarily **within G1** at identical scientific settings.

Freeze a copy of all production Python/native sources, benchmark adapters, configurations, geometries, grids/settings, and CPU reference identities. Record SHA256, source revision plus dirty-tree content, import paths, native library paths, effective contraction backend, compiler/runtime versions, driver, CPU/BLAS thread controls, and device metadata. Validate identities before and after each job. Freeze G0a separately if constructed.

Do not edit the earlier `original_comparison/`, `resident_workspace_results/`, `code_review_results/`, or manuscript evidence. Do not change shared environments or production algorithms as part of a timed campaign. A correctness change requires a new snapshot/campaign ID and requalification of affected cases. Do not inspect or modify the unrelated `record` file.

## 3. Capability audit: distinguish existence from successful execution

Inventory the checked-out implementations, examples, and tests, then execute one tiny CPU/GPU case for each requested task. Record actual class/module, arrays and computational stages, external extensions, and any CPU fallback. A successful `to_gpu()` call alone does not establish GPU execution of a downstream task.

Use these status labels: `qualified`, `accuracy_failed`, `state_mismatch`, `nonconverged`, `unsupported`, `dependency_missing`, `resource_limited`, `timeout`, and `infrastructure_failed`. Separate task capability from scientific acceptance and from measurement mode. A dependency skip is never a pass.

Local starting evidence includes:

- Energy/gradient: `gpu4pyscf/scf/tests/`, `gpu4pyscf/df/tests/`, `gpu4pyscf/grad/tests/`.
- Hessians: `gpu4pyscf/df/hessian/tests/` and `gpu4pyscf/hessian/tests/`.
- Excited states: `gpu4pyscf/tdscf/tests/` and excited-state gradient tests in `gpu4pyscf/df/tests/`.
- Correlation: `gpu4pyscf/mp/tests/test_dfmp2.py`, `test_dfump2.py`, and `gpu4pyscf/cc/tests/test_ccsd.py`.
- Solvent: `gpu4pyscf/solvent/tests/`.
- Properties: `gpu4pyscf/properties/tests/test_polarizability.py`, `test_shielding.py`.
- Optimization: `gpu4pyscf/grad/tests/test_geomopt.py`, `examples/geometry_optimization/`.
- Periodic: `gpu4pyscf/pbc/` tests and `examples/pbc/50-silicon_gamma.py`.

The official README and PySCF GPU page differ in some support descriptions, including ECP and Hessian details. Treat both as orientation; qualification of the pinned local build decides the matrix. MP2/CCSD, properties, and PBC warrant explicit capability gates. Double hybrids, density-Laplacian meta-GGA, and TDDFT Hessians are not default targets. References: [GPU4PySCF README](https://github.com/pyscf/gpu4pyscf#features), [PySCF GPU documentation](https://pyscf.org/user/gpu.html), checked 18 October 2026.

## 4. Core matrix: 60 complete energy-and-gradient configurations

All core jobs request converged energy and analytic gradient, with full available auxiliary and grid response. Restricted HF/RKS is used for closed shells; UHF/UKS is used where specified. Six methods on the anchor set cover distinct kernels: HF, `lda,vwn`, PBE, TPSS, B3LYP, and `wb97x`. Pin LibXC functional identifiers/coefficients so aliases cannot introduce different B3LYP variants.

| Group | Systems | Methods/bases | Configurations |
|---|---|---|---:|
| Size and functional anchors | H2O, benzene, vitamin C, inosine, tamoxifen, Taxol | Six methods; def2-SVP; def2-universal-jkfit | 36 |
| Basis and contraction stress | H2O/cc-pVDZ; benzene/def2-TZVP; vitamin C/def2-TZVP; tamoxifen/def2-TZVP | PBE and B3LYP; matching fixed JKfit choice | 8 |
| Open shells | OH doublet; O2 triplet; NO doublet | PBE and B3LYP; def2-SVP; unrestricted | 6 |
| Charges and diffuse orbitals | Acetate, charge -1, singlet; methylammonium, charge +1, singlet | PBE and B3LYP; def2-SVPD for acetate, def2-SVP for methylammonium | 4 |
| Noncovalent interactions | Water dimer; stacked benzene dimer | PBE and B3LYP; def2-SVP; fixed geometries | 4 |
| Numerical stress geometries | Distorted H2O; stretched N2 at 2.2 angstrom | B3LYP/def2-SVP; specified restricted branch | 2 |
| **Total** | | | **60** |

Use existing organic XYZ files under `benchmarks/molecules/organic/`. Create and freeze missing small-system fixtures and exact charge/spin metadata before runs. H2O/benzene and stress geometries are fixed fixtures, not geometries separately optimized by each backend. Stretched N2 is a numerical/state-selection stress test, not a validation of a single-reference description of dissociation.

For noncovalent cases, primary timing is the dimer energy/gradient workflow. Optional interaction energies require matched monomer/ghost calculations and a separately named task; do not silently change the physical problem through counterpoise treatment.

Do not infer atom or water counts from filenames: the existing `water_clusters/010.xyz` contains 417 atoms, and `gly/gly30.xyz` contains 213. Inventory the XYZ headers, elements, actual AO/auxiliary sizes, occupied/virtual counts, shell contractions, grid points, and estimated tensor sizes before choosing capacity cases.

### Fixed principal settings

- FP64 throughout; no mixed precision, experimental pair packing, changed screening, or learned surrogate.
- SCF `conv_tol=1e-10`, `conv_tol_grad=1e-7`, `max_cycle=100`; unchanged linear-dependence thresholds; `small_rho_cutoff=0`.
- DFT grid level 3, NWChem pruning, with actual backend settings asserted. Save grid size and ordering-independent diagnostics. Any common precomputed-grid diagnostic is separate from primary native-grid construction timing.
- Gradient `auxbasis_response=True`, `grid_response=True` where applicable; no omitted response for speed.
- Four CPU threads for timed GPU jobs; pin and record BLAS/OpenMP settings. CPU reference thread count is separately recorded.
- Identical spin-resolved atomic starting density within each comparison, saved and hashed; fresh fitted integrals in each workflow process. Do not use a converged seed for primary workflow timing.
- Preserve difficult cases that do not converge in the preregistered settings. A level shift, changed initial state, tighter grid, or different solver creates an explicitly separate diagnostic protocol applied to all variants.

## 5. Targeted extensions: 56 configurations, capability gated

| Family | Concrete initial selection | Requested output and timing boundary | Count |
|---|---|---|---:|
| Hessians/frequencies | H2O and benzene at HF/PBE/B3LYP; vitamin C at PBE/B3LYP | SCF plus analytic Hessian; separately report Hessian/CP-SCF and frequency analysis | 8 |
| Excited states | Formaldehyde, benzene, inosine; B3LYP and WB97X; TDA and TDDFT | SCF plus five converged roots and oscillator strengths; response-only cost also reported | 12 |
| Correlation | DF-MP2: H2O, water dimer, benzene, vitamin C; DF-UMP2: OH and NO; CCSD: H2O and formaldehyde | HF preparation, AO-to-MO/integrals, correlation kernel, total energy and correlation energy | 8 |
| Solvation | H2O, vitamin C, inosine, tamoxifen; IEF-PCM and SMD water; B3LYP | Full solvated SCF/gradient, cavity construction, solvent contribution; gas/solvent difference if both qualified | 8 |
| Static properties | H2O, benzene, vitamin C; polarizability and NMR shielding | Ground-state preparation plus response/property; separate CPU preparation if present | 6 |
| Geometry optimization | H2O, formaldehyde, benzene, vitamin C; B3LYP | Complete optimization, accepted steps, energy/gradient calls, final stationary-point criteria | 4 |
| Nonlocal correlation | H2O, benzene, vitamin C, tamoxifen; WB97M-V | Full energy/gradient with VV10 response; record the separate NLC grid and batching | 4 |
| Effective core potentials | CH3I and I2; PBE/def2-SVP with explicit def2 ECP | Energy/gradient, native versus CPU ECP work identified | 2 |
| Dispersion integration | H2O and tamoxifen; B3LYP-D3(BJ), B3LYP-D4 | Full energy/gradient; separately record any CPU dispersion stage | 4 |
| **Total** | | | **56** |

Additional controls:

- Hessians: use full second-order auxiliary response (`auxbasis_response=2` in the local DF implementation), full supported grid response, and matched CP-SCF tolerances. Start at `conv_tol_cpscf=1e-10`; record actual achieved residual. Do not substitute finite differences while labeling the task analytic. Validate selected Hessian columns from central differences of gradients at two displacement sizes.
- TD: target five lowest positive roots; `conv_tol=1e-9`, maximum 100 response iterations. Match states by transition-character overlap, not only sorted energy. Compare near-degenerate subspaces and summed oscillator strengths rather than arbitrary individual eigenvectors. Archive residuals and unstable-reference diagnoses. Excited-state gradients on formaldehyde are an optional follow-on, not included in the 12 response cases.
- Correlation: def2-SVP/def2-SVP-ri for the four restricted MP2 systems; def2-SVP/def2-SVP-ri for OH/NO UMP2; cc-pVDZ for the two CCSD systems. Explicitly set FP64, identical frozen-core policy (`frozen=0`), and `with_t2=False` for primary MP2 energy runs on both backends. A stored-T2 memory study is separate. Correlation-fitting bases are not replaced by JKfit. Match converged HF reference states; diagnose orbital-dependent differences without feeding GPU intermediates into the independent CPU oracle.
- Solvent: IEF-PCM with epsilon 78.3553 and surface Lebedev order 29; SMD `solvent='water'`. Pin radii, surface/cavity settings, dielectric, and library versions; compare actual cavity metadata. Do not attribute a CPU/GPU cavity-definition mismatch to tensor acceleration.
- Optimization: same optimizer/version, starting geometry, constraints, and convergence settings; maximum 50 steps. Record the complete objective-call count and trajectory. Because paths can diverge numerically, add a separate replay of up to ten fixed archived geometries to compare energy/gradient throughput on identical inputs. Actual optimization cost remains the application result.
- Dispersion and ECP: include CPU contributions in application timing. A mixed CPU/GPU task is allowed but explicitly labeled.
- NMR/polarizability: run only with a valid independent CPU implementation or a separately demonstrated finite-field oracle. A missing optional PySCF property extension is `dependency_missing`, not a passed comparison.

### Direct-SCF diagnostic lane, separate from the 116 primary molecular configurations

Six additional energy/gradient cases: H2O and benzene at HF, PBE, and B3LYP without density fitting. These are useful controls for integral-direct kernels and for determining whether gains are specific to the DF path. Compare G0-direct with G1-direct using matching direct settings and an independent direct CPU reference. A direct-versus-DF timing difference is an algorithm/approximation comparison, not an optimization-only speed ratio. Do not require direct and DF energies to agree at the 1e-8 implementation gate. Capability-check direct gradients; do not infer direct Hessian support.

### Periodic feasibility lane, separate from the 116 primary molecular configurations

Four small PBE tasks: primitive Si at Gamma, a 2x2x2 Si supercell at Gamma, primitive Si with a 2x2x2 k mesh, and primitive MgO at Gamma. Freeze lattice vectors, GTH-PBE pseudopotentials, the available matching GTH-DZVP basis definition, k weights, FFT meshes, precision, and electron count. Start with energy; add forces only where the local route is qualified. A symmetry-forced zero-force cell must be supplemented by a small fixed displacement for a meaningful force check.

Compare CPU/GPU with the same periodic integral/DF algorithm; do not use molecular JKfit settings by analogy. If a pseudopotential/basis alias or matching CPU algorithm is unavailable, report the capability failure and defer that exact case. PBC results have a separate report and no assumed gain from the molecular DF changes.

Multi-reference CASSCF/CASCI, spin-flip response, relativistic X2C, transition-state searches, QM/MM, and nonadiabatic dynamics remain possible later campaigns. They are not quietly added to this already substantial first matrix. This is broad coverage, not exhaustive coverage of every repository module.

## 6. Adaptive capacity and storage study

Build a size ladder using actual inventory/AO counts, not filenames:

1. Existing anchors: vitamin C, tamoxifen, Taxol, valinomycin.
2. Three water-cluster fixtures nearest increasing AO targets after inventory.
3. `gly30.xyz`; promote `gly50.xyz` only if the preceding rung leaves adequate headroom.

AO targets for selecting additional rungs are approximately 1500, 2500, and 4000, not promises that a specific file has that many functions. Run PBE and B3LYP at def2-SVP first. Promote one successful medium/large case to def2-TZVP only after the estimated integral/storage sizes fit.

For each admitted G1 case compare:

- Forced GPU-resident fitted integrals, default 4 GiB build target.
- Automatic/host-mapped storage, with the actual resulting placement verified.
- Disk-backed fitted integrals with 4 GiB staging and zero cache; verify scratch deletion.

Keep one default configuration per mode initially. A 1/2/4 GiB budget sweep is a separate diagnostic on one selected large case, not another full Cartesian product. General contractions that bypass the build cap must be identified explicitly.

For successful large cases, collect two separate allocation observations per mode to assess repeatability. Time five resident/host pairs for promoted comparisons; start disk with three repetitions because I/O variability and cost can be high, and extend to five when needed for reporting. Distinguish cache-warm/reused scratch diagnostics from fresh integral workflows. Do not flush global OS caches or create uncontrolled RAM pressure on a shared machine. Without isolated cache/RAM control, label disk results cache-dependent.

Resource admission uses device-reported free storage, OS MemAvailable, cgroup headroom, and scratch capacity as separate constraints. On unified GB10 these are not independent additive budgets. Estimate job requirements conservatively and admit at most 70% of the tightest applicable currently available headroom; retain at least 20% of system RAM as headroom during monitoring. This is an admission policy, not a proof of total physical-RAM bounds. Stop new admissions if observed pressure violates the threshold. Never kill unrelated services. Stop a size ladder after two consecutive resource-limited rungs.

A completed large run without a feasible independent full CPU oracle is a **capacity observation with limited numerical checks**, not a fully accuracy-qualified speed result. Validate smaller members with full CPU references; use matched J/K checks and selected finite-difference gradients as additional diagnostics on the large member. Do not treat them as substitutes for the missing full reference gate.

## 7. Accuracy acceptance, declared before performance runs

All arrays and energies must be finite, and all iterative methods must meet their declared convergence criteria. Reference and candidate must represent the same electronic/spin/root state. Spin contamination is recorded, not silently corrected.

| Observable | Primary proposed acceptance criterion | Additional check |
|---|---|---|
| Molecular SCF total energy | Absolute CPU difference <=1e-8 Eh | Occupation/spin/state identity and stationarity |
| Molecular analytic gradient | Maximum component difference <=1e-6 Eh/Bohr | Selected finite-difference energy derivatives; net force diagnostic |
| Hessian | Maximum component difference <=1e-5 Eh/Bohr^2, retaining existing harness criterion | Symmetry, selected finite-difference gradient columns, mass-weighted eigenvalues |
| Frequencies | Target <=1 cm^-1 for matched vibrational modes above 50 cm^-1 | Report soft/imaginary modes separately; Hessian gate remains primary |
| Excitation energies | <=1e-6 Eh per matched root | Residual, state/subspace matching; oscillator-strength absolute difference <=5e-5 |
| MP2/CCSD | Correlation and total energy difference each <=1e-8 Eh | Reference-state agreement; CCSD residual/iteration record |
| Polarizability | Maximum tensor component difference <=1e-5 atomic units | Tensor symmetry and finite-field checks where feasible |
| NMR shielding | Maximum tensor component difference <=0.1 ppm | Fixed gauge/origin and identical relativistic treatment |
| Solvated energy/gradient | Same 1e-8 Eh /1e-6 Eh/Bohr molecular gates | Cavity/model identity; report gas-to-solution difference |
| Periodic energy/forces | <=1e-8 Eh per simulation cell and <=1e-6 Eh/Bohr | Identical cell, k mesh, grids, pseudopotential, and integral algorithm |
| Optimized structure | Both variants satisfy the same optimizer criteria: maximum force <=3e-4 Eh/Bohr and RMS force <=1e-4 Eh/Bohr; compare final aligned structures and internal coordinates | Re-evaluate both endpoints with independent CPU energies/forces; publish endpoint differences rather than assuming paths coincide |

The energy/gradient/Hessian limits preserve established harness criteria. New-task values are **predeclared targets**, not already demonstrated accuracy. Failure triggers diagnosis; do not loosen a gate after observing a result to count it as a pass. If the oracle itself is not converged sufficiently, create a clearly versioned diagnostic protocol and rerun both variants. Comparing different DF/direct approximations, grids, solvent models, ECPs, or frozen spaces does not isolate GPU numerical error.

For finite differences, use at least two step sizes (e.g. 1e-3 and 5e-4 Bohr for nuclear derivatives; field strengths 5e-4 and 2.5e-4 atomic units for polarizability) and demonstrate a stable error range rather than accepting one accidental cancellation. Tighten diagnostic SCF convergence identically where necessary and report these jobs outside primary timing.

## 8. Timing, allocation, and communication protocol

### Uninstrumented workflow timings

- Fresh process per observation, imports/provenance outside the principal timer. Define setup, ground-state preparation, task-specific work, and total boundaries in the task manifest. Include essential CPU setup and CPU fallback work in the workflow total.
- Initialize the CUDA context and BLAS library with the same tiny untimed operation in each fresh process; record this policy. Do not prebuild DF integrals, grids, or a converged state as a warmup for workflow timing. First-use/import/JIT startup can be measured in the separate full-command lane.
- Synchronize at declared phase boundaries. Do not synchronize every iteration or wrap nested kernels in the timing lane. Lightweight callbacks record iteration counts/energies without device synchronization.
- Core: five paired observations per supported, accuracy-qualified comparator/G1 case. Use a reproducible balanced order (three AB and two BA, randomized with a recorded seed); keep all accepted observations. Extensions: three initial pairs, promoted to five for headline performance claims. Inaccurate controls are measured only in a separately labeled diagnostic lane if useful.
- If either variant has normalized MAD >3%, or the observed ratio lies between 0.95 and 1.05, extend both variants to nine samples when budget permits. This rule applies to small regressions and small gains alike. Otherwise label the estimate unresolved if the required extension cannot be completed.
- Record GPU co-tenancy, utilization, temperature/power/clock data where exposed, and available memory before/after each run. A launch conflict or driver error is archived. Successful slow samples are not removed post hoc because they are inconvenient.
- Report medians, ranges, MAD, sample count, order, and individual paired ratios. Any bootstrap interval with this sample count is descriptive and does not establish hardware-wide significance.
- A separate optional timing mode can measure full command elapsed time including imports. Do not mix it with the principal workflow definition.

### Allocation observations

Use the existing `phase_memory.py` ledger in fresh dedicated processes. Distinguish requested live CuPy bytes, pool live/reserved bytes, process VmHWM, system memory availability, cgroup usage/limits if exposed, and scratch/file-cache effects. Sampled host/system observations record their interval, initially 100 ms, and are not presented as exact transient peaks. Retain the exact allocator ledger and its known native/preexisting exclusions.

Inclusive phase peaks cannot be summed. GPU requested storage plus host RSS is not total physical RAM on GB10. Monitor persistent fitted integrals, metric factors, raw/build buffers, dense AO unpack, occupied transforms, response intermediates, and output lifetimes. Report both the maximum observed storage and the region responsible for it.

### Communication and profiling subset

Profile one small and one large representative of ground-state hybrid, pure DFT, gradient, Hessian, response, and correlation; add one disk case. Use dedicated diagnostic runs, not timing samples.

- Count explicit H2D/D2H copy calls and bytes, synchronization calls, host waits, and CPU-only stages.
- Use Nsight Systems/CUPTI only if available and functional on this device; otherwise record the instrumentation limitation and use narrowly scoped copy API diagnostics. Do not install or alter the execution environment during a frozen campaign.
- Report unified-memory migration or mapped-memory traffic separately when it is observable. Copy API counters alone do not measure all physical traffic on GB10; a wrapper must not claim to observe hidden native transfers.
- Locate repeated transfers in the SCF loop, per-grid-batch transfers, final-only output copies, disk staging, and necessary CPU-property/dispersion stages.
- Record arithmetic/transfer overlap from the timeline where possible. Bytes divided by workflow time is an observed traffic rate, not a hardware bandwidth benchmark.

## 9. Pilot, cost estimation, and execution order

### Pilot: ten representative configurations

1. H2O B3LYP energy/gradient.
2. Benzene TPSS energy/gradient.
3. OH WB97X energy/gradient (an additional capability probe outside the 60-core count).
4. Vitamin C B3LYP energy/gradient.
5. Taxol B3LYP energy/gradient, including its known G0 qualification failure.
6. H2O B3LYP Hessian.
7. Formaldehyde B3LYP TDA, five roots.
8. H2O DF-MP2.
9. H2O B3LYP IEF-PCM energy/gradient.
10. Primitive Si Gamma PBE feasibility case.

Collect/reuse valid independent CPU references, one separate allocation/qualification observation per available GPU variant, and three timing pairs where both variants qualify. Reused references require matching input/model identities and sufficient convergence; they are not CPU timing samples. Initial planning ceilings: 8 GPU-hours and 24 CPU process-hours, sequentially scheduled rather than overlapping CPU reference generation with GPU timing. These ceilings are resource controls, not predicted completion time. A ceiling reached means a partial pilot with explicit outstanding tasks.

Set per-observation timeout to 30 minutes for the initial pilot, 2 hours for promoted expensive GPU tasks, and 4 hours for new CPU reference jobs. A resource forecast can reject a job before launch. Resume through immutable per-job records; do not overwrite successful or failed records. Terminate only the benchmark's own process group and verify that no child remains.

### Cost calculation after the pilot

For a case with two GPU variants and five repeats, approximate GPU execution cost as:

`5 * (median_G0_seconds + median_G1_seconds) + separate_memory_and_profile_cost`.

Add capability probes, CPU references, failure diagnostics, and any G0a as separate terms. Forecast from the measured pilot and actual dimensions within each task family; AO count alone is insufficient for correlation and response.

For orientation only, the historical B3LYP/def2-SVP Taxol G0/G1-era medians imply `(1103.110 + 566.023) * 5 / 3600 = 2.32` GPU-hours for five timing pairs alone. This is not a prediction for the final tree, and G0 fails the gradient gate. It illustrates why a comprehensive campaign should not be promised within a single day without a pilot.

The 60-case core with two fully qualified GPU variants entails up to 600 timing processes +120 separate allocation processes +60 CPU reference jobs = **780 processes**, excluding initial probes, repeats above five, failures, G0a, and extensions. Actual totals decrease for unsupported or rejected controls. Do not call these 780 distinct chemical configurations.

Execution sequence:

1. Freeze registry, source identities, and capability probes.
2. Complete the bounded pilot and publish a cost/coverage ledger.
3. Run core cases in increasing measured resource cost; alternate variants per case.
4. Promote Hessian, TD, MP2, and solvent tasks after tiny-case acceptance.
5. Run property/optimization/NLC/ECP/dispersion extensions after their dependency gates.
6. Run the adaptive capacity lane with resource monitors.
7. Run selected communication profiles and a separate periodic feasibility report.
8. Independently audit all records, publish failures and qualified comparisons, then update the scientific report through a new version.

A second discrete NVIDIA GPU and a genuinely cuTENSOR-enabled compatible environment would be valuable later replications. They are separate hardware/backend strata with their own snapshots; no silent environment replacement or pooling of ratios with GB10 fallback results.

## 10. Benchmark implementation work packages

All new runtime work stays under `benchmarks/single_gpu/qm_suite/`. Reuse existing measurement principles, not the historical drivers' mutable global factories or hard-coded three-molecule settings. The commands below describe the proposed new CLI; it does not exist yet.

### Task 1: Registry and immutable manifests

**Create:** `cases.json`, `manifest.py`, `tests/test_manifest.py`, `fixtures/*.xyz` under the new suite directory.

- [ ] Encode the exact 60 core entries and 56 molecular extension entries with task/method, charge/spin, basis, response flags, tolerance, storage, and geometry hash.
- [ ] Keep four PBC cases and adaptive capacity entries in separate registry groups; mark generated/proposed fixtures as unusable until their exact coordinates and hashes are frozen.
- [ ] Write CPU-only tests rejecting missing charge/spin, inconsistent electron parity, duplicate IDs, unknown tasks, basis/auxiliary confusion, changed hashes, missing observables, and incompatible comparison identities.
- [ ] Verify the explicit 60/56/4 counts and that every admitted case can be dry-run without importing CuPy or initializing CUDA.
- [ ] Save source/native/backend and per-case manifests without modifying old campaign manifests.

### Task 2: Task adapters and scientific oracles

**Create:** `adapters.py`, `compare.py`, `references.py`, `tests/test_compare.py`.

- [ ] Add adapters by family, starting with core SCF/gradient, then Hessian, TD, correlation, solvent, properties, optimization, and PBC.
- [ ] Add fixture-based comparator tests that pass an exact reference and reject a deliberately corrupted energy, gradient, Hessian entry, excitation root, and missing/nonfinite result.
- [ ] Verify state matching, units, Hessian tensor ordering, auxiliary response level 2, and correlation fitting basis selection explicitly.
- [ ] Assert effective settings on constructed objects. Keep CPU oracle runs independent of GPU intermediates; saved common atomic guesses are allowed and identified.
- [ ] Run tiny native CPU/GPU acceptance for each adapter before that family can enter timing scheduling. Do not mark an adapter qualified merely because its Python unit test passes.

### Task 3: Isolated execution and measurement modes

**Create:** `run.py`, `metrics.py`, `tests/test_run.py`.

**Reuse read-only:** `benchmarks/single_gpu/phase_memory.py`.

- [ ] Implement modes `capability`, `reference`, `timing`, `memory`, and `profile`; prevent instrumentation from entering timing mode.
- [ ] Launch an explicit interpreter in a fresh process with controlled import paths, one visible GPU, and pinned threads; validate the imported snapshot path and effective backend.
- [ ] Test with lightweight fake jobs that timing excludes allocation/profiling hooks, callbacks do not synchronize per iteration, timeout retains a record, and failed jobs cannot be overwritten.
- [ ] Restore hooks, close disk files, delete owned scratch storage, and verify cleanup on success, Python exception, and handled timeout.
- [ ] Save actual arrays necessary for later independent comparisons, iterative histories, placement, and stage boundaries; use JSON plus NPZ rather than huge repeated source dictionaries in every result.

### Task 4: Resource-aware scheduling

**Create:** `schedule.py`, `tests/test_schedule.py`.

- [ ] Build a deterministic seeded paired schedule and a dependency graph: reference/capability before performance, pilot before cost extrapolation, preceding size rung before larger capacity work.
- [ ] Unit-test budget exhaustion, resource rejection, process-group cleanup, resume identity, balanced AB/BA ordering, nine-sample extension rules, and retention of failed controls.
- [ ] Add `--dry-run` that emits planned jobs/counts/resources without executing QM.
- [ ] Use a campaign root `benchmarks/single_gpu/qm_suite/results/<campaign-id>/` with append-only per-job records and a separate attempt ID for every retry.

### Task 5: Independent aggregation and report

**Create:** `audit.py`, `summarize.py`, `tests/test_audit.py`, `README.md`.

- [ ] Recompute accuracy from saved CPU/GPU arrays; do not trust stored `passed` flags.
- [ ] Check hashes, settings, backend/storage, convergence, state matching, repetition counts, cleanup, and measurement-mode separation.
- [ ] Add corruption tests: changed gradient entry, missing sample, mismatched grid, wrong auxiliary basis, inaccurate baseline, altered source, and memory sample accidentally included in timing.
- [ ] Generate per-family tables/figures, a complete coverage/failure matrix, resource forecasts, and a list of limitations from audited data only.
- [ ] Reproduce plots/tables in a CPU-only pass. Retain report input hashes and compile/render any final scientific PDF before calling it current.

### Task 6: Pilot and campaign execution

- [ ] Run CPU-only harness tests and dry-run first; inspect enumerated configurations and job budget.
- [ ] Run the ten pilot cases, audit them, and publish a costed remainder with explicit deferred tasks.
- [ ] Execute admitted core/extension/capacity jobs in the declared order; do not mix performance tuning into data collection.
- [ ] Audit final results and publish a new report version with a complete exclusions and failures appendix.

Proposed commands after implementation, from the repository root:

```bash
python -m pytest benchmarks/single_gpu/qm_suite/tests -q
python -m benchmarks.single_gpu.qm_suite.schedule --campaign pilot --dry-run
python -m benchmarks.single_gpu.qm_suite.schedule --campaign pilot --gpu-hours 8 --cpu-hours 24
python -m benchmarks.single_gpu.qm_suite.audit --campaign pilot
python -m benchmarks.single_gpu.qm_suite.summarize --campaign pilot
```

The suite will provide the package entry files needed for these module invocations and document the exact interpreter path. CPU-only collection must not import GPU task adapters eagerly. Do not run the proposed commands against today's tree expecting them to exist.

## 11. Deliverables and completion criteria

Deliverables:

1. Frozen case registry and source/native/reference manifests.
2. Capability matrix showing GPU, CPU fallback, unavailable dependencies, and unsupported combinations per task.
3. Per-case raw arrays and logs with accuracy, timings, allocation observations, and explicit transfer/profile records where available.
4. Median/variation tables for both phase and complete-workflow elapsed time.
5. Memory-versus-time Pareto plots, size scaling by actual AO/auxiliary/occupied/virtual dimensions, and communication breakdowns.
6. Accuracy plots with predeclared thresholds, including failed controls.
7. A report of recommended default storage modes and the next optimization priorities, based on measured remaining costs.
8. Reproducible source/data archive and a new version of the journal-style scientific report.

Publish per-family geometric-mean speed ratios only over shared, scientifically qualified cases, alongside the inclusion denominator and every excluded case. Report resource failures separately; no zero-time substitutions or infinite speedups. Avoid a single headline average combining Hessians, correlation, periodic systems, and cheap properties with arbitrary weights. CPU accuracy jobs are not CPU speed baselines. An optional CPU speed study needs repeated independent runs and a declared thread/affinity policy; do not reuse a single oracle runtime as a tuned CPU denominator. Verification against CPU also does not establish experimental accuracy or basis/grid convergence.

A larger method suite is successful even if some method families show no improvement: identifying regressions and bottlenecks is an intended outcome.

Completion means all scheduled admitted cases have a recorded terminal status, every claimed speed ratio has qualified matching observables and settings, raw data can reproduce the tables, and all deferred cases have explicit reasons. It does not mean every proposed task fits the machine or all extensions are supported.

## 12. Plan review

- The 60 core configurations and 56 molecular extension configurations are explicitly enumerated by group; the six direct-SCF tasks, four periodic tasks, and adaptive capacity rungs are separate.
- The known original-Taxol accuracy failure has a defined treatment; G0a never masquerades as untouched G0.
- Fresh final-tree timing/allocation measurements are required; old source snapshots are not relabeled as final results.
- CPU reference cost, shared-device interference, disk cache, unified-memory accounting, and missing task capabilities are covered.
- This document is a concrete experimental and work-package plan, not an assertion that the proposed harness or benchmarks already exist.
