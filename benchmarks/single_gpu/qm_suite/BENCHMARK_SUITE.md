# GPU4PySCF Comprehensive Quantum-Chemistry Benchmark Suite

**Protocol version:** 1.0 draft  
**Date:** 18 October 2026  
**Target platform:** NVIDIA DGX Spark, one GB10 GPU, compute capability 12.1 (SM121)  
**Primary objectives:** accuracy-preserving acceleration, reduced memory demand, and reduced avoidable CPU–GPU communication  
**Execution status:** specification and harness preparation; no GPU measurements from this new suite have been collected.

This document defines the experiments to perform. Proposed task coverage does not imply that every task is already qualified on the local GPU build. The existing optimization report contains the earlier measured results; the measurements defined here constitute a new campaign.

Related files:

- [Machine-readable case registry](cases.json).
- [Geometry fixture descriptions](fixtures/README.md).
- [Detailed implementation and execution plan](../../../docs/superpowers/plans/2026-10-18-broad-qm-benchmark.md).
- [Earlier optimization report](../../../reports/gpu4pyscf_optimization/manuscript.pdf).

## 1. Scientific questions

The suite will answer the following questions separately:

1. Which QM tasks become faster with the optimized GPU4PySCF?
2. How do complete-workflow and individual-stage performance change with system size, basis, spin, and functional?
3. How much GPU allocation, host residency, and scratch storage does each task require?
4. When does resident, mapped-host, or disk-backed storage provide the best capacity–time tradeoff?
5. Which CPU–GPU transfers and synchronization points remain avoidable?
6. Do all claimed improvements preserve independently checked energies, derivatives, response quantities, and convergence?
7. Which combinations are unsupported, unavailable in the current environment, inaccurate, nonconvergent, or resource limited?

The study concerns computational implementation accuracy. Agreement with CPU PySCF does not demonstrate experimental accuracy of a density functional or convergence to the complete-basis limit.

## 2. Comparators and interpretation

| Label | Definition | Purpose |
|---|---|---|
| **G0** | Untouched GPU Python implementation at revision `9af748a71ca8f4881c604332c2f0cc1632e033ac` | Original GPU control |
| **G1** | A newly frozen snapshot of the final reviewed optimized working tree | Optimized GPU implementation |
| **G0a**, optional | G0 with independently audited correctness-only repairs | Equal-accuracy comparison when a separable repair is necessary |
| **CPU-ref** | Independently converged CPU PySCF calculation of the same physical problem | Scientific reference; not automatically a CPU performance baseline |

G0 and G1 use the same compatible native numerical libraries where their algorithms permit this. Their source files, shared libraries, environments, and actual import locations must be identified by hashes. A change to a source snapshot or numerical setting starts a new campaign identity.

The earlier Taxol G0 gradient failed the CPU accuracy gate. Consequently:

- A G0 result that fails the requested observable cannot provide an equal-accuracy speedup denominator.
- Its elapsed time may be retained as an explicitly accuracy-unqualified observation.
- A correction-only G0a must remain separately labeled; it must not be described as untouched G0.
- An unsupported task, failed run, or out-of-memory control does not imply an infinite speedup.

Historical `current` snapshots are not interchangeable with the final G1 snapshot. Component speed ratios from different campaigns must not be multiplied into a claimed workflow acceleration.

## 3. Suite size and organization

A **configuration** specifies one system, physical model, numerical protocol, and task. Repetitions, backend variants, and measurement modes are additional observations of that configuration.

| Group | Configurations | Role |
|---|---:|---|
| Ground-state energy and gradient core | 60 | Main performance and accuracy comparison |
| Hessians and vibrational analysis | 8 | Second derivatives and response |
| TDA/TDDFT excited states | 12 | Iterative electronic response |
| MP2, unrestricted MP2, and CCSD | 8 | Correlated electronic structure |
| PCM and SMD solvation | 8 | Environment-dependent workloads |
| Polarizability and NMR shielding | 6 | Molecular response properties |
| Geometry optimization | 4 | Complete application workflows |
| Nonlocal correlation | 4 | Additional quadrature and response |
| Effective core potentials | 2 | Heavy-element integral treatment |
| Dispersion integration | 4 | Mixed CPU/GPU application overhead |
| Direct SCF diagnostic lane | 6 | Integral-direct control |
| Periodic feasibility lane | 4 | Separate periodic implementation assessment |
| **Fixed main suite** | **126** | Core, extensions, direct, and periodic tasks |
| Additional OH/WB97X pilot probe | 1 | Open-shell, range-separated pilot coverage |
| Registered capacity configurations | 12 | Six systems at PBE and B3LYP |
| **Currently registered entries** | **139** | Includes the probe and capacity configurations |

Additional water-cluster capacity rungs will be selected from an actual AO inventory. They are not included in the count of 139. Repeating a capacity configuration with three storage modes also does not create three different molecular geometries.

Each requested task begins with a small capability and accuracy check before admission to larger performance runs. Repository tests or a successful `to_gpu()` conversion alone do not qualify an entire task family.

## 4. Systems and numerical settings

### 4.1 Molecular coverage

The anchor series is:

| System | Atoms | Intended coverage |
|---|---:|---|
| H2O | 3 | Small calculation, setup overhead, derivatives and properties |
| Benzene | 12 | Symmetry, delocalization, excited states |
| Vitamin C | 20 | Existing small organic benchmark; polar molecule |
| Inosine | 31 | Intermediate size and excited-state/solvent coverage |
| Tamoxifen | 57 | Existing medium organic benchmark |
| Taxol | 113 | Existing large organic benchmark and auxiliary-metric stress |

Additional systems include OH, O2, NO, acetate, methylammonium, water and benzene dimers, distorted H2O, stretched N2, formaldehyde, CH3I, and I2. Large-system capacity cases add valinomycin and glycine-chain geometries.

Geometries are fixed across backends. Existing archive geometries are retained; newly supplied fixtures are explicitly documented as deterministic test geometries. They must not be presented as experimentally determined or independently optimized structures without supporting evidence.

### 4.2 Shared principal settings

| Quantity | Setting |
|---|---|
| Floating-point arithmetic | FP64; complex128 where the method requires complex arrays |
| Main orbital basis | def2-SVP |
| Basis stress cases | def2-TZVP, cc-pVDZ, and def2-SVPD as specified |
| SCF auxiliary basis | def2-universal-jkfit for the molecular DF SCF lane |
| MP2 auxiliary basis | def2-SVP-ri; distinct from SCF JK fitting |
| SCF energy convergence | `conv_tol = 1e-10` Eh |
| SCF orbital-gradient convergence | `conv_tol_grad = 1e-7` |
| Maximum SCF cycles | 100 |
| Main DFT grid | Level 3, NWChem pruning |
| Small-density grid cutoff | `small_rho_cutoff = 0` |
| Analytic gradients | Full available grid and auxiliary-basis response |
| DF Hessian auxiliary response | Second order, `auxbasis_response = 2` |
| Initial CP-SCF tolerance | `conv_tol_cpscf = 1e-10` |
| Timed GPU job CPU threads | Four; BLAS/OpenMP controls recorded |
| Resident DF build target | 4 GiB; its actual applicability must be recorded |
| Experimental integral cache | Disabled for primary measurements |

Screening and linear-dependence thresholds remain fixed. No mixed precision, relaxed convergence, reduced response, experimental occupied-pair packing, or learned surrogate is introduced to improve the reported timings.

The 4 GiB construction target is a local workspace target. It is not a limit on persistent integrals, all native workspaces, total process memory, or physical RAM; generally contracted bases may follow a different build path.

### 4.3 Functional definitions

Use pinned LibXC expressions instead of relying on ambiguous aliases:

| Label | Registry expression |
|---|---|
| HF | `HF` |
| LDA | `LDA_X,LDA_C_VWN` |
| PBE | `GGA_X_PBE,GGA_C_PBE` |
| TPSS | `MGGA_X_TPSS,MGGA_C_TPSS` |
| B3LYP | `0.20*HF + 0.08*LDA_X + 0.72*GGA_X_B88, 0.81*GGA_C_LYP + 0.19*LDA_C_VWN` |
| WB97X | `HYB_GGA_XC_WB97X` |
| WB97M-V | `HYB_MGGA_XC_WB97M_V` |

The explicit B3LYP expression fixes the VWN choice. Before reusing an older reference, verify its exact functional definition rather than assuming that a historical `B3LYP` alias denotes the same expression.

### 4.4 Initial states

Prepare a common independent CPU atomic MINAO density for each configuration and store its exact array and file hashes. All variants load that same artifact; it is not a saved converged GPU state.

This avoids incidental floating-point differences when atomic guesses are regenerated independently. The workflow timer includes loading the density, uploading it when required, and subsequent setup. Preparing the common benchmark input is outside the timer and must be disclosed. Grids, fitted integrals, and converged SCF states are not prebuilt for primary workflow timings.

For open shells, preserve separate alpha and beta densities. Record occupations, spin, spin contamination where available, convergence, and state diagnostics. A different electronic state is not an acceptable substitute for an accuracy-matched calculation.

## 5. Ground-state core: 60 configurations

| Subgroup | Systems | Methods and basis | Count |
|---|---|---|---:|
| Anchor series | H2O, benzene, vitamin C, inosine, tamoxifen, Taxol | HF, LDA, PBE, TPSS, B3LYP, WB97X; def2-SVP | 36 |
| Basis/contraction stress | H2O/cc-pVDZ; benzene, vitamin C, tamoxifen/def2-TZVP | PBE and B3LYP | 8 |
| Open shells | OH doublet, O2 triplet, NO doublet | Unrestricted PBE and B3LYP; def2-SVP | 6 |
| Charged systems | Acetate -1 singlet; methylammonium +1 singlet | PBE/B3LYP; acetate def2-SVPD, methylammonium def2-SVP | 4 |
| Noncovalent systems | Water dimer and stacked benzene dimer | PBE/B3LYP; def2-SVP | 4 |
| Difficult geometries | Distorted H2O; restricted stretched N2 at 2.2 angstrom | B3LYP/def2-SVP | 2 |
| **Total** | | | **60** |

The primary task is converged SCF followed by full analytic gradients. Record total workflow, setup, DF construction where separately observable, SCF, and gradient times.

Stretched N2 tests numerical robustness and state control; it does not validate a single-reference description of bond dissociation. Dimer interaction energies, counterpoise corrections, or alternative spin branches require separately identified protocols.

## 6. Additional QM task families

### 6.1 Hessians and frequencies — eight configurations

- H2O and benzene at HF, PBE, and B3LYP: six configurations.
- Vitamin C at PBE and B3LYP: two configurations.
- Def2-SVP, full available grid response, and second-order auxiliary response.

Measure ground-state preparation, Hessian/CP-SCF work, vibrational analysis, and complete workflow separately. Store the full Hessian in a declared tensor ordering and report the convention for imaginary frequencies. Check selected columns using central differences of analytic gradients at two displacement sizes.

### 6.2 Excited states — twelve configurations

- Formaldehyde, benzene, and inosine.
- B3LYP and WB97X.
- TDA and TDDFT: `3 systems × 2 functionals × 2 methods = 12`.
- Five requested roots, response tolerance `1e-9`, and maximum 100 response iterations.

Record SCF preparation, response time, iteration/convergence information, excitation energies, oscillator strengths, and transition character. Match roots by phase-invariant AO transition-density character. Near-degenerate states require subspace and summed-strength comparisons; sorted root energies alone are insufficient.

Excited-state gradients and spin-flip response are possible later extensions and are not counted in these twelve cases.

### 6.3 Correlated methods — eight configurations

| Method | Systems | Basis and fitting |
|---|---|---|
| Restricted DF-MP2 | H2O, water dimer, benzene, vitamin C | def2-SVP with def2-SVP-ri correlation fitting |
| DF-UMP2 | OH and NO | def2-SVP with def2-SVP-ri correlation fitting |
| CCSD | H2O and formaldehyde | cc-pVDZ; direct RHF reference for the selected in-core route |

Use no frozen core (`frozen=0`) and FP64. Primary MP2 energy measurements do not retain T2 amplitudes (`with_t2=False`) on either backend. Retaining amplitudes is a separate memory experiment.

Save HF reference energy, correlation energy, and **correlated total energy** explicitly:

`E_total = E_HF + E_correlation`.

Measure reference preparation, integral/AO-to-MO work, correlation work, and total workflow. Record any CPU preparation inside an otherwise GPU-oriented implementation.

### 6.4 Solvation — eight configurations

- H2O, vitamin C, inosine, and tamoxifen.
- B3LYP with IEF-PCM water and SMD water.
- IEF-PCM: dielectric 78.3553, surface Lebedev order 29.
- SMD: the explicit `water` solvent specification.

Pin cavity radii, surface settings, solvent parameters, and library versions. Measure cavity/setup, solvated SCF, solvent response, full gradient, and total workflow. Gas-to-solution energy differences require qualified gas and solvent calculations at the same geometry.

### 6.5 Response properties — six configurations

H2O, benzene, and vitamin C, each with:

- Static polarizability.
- NMR shielding.

Include ground-state preparation and response/property work. Fix gauge, origin, and relativistic conventions. A missing CPU property extension is a dependency failure; it does not qualify a GPU result. Finite-field checks may supplement an available analytic reference.

### 6.6 Geometry optimization — four configurations

Optimize H2O, formaldehyde, benzene, and vitamin C at B3LYP/def2-SVP using identical optimizer/version, initial coordinates, constraints, and stopping criteria, with at most 50 steps.

Report total elapsed time, accepted steps, energy/gradient evaluations, convergence, and final stationary-point properties. Compare aligned endpoints and internal coordinates rather than expecting identical trajectories.

A separate replay of up to ten common archived geometries will measure energy/gradient throughput on identical inputs. Replay timing is not the complete optimization runtime.

### 6.7 Nonlocal correlation — four configurations

Run WB97M-V energy and gradient calculations on H2O, benzene, vitamin C, and tamoxifen. Record NLC grids, quadrature, batching, and full supported response explicitly. The ordinary DF build workspace target does not bound every NLC allocation.

### 6.8 ECPs — two configurations

Run PBE/def2-SVP energy and gradients for CH3I and I2 with the corresponding explicit iodine ECP definition. Verify the actual ECP/electron count and identify native GPU and CPU stages. A basis label alone is insufficient proof that the intended ECP was used.

### 6.9 Dispersion — four configurations

H2O and tamoxifen at B3LYP-D3(BJ) and B3LYP-D4. Fix dispersion parameters and versions. Include any CPU dispersion contribution in application time and report it separately.

## 7. Direct and periodic lanes

### 7.1 Direct SCF — six configurations

H2O and benzene at HF, PBE, and B3LYP, without density fitting. Compare G0-direct with G1-direct and an independent direct CPU reference.

A direct-versus-DF difference is an algorithm/approximation comparison, not an optimization-only speedup. Direct and DF energies are not required to agree at the implementation-accuracy threshold. Direct gradients require local capability checks; direct Hessians are not assumed supported.

### 7.2 Periodic feasibility — four configurations

1. Primitive Si at Gamma.
2. A 2×2×2 Si supercell at Gamma.
3. Primitive Si with a 2×2×2 k mesh.
4. Primitive MgO at Gamma.

Use PBE with frozen lattice vectors, GTH-PBE pseudopotentials, the specified GTH-DZVP basis, FFT mesh, precision, k weights, and electron count. Match periodic integral algorithms between CPU and GPU. These cases begin with energy; forces require separate qualification and a fixed symmetry-breaking displacement when an ideal cell would yield trivial zero forces.

Periodic results remain separate from molecular DF scaling. Their presence in the registry is a feasibility target, not evidence of an already measured acceleration.

## 8. Large-system capacity and storage

The registered capacity systems are vitamin C, tamoxifen, Taxol, valinomycin, `gly30`, and `gly50`, each at PBE and B3LYP/def2-SVP: twelve configurations.

Select additional water-cluster rungs near increasing AO targets of approximately 1500, 2500, and 4000 after measuring the actual AO inventory. File names do not determine size: existing `gly30.xyz` has 213 atoms, and `water_clusters/010.xyz` has 417 atoms.

For each admitted G1 case, compare:

| Storage mode | Required observation |
|---|---|
| GPU resident | Confirm persistent fitted-integral placement and effective build budget |
| Automatic/mapped host | Record actual chosen placement and mapped-memory behavior |
| Disk backed | Record staging budget, cache policy, scratch size, transfer volume, and cleanup |

Default disk staging is 4 GiB and integral cache is zero. A 1/2/4 GiB budget sweep is a separate diagnostic on one selected large case, rather than a full cross-product over every task.

Start with smaller rungs and promote only after resource and accuracy checks. Stop the ladder after two consecutive resource-limited rungs. Collect two separate allocation observations for promoted large configurations.

Resource checks must consider current GPU free memory, system MemAvailable, applicable cgroup headroom, and scratch capacity separately. Admit conservatively against the tightest applicable constraint and monitor pressure during the owned job. Do not add unified-memory counters, flush shared OS caches, or terminate unrelated services.

Without controlled cache/RAM conditions, disk timings remain OS-cache-dependent. A large calculation lacking a feasible independent full CPU reference may establish capacity with limited diagnostics, but it cannot enter fully accuracy-qualified speed aggregates.

## 9. Accuracy gates

The following thresholds are declared before performance measurements. The new-task thresholds are targets to demonstrate, not already established results.

| Observable | Acceptance criterion |
|---|---|
| Molecular SCF energy | Absolute CPU difference ≤1e-8 Eh |
| Analytic gradient | Maximum component difference ≤1e-6 Eh/Bohr |
| Analytic Hessian | Maximum component difference ≤1e-5 Eh/Bohr² |
| Vibrational frequencies | Target ≤1 cm⁻¹ for matched modes above 50 cm⁻¹; report soft/imaginary modes separately |
| Excitation energy | ≤1e-6 Eh per character-matched root |
| Oscillator strength | Absolute difference ≤5e-5; use group sums for degenerate subspaces |
| MP2/CCSD energy | Correlation and correlated total energy differences each ≤1e-8 Eh |
| Polarizability | Maximum tensor component difference ≤1e-5 atomic units |
| NMR shielding | Maximum tensor component difference ≤0.1 ppm |
| Solvated energy/gradient | Same molecular energy/gradient gates, with matching cavity and physical model |
| Periodic energy/forces | ≤1e-8 Eh per simulation cell and ≤1e-6 Eh/Bohr for qualified forces |
| Optimized endpoint | Both calculations meet maximum force ≤3e-4 Eh/Bohr and RMS force ≤1e-4 Eh/Bohr |

All required arrays must be finite and have the correct shape and units. Every iterative stage must satisfy its own convergence requirement. Incorrect state/root selection is reported separately from numerical error.

Useful supplementary diagnostics include net force, Hessian symmetry, mass-weighted vibrational modes, spin contamination, occupation consistency, stationarity, and selected finite differences. Nuclear finite-difference checks should use at least two step sizes, initially 1e-3 and 5e-4 Bohr. Polarizability checks may use fields of 5e-4 and 2.5e-4 atomic units.

Do not relax a threshold after observing a failure to count that result as successful. A revised physical model or numerical protocol requires a separately identified comparison applied to both variants.

## 10. Timing protocol

### 10.1 Complete-workflow boundary

Primary workflow time includes:

- Required CPU/GPU object setup.
- Loading the common atomic density and uploading it where necessary.
- Fresh grids and fresh fitted-integral construction.
- SCF and the requested downstream QM task.
- Necessary result conversion defined by the task protocol.

Imports, provenance hashing, common input preparation, minimal CUDA/BLAS initialization, and final cleanup are outside the principal timer. Cleanup is still required and verified. Record a separate full-command time if first-use and import costs are of interest.

No converged-state, DF-integral, or grid warmup is used in the primary fresh-process workflow. Synchronize at declared timing boundaries; do not add synchronization to every iteration in the timing lane.

### 10.2 Repetitions and ordering

- One fresh process per observation.
- Core: five paired timing observations per qualified GPU comparison.
- Pilot/extensions: three initial pairs; promote to five for headline comparisons.
- Balance AB/BA order using a recorded random seed and explicit pair slots.
- Preserve pair indices across retries and resumes.
- If normalized median absolute deviation exceeds 3%, or the observed ratio lies between 0.95 and 1.05, extend both variants toward nine observations when budget permits.
- Retain valid slow samples. Archive infrastructure failures and exclude them by declared status, not by whether their timing is inconvenient.

Only complete pairs with both observables qualified contribute to a paired headline ratio. Retain unmatched samples separately. Five accepted observations on each side are insufficient if they belong to disjoint pair slots.

Report median, range, median absolute deviation, repetitions, order, and individual paired ratios. Small-sample intervals are descriptive and do not establish hardware-wide statistical significance.

## 11. Memory and communication measurements

### 11.1 Separate allocation lane

Run allocation observations in fresh processes separate from workflow timings. Report:

- Peak requested live CuPy bytes and inclusive phase peaks.
- Pool live and reserved bytes where available.
- Host address-space VmHWM.
- Sampled system availability and cgroup usage/limits.
- Persistent fitted-integral size, scratch file size, and staging configuration.
- Scope exclusions and sampling interval.

CuPy allocation tracking excludes native/preexisting storage it does not observe. Phase peaks overlap and must not be summed. On unified-memory GB10, GPU requested storage plus host RSS does not equal total physical RAM.

Use decimal GB for measured byte quantities and binary GiB for configured budgets. Label estimates separately from measured peaks.

### 11.2 Communication and profiling lane

Select small and large representatives from hybrid SCF, pure DFT, gradients, Hessians, response, correlation, and disk storage. Dedicated profiling runs will examine:

- Explicit H2D/D2H bytes and copy calls.
- Synchronization calls and host waits.
- Repeated transfers inside iterative loops or grid batches.
- Mapped/unified-memory migration where observable.
- CPU-only stages and arithmetic/transfer overlap.

Use Nsight Systems/CUPTI only where available and qualified. Copy-API counters do not measure all physical traffic on GB10. A normal run labeled `profile` without instrumentation does not constitute communication evidence.

## 12. Pilot and compute budget

The ten-case pilot is:

| Order | Case ID | Purpose |
|---:|---|---|
| 1 | `core-h2o-b3lyp-svp` | Small hybrid energy/gradient |
| 2 | `core-benzene-tpss-svp` | Meta-GGA energy/gradient |
| 3 | `probe-oh-wb97x-svp` | Open-shell range-separated functional |
| 4 | `core-vitamin_c-b3lyp-svp` | Existing small organic anchor |
| 5 | `core-taxol-b3lyp-svp` | Large anchor and known G0 gradient issue |
| 6 | `ext-h2o-b3lyp-hessian` | Analytic second derivatives |
| 7 | `ext-formaldehyde-b3lyp-tda` | Five-root excited-state response |
| 8 | `ext-h2o-dfmp2` | Correlated energy and correlation fitting |
| 9 | `ext-h2o-b3lyp-iefpcm` | Solvent energy/gradient |
| 10 | `pbc-si-gamma` | Periodic feasibility |

Initial planning ceilings are **8 GPU-hours and 24 CPU process-hours**. These are ceilings, not completion estimates. CPU reference generation does not overlap timed GPU work on the shared machine.

The pilot first qualifies each observable, then collects separate allocation observations and uninstrumented timing pairs for admitted controls. Initial density/input preparation, probes, retries, and profiling must also appear in the cost ledger, even when outside the workflow timing boundary.

For two GPU variants, a per-case estimate is:

`timing_cost ≈ repetitions × (median_G0 + median_G1)`

Add memory observations, CPU references, setup, failures, profiling, and any G0a comparison separately. Automatic nine-sample extensions must be included when presenting an upper bound.

The 60-case core with two available qualified GPU variants and five repeats entails up to 600 timing processes, 120 separate allocation processes, and 60 CPU reference jobs, before probes and extensions. Unsupported or rejected controls reduce the admitted count; failure diagnostics may increase attempted jobs.

Use pilot measurements to cost the remaining suite instead of assuming that all 126 configurations fit in a single day.

## 13. Record format, audit, and reporting

Every observation must preserve:

- Campaign, configuration, variant, mode, attempt, and pair-slot identifiers.
- Frozen scientific inputs and source/native/helper hashes.
- Actual imported implementation, device/backend, library versions, and effective settings.
- Common density identity and independent CPU reference identity.
- Required numerical outputs and convergence/state diagnostics.
- Task-stage and workflow times with explicit boundaries.
- Allocation, placement, scratch, and communication evidence for their respective modes.
- Terminal status and any traceback or resource-admission reason.

Allowed outcomes include `qualified`, `accuracy_failed`, `state_mismatch`, `nonconverged`, `unsupported`, `dependency_missing`, `resource_limited`, `timeout`, and `infrastructure_failed`. A successfully generated CPU reference is labeled separately from a CPU/GPU accuracy pass.

An independent CPU-only audit must recompute numerical errors from saved arrays, validate source/input/reference identities, verify backend provenance, reject instrumented timing samples, and check complete pair slots. Stored `passed` flags alone are insufficient.

Final deliverables:

1. Coverage/failure matrix with inclusion denominators.
2. Accuracy-qualified per-case and per-family timing tables.
3. Workflow decomposition and remaining bottleneck profiles.
4. Memory-versus-time Pareto plots and size scaling.
5. Communication breakdowns where instrumentation supports them.
6. Storage-mode recommendations and capacity limits with explicit exclusions.
7. Source/data archive and a new version of the scientific report.

Per-family geometric means may include only shared qualified cases. Do not combine unrelated task families into a single headline average with arbitrary weights. CPU oracle times are not tuned CPU performance measurements.

## 14. Current readiness and execution gates

This Markdown file is the benchmark specification. The case registry, CPU task adapters, and scientific comparison logic have been prepared and exercised with CPU-side tests. GPU execution of the new suite remains unqualified.

Before admitting the pilot, complete the remaining harness integration checks, including scheduler safety, owned-process cleanup, immutable request/result provenance, paired resumption, resource admission, and package/CLI execution. Adaptive capacity scheduling and communication instrumentation require their own execution qualification; their inclusion in this protocol is not a claim that they have already run.

At the most recent environment check, the GB10 was not accessible from the restricted agent execution session. This is separate from the DGX Spark hardware configuration. Execution requires a GPU-visible session that passes both `nvidia-smi` and a native CuPy operation.

Further benchmark execution is on hold while this specification is prepared, as requested.

## Appendix A. Complete fixed-case inventory

The tables below are transcribed from `cases.json`. The **Basis** column is authoritative; an abbreviated case ID suffix does not override the declared basis (for example, acetate uses def2-SVPD).


### A.1 Ground-state core

**Configurations: 60.**

| Case ID | System | Task | Method | Basis | Charge | 2S |
|---|---|---|---|---|---:|---:|
| `core-acetate-b3lyp-svp` | acetate | energy_gradient | B3LYP | def2-svpd | -1 | 0 |
| `core-acetate-pbe-svp` | acetate | energy_gradient | PBE | def2-svpd | -1 | 0 |
| `core-benzene-b3lyp-svp` | benzene | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-benzene-b3lyp-tzvp` | benzene | energy_gradient | B3LYP | def2-tzvp | 0 | 0 |
| `core-benzene-hf-svp` | benzene | energy_gradient | HF | def2-svp | 0 | 0 |
| `core-benzene-lda-svp` | benzene | energy_gradient | LDA | def2-svp | 0 | 0 |
| `core-benzene-pbe-svp` | benzene | energy_gradient | PBE | def2-svp | 0 | 0 |
| `core-benzene-pbe-tzvp` | benzene | energy_gradient | PBE | def2-tzvp | 0 | 0 |
| `core-benzene-tpss-svp` | benzene | energy_gradient | TPSS | def2-svp | 0 | 0 |
| `core-benzene-wb97x-svp` | benzene | energy_gradient | WB97X | def2-svp | 0 | 0 |
| `core-benzene_dimer-b3lyp-svp` | benzene_dimer | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-benzene_dimer-pbe-svp` | benzene_dimer | energy_gradient | PBE | def2-svp | 0 | 0 |
| `core-distorted_h2o-b3lyp-svp` | distorted_h2o | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-h2o-b3lyp-pvdz` | h2o | energy_gradient | B3LYP | cc-pvdz | 0 | 0 |
| `core-h2o-b3lyp-svp` | h2o | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-h2o-hf-svp` | h2o | energy_gradient | HF | def2-svp | 0 | 0 |
| `core-h2o-lda-svp` | h2o | energy_gradient | LDA | def2-svp | 0 | 0 |
| `core-h2o-pbe-pvdz` | h2o | energy_gradient | PBE | cc-pvdz | 0 | 0 |
| `core-h2o-pbe-svp` | h2o | energy_gradient | PBE | def2-svp | 0 | 0 |
| `core-h2o-tpss-svp` | h2o | energy_gradient | TPSS | def2-svp | 0 | 0 |
| `core-h2o-wb97x-svp` | h2o | energy_gradient | WB97X | def2-svp | 0 | 0 |
| `core-inosine-b3lyp-svp` | inosine | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-inosine-hf-svp` | inosine | energy_gradient | HF | def2-svp | 0 | 0 |
| `core-inosine-lda-svp` | inosine | energy_gradient | LDA | def2-svp | 0 | 0 |
| `core-inosine-pbe-svp` | inosine | energy_gradient | PBE | def2-svp | 0 | 0 |
| `core-inosine-tpss-svp` | inosine | energy_gradient | TPSS | def2-svp | 0 | 0 |
| `core-inosine-wb97x-svp` | inosine | energy_gradient | WB97X | def2-svp | 0 | 0 |
| `core-methylammonium-b3lyp-svp` | methylammonium | energy_gradient | B3LYP | def2-svp | 1 | 0 |
| `core-methylammonium-pbe-svp` | methylammonium | energy_gradient | PBE | def2-svp | 1 | 0 |
| `core-no-b3lyp-svp` | no | energy_gradient | B3LYP | def2-svp | 0 | 1 |
| `core-no-pbe-svp` | no | energy_gradient | PBE | def2-svp | 0 | 1 |
| `core-o2-b3lyp-svp` | o2 | energy_gradient | B3LYP | def2-svp | 0 | 2 |
| `core-o2-pbe-svp` | o2 | energy_gradient | PBE | def2-svp | 0 | 2 |
| `core-oh-b3lyp-svp` | oh | energy_gradient | B3LYP | def2-svp | 0 | 1 |
| `core-oh-pbe-svp` | oh | energy_gradient | PBE | def2-svp | 0 | 1 |
| `core-stretched_n2-b3lyp-svp` | stretched_n2 | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-tamoxifen-b3lyp-svp` | tamoxifen | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-tamoxifen-b3lyp-tzvp` | tamoxifen | energy_gradient | B3LYP | def2-tzvp | 0 | 0 |
| `core-tamoxifen-hf-svp` | tamoxifen | energy_gradient | HF | def2-svp | 0 | 0 |
| `core-tamoxifen-lda-svp` | tamoxifen | energy_gradient | LDA | def2-svp | 0 | 0 |
| `core-tamoxifen-pbe-svp` | tamoxifen | energy_gradient | PBE | def2-svp | 0 | 0 |
| `core-tamoxifen-pbe-tzvp` | tamoxifen | energy_gradient | PBE | def2-tzvp | 0 | 0 |
| `core-tamoxifen-tpss-svp` | tamoxifen | energy_gradient | TPSS | def2-svp | 0 | 0 |
| `core-tamoxifen-wb97x-svp` | tamoxifen | energy_gradient | WB97X | def2-svp | 0 | 0 |
| `core-taxol-b3lyp-svp` | taxol | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-taxol-hf-svp` | taxol | energy_gradient | HF | def2-svp | 0 | 0 |
| `core-taxol-lda-svp` | taxol | energy_gradient | LDA | def2-svp | 0 | 0 |
| `core-taxol-pbe-svp` | taxol | energy_gradient | PBE | def2-svp | 0 | 0 |
| `core-taxol-tpss-svp` | taxol | energy_gradient | TPSS | def2-svp | 0 | 0 |
| `core-taxol-wb97x-svp` | taxol | energy_gradient | WB97X | def2-svp | 0 | 0 |
| `core-vitamin_c-b3lyp-svp` | vitamin_c | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-vitamin_c-b3lyp-tzvp` | vitamin_c | energy_gradient | B3LYP | def2-tzvp | 0 | 0 |
| `core-vitamin_c-hf-svp` | vitamin_c | energy_gradient | HF | def2-svp | 0 | 0 |
| `core-vitamin_c-lda-svp` | vitamin_c | energy_gradient | LDA | def2-svp | 0 | 0 |
| `core-vitamin_c-pbe-svp` | vitamin_c | energy_gradient | PBE | def2-svp | 0 | 0 |
| `core-vitamin_c-pbe-tzvp` | vitamin_c | energy_gradient | PBE | def2-tzvp | 0 | 0 |
| `core-vitamin_c-tpss-svp` | vitamin_c | energy_gradient | TPSS | def2-svp | 0 | 0 |
| `core-vitamin_c-wb97x-svp` | vitamin_c | energy_gradient | WB97X | def2-svp | 0 | 0 |
| `core-water_dimer-b3lyp-svp` | water_dimer | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `core-water_dimer-pbe-svp` | water_dimer | energy_gradient | PBE | def2-svp | 0 | 0 |

### A.2 Molecular extensions

**Configurations: 56.**

| Case ID | System | Task | Method | Basis | Charge | 2S |
|---|---|---|---|---|---:|---:|
| `ext-benzene-b3lyp-hessian` | benzene | hessian | B3LYP | def2-svp | 0 | 0 |
| `ext-benzene-b3lyp-tda` | benzene | tda | B3LYP | def2-svp | 0 | 0 |
| `ext-benzene-b3lyp-tddft` | benzene | tddft | B3LYP | def2-svp | 0 | 0 |
| `ext-benzene-dfmp2` | benzene | dfmp2 | HF | def2-svp | 0 | 0 |
| `ext-benzene-hf-hessian` | benzene | hessian | HF | def2-svp | 0 | 0 |
| `ext-benzene-optimization` | benzene | optimization | B3LYP | def2-svp | 0 | 0 |
| `ext-benzene-pbe-hessian` | benzene | hessian | PBE | def2-svp | 0 | 0 |
| `ext-benzene-polarizability` | benzene | polarizability | B3LYP | def2-svp | 0 | 0 |
| `ext-benzene-shielding` | benzene | shielding | B3LYP | def2-svp | 0 | 0 |
| `ext-benzene-wb97mv` | benzene | energy_gradient | WB97M-V | def2-svp | 0 | 0 |
| `ext-benzene-wb97x-tda` | benzene | tda | WB97X | def2-svp | 0 | 0 |
| `ext-benzene-wb97x-tddft` | benzene | tddft | WB97X | def2-svp | 0 | 0 |
| `ext-ch3i-pbe-ecp` | ch3i | energy_gradient | PBE / ECP | def2-svp | 0 | 0 |
| `ext-formaldehyde-b3lyp-tda` | formaldehyde | tda | B3LYP | def2-svp | 0 | 0 |
| `ext-formaldehyde-b3lyp-tddft` | formaldehyde | tddft | B3LYP | def2-svp | 0 | 0 |
| `ext-formaldehyde-ccsd` | formaldehyde | ccsd | HF | cc-pvdz | 0 | 0 |
| `ext-formaldehyde-optimization` | formaldehyde | optimization | B3LYP | def2-svp | 0 | 0 |
| `ext-formaldehyde-wb97x-tda` | formaldehyde | tda | WB97X | def2-svp | 0 | 0 |
| `ext-formaldehyde-wb97x-tddft` | formaldehyde | tddft | WB97X | def2-svp | 0 | 0 |
| `ext-h2o-b3lyp-d3bj` | h2o | energy_gradient | B3LYP / d3bj | def2-svp | 0 | 0 |
| `ext-h2o-b3lyp-d4` | h2o | energy_gradient | B3LYP / d4 | def2-svp | 0 | 0 |
| `ext-h2o-b3lyp-hessian` | h2o | hessian | B3LYP | def2-svp | 0 | 0 |
| `ext-h2o-b3lyp-iefpcm` | h2o | energy_gradient | B3LYP / PCM | def2-svp | 0 | 0 |
| `ext-h2o-b3lyp-smd` | h2o | energy_gradient | B3LYP / SMD | def2-svp | 0 | 0 |
| `ext-h2o-ccsd` | h2o | ccsd | HF | cc-pvdz | 0 | 0 |
| `ext-h2o-dfmp2` | h2o | dfmp2 | HF | def2-svp | 0 | 0 |
| `ext-h2o-hf-hessian` | h2o | hessian | HF | def2-svp | 0 | 0 |
| `ext-h2o-optimization` | h2o | optimization | B3LYP | def2-svp | 0 | 0 |
| `ext-h2o-pbe-hessian` | h2o | hessian | PBE | def2-svp | 0 | 0 |
| `ext-h2o-polarizability` | h2o | polarizability | B3LYP | def2-svp | 0 | 0 |
| `ext-h2o-shielding` | h2o | shielding | B3LYP | def2-svp | 0 | 0 |
| `ext-h2o-wb97mv` | h2o | energy_gradient | WB97M-V | def2-svp | 0 | 0 |
| `ext-i2-pbe-ecp` | i2 | energy_gradient | PBE / ECP | def2-svp | 0 | 0 |
| `ext-inosine-b3lyp-iefpcm` | inosine | energy_gradient | B3LYP / PCM | def2-svp | 0 | 0 |
| `ext-inosine-b3lyp-smd` | inosine | energy_gradient | B3LYP / SMD | def2-svp | 0 | 0 |
| `ext-inosine-b3lyp-tda` | inosine | tda | B3LYP | def2-svp | 0 | 0 |
| `ext-inosine-b3lyp-tddft` | inosine | tddft | B3LYP | def2-svp | 0 | 0 |
| `ext-inosine-wb97x-tda` | inosine | tda | WB97X | def2-svp | 0 | 0 |
| `ext-inosine-wb97x-tddft` | inosine | tddft | WB97X | def2-svp | 0 | 0 |
| `ext-no-dfump2` | no | dfump2 | HF | def2-svp | 0 | 1 |
| `ext-oh-dfump2` | oh | dfump2 | HF | def2-svp | 0 | 1 |
| `ext-tamoxifen-b3lyp-d3bj` | tamoxifen | energy_gradient | B3LYP / d3bj | def2-svp | 0 | 0 |
| `ext-tamoxifen-b3lyp-d4` | tamoxifen | energy_gradient | B3LYP / d4 | def2-svp | 0 | 0 |
| `ext-tamoxifen-b3lyp-iefpcm` | tamoxifen | energy_gradient | B3LYP / PCM | def2-svp | 0 | 0 |
| `ext-tamoxifen-b3lyp-smd` | tamoxifen | energy_gradient | B3LYP / SMD | def2-svp | 0 | 0 |
| `ext-tamoxifen-wb97mv` | tamoxifen | energy_gradient | WB97M-V | def2-svp | 0 | 0 |
| `ext-vitamin_c-b3lyp-hessian` | vitamin_c | hessian | B3LYP | def2-svp | 0 | 0 |
| `ext-vitamin_c-b3lyp-iefpcm` | vitamin_c | energy_gradient | B3LYP / PCM | def2-svp | 0 | 0 |
| `ext-vitamin_c-b3lyp-smd` | vitamin_c | energy_gradient | B3LYP / SMD | def2-svp | 0 | 0 |
| `ext-vitamin_c-dfmp2` | vitamin_c | dfmp2 | HF | def2-svp | 0 | 0 |
| `ext-vitamin_c-optimization` | vitamin_c | optimization | B3LYP | def2-svp | 0 | 0 |
| `ext-vitamin_c-pbe-hessian` | vitamin_c | hessian | PBE | def2-svp | 0 | 0 |
| `ext-vitamin_c-polarizability` | vitamin_c | polarizability | B3LYP | def2-svp | 0 | 0 |
| `ext-vitamin_c-shielding` | vitamin_c | shielding | B3LYP | def2-svp | 0 | 0 |
| `ext-vitamin_c-wb97mv` | vitamin_c | energy_gradient | WB97M-V | def2-svp | 0 | 0 |
| `ext-water_dimer-dfmp2` | water_dimer | dfmp2 | HF | def2-svp | 0 | 0 |

### A.3 Direct SCF

**Configurations: 6.**

| Case ID | System | Task | Method | Basis | Charge | 2S |
|---|---|---|---|---|---:|---:|
| `direct-benzene-b3lyp` | benzene | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `direct-benzene-hf` | benzene | energy_gradient | HF | def2-svp | 0 | 0 |
| `direct-benzene-pbe` | benzene | energy_gradient | PBE | def2-svp | 0 | 0 |
| `direct-h2o-b3lyp` | h2o | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `direct-h2o-hf` | h2o | energy_gradient | HF | def2-svp | 0 | 0 |
| `direct-h2o-pbe` | h2o | energy_gradient | PBE | def2-svp | 0 | 0 |

### A.4 Periodic feasibility

**Configurations: 4.**

| Case ID | System | Task | Method | Basis | Charge | 2S |
|---|---|---|---|---|---:|---:|
| `pbc-mgo-gamma` | mgo | pbc_energy | PBE | gth-dzvp | 0 | 0 |
| `pbc-si-gamma` | si | pbc_energy | PBE | gth-dzvp | 0 | 0 |
| `pbc-si-k222` | si | pbc_energy | PBE | gth-dzvp | 0 | 0 |
| `pbc-si-supercell-gamma` | si_supercell | pbc_energy | PBE | gth-dzvp | 0 | 0 |

## Appendix B. Additional pilot probe

**Configurations: 1.**

| Case ID | System | Task | Method | Basis | Charge | 2S |
|---|---|---|---|---|---:|---:|
| `probe-oh-wb97x-svp` | oh | energy_gradient | WB97X | def2-svp | 0 | 1 |

## Appendix C. Registered capacity configurations

**Configurations: 12.**

| Case ID | System | Task | Method | Basis | Charge | 2S |
|---|---|---|---|---|---:|---:|
| `capacity-gly30-b3lyp-svp` | gly30 | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `capacity-gly30-pbe-svp` | gly30 | energy_gradient | PBE | def2-svp | 0 | 0 |
| `capacity-gly50-b3lyp-svp` | gly50 | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `capacity-gly50-pbe-svp` | gly50 | energy_gradient | PBE | def2-svp | 0 | 0 |
| `capacity-tamoxifen-b3lyp-svp` | tamoxifen | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `capacity-tamoxifen-pbe-svp` | tamoxifen | energy_gradient | PBE | def2-svp | 0 | 0 |
| `capacity-taxol-b3lyp-svp` | taxol | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `capacity-taxol-pbe-svp` | taxol | energy_gradient | PBE | def2-svp | 0 | 0 |
| `capacity-valinomycin-b3lyp-svp` | valinomycin | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `capacity-valinomycin-pbe-svp` | valinomycin | energy_gradient | PBE | def2-svp | 0 | 0 |
| `capacity-vitamin_c-b3lyp-svp` | vitamin_c | energy_gradient | B3LYP | def2-svp | 0 | 0 |
| `capacity-vitamin_c-pbe-svp` | vitamin_c | energy_gradient | PBE | def2-svp | 0 | 0 |

The twelve capacity entries precede storage-mode expansion and adaptive water-cluster selection. They are not twelve additional independent size families.

## Registry provenance

Registry file: `cases.json`  
SHA256: `d77a9fbd7ecaebe244f6de4d481b09ee7ed5733012c676863d5af1b49c344422`

The inventory was checked against the current registry during document preparation. This hash identifies configuration data, not successful GPU execution.
