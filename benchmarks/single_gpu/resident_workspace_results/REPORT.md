# Bounded resident DF construction

Taxol: 113 atoms, 1123 AOs; B3LYP/def2-SVP, def2-universal-jkfit, level-3 NWChem grids, full grid/auxiliary gradient response, FP64. Resident GPU integrals are forced. The control is the previously optimized, accuracy-corrected GPU build function, before this cap.

Control/current workflow medians: 565.264s / 566.881s (0.9971x; 0.286% more time).

| Allocation phase | Control peak, GB | Candidate peak, GB |
| --- | ---: | ---: |
| setup | 0.000 | 0.000 |
| df_build | 30.100 | 20.051 |
| scf | 30.100 | 20.051 |
| gradient | 16.669 | 16.669 |

Whole-workflow tracked peak: 30.100 to 20.051GB (33.4% lower).

Host address-space peak RSS, measured separately: control 1.476GB, candidate 1.476GB.

Timing samples (seconds): {"control": [565.3397146547213, 565.2638336997479, 565.1942625092342], "candidate": [566.9052484389395, 566.8810347039253, 566.09962443728]}.

Paired speed ratios: 0.9972, 0.9971, 0.9984.
Three pairs describe observed variation; they do not establish a hardware-wide speed guarantee.

Maximum CPU energy error: 2.228e-09Eh; gradient component error: 2.108e-08Eh/Bohr. Acceptance gates remain 1e-8Eh / 1e-6Eh/Bohr.

All ten workflow observations pass, including full small/medium validation and the six large timing workflows. All large runs use the same atomic density and converge in 20 recorded cycles.

20 focused checks and 13 audit checks pass; XML evidence is hashed in summary.json.

Three alternating fresh-process timing pairs; separate fresh phase-allocation observations. Control diagnostic scopes include synchronization and must not be used as speed controls. Only the build function changes. Requested CuPy bytes exclude native/preexisting storage; host RSS is separate and neither measures total physical RAM on unified GB10. The cap applies to splittable nongeneral contractions; general contractions keep their existing path. The pre-change control profile uses its individually hash-pinned archived runner; all other accepted observations assert and record the effective 4GiB build target.

The preliminary candidate memory observation is preserved under pre-runtime-budget-audit/ and excluded because its runner predates the explicit runtime-budget assertion. The exact archived runner is accepted only for the separately pinned pre-change control profile.

A subprocess launcher failed CUDA initialization before calculation; candidate-small-validation-0.log retains that infrastructure failure. Direct launches passed and provide the accepted observations.
