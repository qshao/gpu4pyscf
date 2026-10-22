# External quantum chemistry qualification

These are opt-in experiments with external dependencies. The original DF force
failures are corrected by the local patch described below. They do not qualify
the neural Skala checkpoint, cuEST binaries, or Mako execution.

Source revisions:

- Microsoft Skala v2026.9: c0487c9a241f652e3bcee9042677da835dde3dda.
- NVIDIA CUDALibrarySamples: 58c811d2d3bccdcea6fcd8a2c45fddd19ca2fb00.
- Microsoft Skala-1.1 model metadata: 99b5ed87e5f69d9216e1f9e30148b922eaea1241.

All sources are under ignored build/vendor. The model checkout contains LFS
pointers, not executable checkpoint data. Shared environments were not modified.

## Reproduce the executed source overlay

Use /home/qshao/miniconda3/envs/dft_oracle/bin/python. Tests load Skala source
from build/vendor/skala/src and append read-only dependencies from labeling.
The opt-in local configuration build/external-qm/extra_site_paths.json contains:

    ["/home/qshao/miniconda3/envs/labeling/lib/python3.11/site-packages"]

Alternatively set QM_EXTRA_SITE to the dependency site-packages path. The report
preflight_imports.json records actual imported paths. This is process-local
dependency reuse, not a clean installation or qualification of a published wheel.
Repository-local venvs were created under build/external-qm, but dependency
downloads could not populate the cuEST environment.

Run from the root using that interpreter:

    python -m pytest benchmarks/external_qm/test_preflight.py benchmarks/external_qm/test_grid_order.py -q
    python -m pytest benchmarks/external_qm/test_skala_pbe.py -q

The first command probes real Torch/CuPy allocations, zero-copy DLPack exchange,
Skala imports and preservation of empty grid blocks. The second compares native
libxc PBE, Skala's Torch PBE on CPU, and the same Torch PBE on GPU for neutral
water and its doublet cation, with and without DF. Grid coordinates/weights are
checked, pruning/dispersion disabled, auxiliary basis explicit, and SCF convergence
required. Gradients include Skala's internally differentiated grid response;
setting the native PySCF grid_response flag on Skala objects is inappropriate.

CPU/GPU gates compare the same Torch functional: 1e-8 Eh energy, 1e-6 Eh/Bohr
gradient. Native libxc results remain a separate independent implementation
comparison. The measured roughly 3.5e-6 Eh energy difference has not been
attributed conclusively.
A central finite difference (oxygen z, 1e-3 Bohr) checks energy/gradient consistency
independently. Only one derivative component is checked per case.

Before patching, the DF finite-difference gates failed at approximately 8e-6 and
9e-6 Eh/Bohr. baseline_summary.json preserves that evidence. They now pass with
the local Skala patch and unchanged thresholds. Assertions are not skipped,
marked expected failures or weakened to advertise qualification.
JSON files retain each backend's outputs and single-call elapsed times. These
timings include first-use overhead and do not establish speedups.

## Local DF force correction

The released Skala integration used direct Coulomb derivatives for DF energies.
patches/skala-v2026.9-df-gradients.patch corrects both CPU and GPU dispatch and
includes auxiliary-basis response. It is already applied to the local vendor
checkout; it has not been installed into any shared environment or published.
For a fresh checkout at the pinned Skala revision, apply from the GPU4PySCF root:

    git -C build/vendor/skala apply "$PWD/benchmarks/external_qm/patches/skala-v2026.9-df-gradients.patch"

Run the broader all-component/two-step force check:

    python -m pytest benchmarks/external_qm/test_skala_df_forces.py -q

This checks distorted water and its doublet cation, two auxiliary bases, and all
nine force components against central differences at 1e-3 and 5e-4 Bohr, followed
by Richardson extrapolation. Reports are df_forces_charge{0,1}.json. CPU/GPU
energy/gradient gates and independent force gates all contribute to final status.
The patch is tested on PySCF 2.13.1 and this GPU4PySCF checkout; older GPU gradient
APIs and custom/range-separated DF backends require separate qualification.

df_diagnosis_before_patch.json is historical frozen-density Coulomb evidence.
df_diagnosis.py can be run explicitly with pytest; subsequent diagnostic output
uses df_diagnosis_current.json to avoid overwriting the historical record.

## Downloads still needed

cuEST needs Python >=3.12; a 3.12 venv exists at build/external-qm/cuest-env.
Once package-index access works, populate it normally:

    build/external-qm/cuest-env/bin/python -m pip install nvidia-cuest-cu13==0.2.2
    build/external-qm/cuest-env/bin/python -m pip install './build/vendor/CUDALibrarySamples/cuEST[cu13]' pytest

Then begin with the official one-electron and CoreDF J/K examples in
build/vendor/CUDALibrarySamples/cuEST/python_examples. Complete matched CPU/GPU
SCF/gradient tests before integration; MP2 examples additionally need nvmath.
AO ordering, normalization, fitting metric, thresholds and workspace accounting
must be checked explicitly. A source audit is not an executed cuEST benchmark.

Skala checkpoints can be obtained using Git LFS or Hugging Face's download CLI
from microsoft/skala-1.1 at revision 99b5ed87e5f69d9216e1f9e30148b922eaea1241:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| skala-1.1-rev1.fun | 2388988 | 7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd |
| skala-1.1-rev1-cuda.fun | 2403409 | f848eae769dca91741a518ae7275d10caac398ab21db649f91bc1f136872f223 |

Retain the release loader's SHA256 verification. Do not point it at the 132-byte
LFS pointer files. Run published neural CPU/GPU integration tests after the real
checkpoints are available; PBE infrastructure tests do not substitute for them.

Mako's paper points to https://doi.org/10.5281/zenodo.15269220. Neither browser
access nor the normal local download succeeded. Its files and licensing remain
unverified. No source or binary execution is claimed.
