"""Create a new source-frozen campaign without changing the development branch."""
from pathlib import Path
import json
import shutil

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
# Native libraries are architecture specific; the default is this checkout, an override selects another build
# (for example the sm70/80/90 build used on LCC). Python sources always come from ROOT / the frozen original.
NATIVE=Path(__import__('os').environ.get('QM_SUITE_NATIVE_ROOT',str(ROOT)))
GPU_TARGET=__import__('os').environ.get('QM_SUITE_GPU_TARGET','DGX Spark / NVIDIA GB10, SM121')


def prepare(output,cases,policy=None):
    from manifest import write_manifest
    from run import digest,write_once
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    variants={}
    for label,source in [('G0',ROOT/'benchmarks/single_gpu/original_comparison/original/gpu4pyscf'),('G1',ROOT/'gpu4pyscf')]:
        dest=output/'snapshots'/label/'gpu4pyscf'
        shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so','*.so.*','deps'))
        for native in (NATIVE/'gpu4pyscf/lib').glob('*.so'):
            (dest/'lib'/native.name).symlink_to(native.resolve())
        if (NATIVE/'gpu4pyscf/lib/deps').exists():
            (dest/'lib/deps').symlink_to((NATIVE/'gpu4pyscf/lib/deps').resolve(),target_is_directory=True)
        variants[label]=str(dest.parent)
    import copy
    from references import prepare_density
    cases=copy.deepcopy(cases);initials={}
    for case in cases:
        initial=prepare_density(case,output/'inputs'/(case['id']+'.npy'))
        case['task_options']['initial_density_file']=initial['initial_density_file']
        case['task_options']['initial_density_file_sha256']=initial['initial_density_file_sha256']
        initials[case['id']]=initial
    sources=write_manifest(cases,variants,output/'source_manifest.json')
    helpers={str(p):digest(p) for p in sorted(BASE.glob('*.py'))}
    helpers[str(BASE.parent/'phase_memory.py')]=digest(BASE.parent/'phase_memory.py')
    campaign={'schema_version':1,'sources':sources,'helper_identities':helpers,
       'native_identities':{str(p.resolve()):digest(p) for p in sorted(set((NATIVE/'gpu4pyscf/lib/deps').resolve().rglob('*')) | set(q.resolve() for q in (NATIVE/'gpu4pyscf/lib').glob('*.so'))) if p.is_file() and '.so' in p.name},
       'schedule_policy':policy,'initial_densities':initials,'input_identities':{d['initial_density_file']:d['initial_density_file_sha256'] for d in initials.values()},
       'scope':'Final reviewed G1 versus untouched G0; shared native numerical libraries. CPU-only jobs are references, not GPU timing samples.',
       'timing_policy':'Frozen independent atomic MINAO input preparation excluded; loading and uploading included in setup. Same tiny CUDA/BLAS initialization, no fitted-integral/grid/SCF warmup; imports and provenance outside workflow timer; allocation/profile lanes separate.',
       'memory_policy':'Requested live CuPy bytes excludes native/preexisting storage. Host RSS and GPU ledger are nonadditive on unified-memory devices (GB10); discrete GPUs have separate device memory.',
       'gpu_target':GPU_TARGET,'native_root':str(NATIVE),'cpu_threads':4,
       'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    write_once(output/'campaign.json',campaign)
    return campaign
