"""CPU-only case registry and immutable, content-addressed source manifests.

Registry acceptance means syntactically frozen inputs, not GPU qualification.
Source roots are explicit snapshots supplied by the caller; G0 is never inferred
from historical campaign output. Manifest files use exclusive creation.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = Path(__file__).with_name('cases.json')
PILOT_IDS = (
    'core-h2o-b3lyp-svp', 'core-benzene-tpss-svp', 'probe-oh-wb97x-svp',
    'core-vitamin_c-b3lyp-svp', 'core-taxol-b3lyp-svp', 'ext-h2o-b3lyp-hessian',
    'ext-formaldehyde-b3lyp-tda', 'ext-h2o-dfmp2', 'ext-h2o-b3lyp-iefpcm',
    'pbc-si-gamma',
)
OBSERVABLES = {
    'energy_gradient': {'energy', 'gradient'},
    'hessian': {'energy', 'hessian', 'frequencies'},
    'tda': {'energy', 'excitation_energy', 'oscillator_strength'},
    'tddft': {'energy', 'excitation_energy', 'oscillator_strength'},
    'dfmp2': {'energy', 'correlation_energy'},
    'dfump2': {'energy', 'correlation_energy'},
    'ccsd': {'energy', 'correlation_energy'},
    'polarizability': {'energy', 'polarizability'},
    'shielding': {'energy', 'shielding'},
    'optimization': {'energy', 'gradient', 'coordinates'},
    'pbc_energy': {'energy'},
}
UNITS = {'energy': 'Eh', 'correlation_energy': 'Eh', 'gradient': 'Eh/Bohr',
         'hessian': 'Eh/Bohr^2', 'frequencies': 'cm^-1', 'coordinates': 'Angstrom',
         'excitation_energy': 'Eh', 'oscillator_strength': 'dimensionless',
         'polarizability': 'atomic units', 'shielding': 'ppm'}
# Elements present in frozen molecular inputs, including archive gly geometries.
CHARGES = dict(zip(('H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca'), range(1,21)))
CHARGES.update(Br=35,I=53)


def digest(path):
    """Streaming SHA256 of a file's actual bytes (including symlink target)."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def xyz_atoms(path):
    lines = Path(path).read_text().splitlines()
    count = int(lines[0])
    atoms = []
    for line in lines[2:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4 or fields[0] not in CHARGES:
            raise ValueError(f'invalid XYZ atom in {path}')
        xyz = [float(v) for v in fields[1:]]
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError('nonfinite geometry')
        atoms.append((fields[0], xyz))
    if len(atoms) != count:
        raise ValueError(f'XYZ header count mismatch: {path}')
    return atoms


def validate_case(case, check_basis=False):
    """Reject inconsistent inputs without importing CuPy or starting CUDA.

    check_basis additionally asks installed CPU PySCF to resolve orbital and
    auxiliary bases; it performs no SCF calculation.
    """
    required = {'id','group','family','task','system','charge','spin','basis',
                'auxbasis','xc','df','settings','task_options'}
    missing = required - case.keys()
    if missing:
        raise ValueError(f'missing fields: {sorted(missing)}')
    if case['task'] not in OBSERVABLES:
        raise ValueError(f'unknown task: {case["task"]}')
    if not OBSERVABLES[case['task']] <= set(case['task_options'].get('observables', [])):
        raise ValueError('missing required observables')
    if any(k in case['basis'].lower() for k in ('jkfit','-ri')):
        raise ValueError('fitting basis used as orbital basis')
    if case['task'] in ('dfmp2','dfump2') and case['auxbasis'] != 'def2-svp-ri':
        raise ValueError('correlation fitting basis must be def2-svp-ri')
    if not isinstance(case['df'], bool):
        raise ValueError('df must be boolean')
    if not isinstance(case['charge'], int) or not isinstance(case['spin'], int) or case['spin'] < 0:
        raise ValueError('charge and spin must be integers; spin nonnegative')
    if case['task'] == 'pbc_energy':
        cell = case.get('cell', {})
        for field in ('a','atom','basis','pseudo','kmesh','mesh','precision','unit'):
            if field not in cell:
                raise ValueError(f'missing cell {field}')
        atoms = cell['atom']
    else:
        path = ROOT / case['geometry']
        if not path.is_relative_to(ROOT):
            raise ValueError('geometry must be repository relative')
        if digest(path) != case.get('geometry_sha256'):
            raise ValueError('geometry hash changed or missing')
        atoms = xyz_atoms(path)
    electrons = sum(CHARGES[e] for e, _ in atoms) - case['charge']
    if electrons < case['spin'] or (electrons - case['spin']) % 2:
        raise ValueError('inconsistent electron/spin parity')
    if check_basis:
        from pyscf import gto
        for element in {e for e, _ in atoms}:
            gto.basis.load(case['basis'], element)
            if case['auxbasis']:
                gto.basis.load(case['auxbasis'], element)
    return case


def load_cases(group=None):
    cases = json.loads(REGISTRY.read_text())
    ids = [case['id'] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate case IDs')
    for case in cases:
        validate_case(case)
    return [case for case in cases if group is None or case['group'] == group]


def get_case(id):
    for case in load_cases():
        if case['id'] == id:
            return case
    raise KeyError(id)


def pilot_cases():
    cases = {case['id']: case for case in load_cases()}
    return [cases[id] for id in PILOT_IDS]


def comparison_identity(case):
    """Hash scientific inputs, excluding only bookkeeping labels."""
    settings = {k:v for k,v in case.items() if k not in ('id','group','family')}
    return hashlib.sha256(_canonical(settings).encode()).hexdigest()


def _source_files(root):
    root = Path(root).resolve()
    # Scope only the supplied production package; never inspect unrelated record.
    package = root if root.name == 'gpu4pyscf' else root / 'gpu4pyscf'
    if not package.is_dir():
        raise ValueError(f'missing production package: {package}')
    return {str(p.relative_to(root)): digest(p) for p in sorted(package.rglob('*'))
            if p.is_file() and (p.suffix in ('.py','.so','.cu','.c','.h','.cpp','.cuh') or '.so.' in p.name)}


def write_manifest(case_list, variant_paths, output_path):
    """Freeze input cases and explicit source/native roots; never overwrite.

    Return schema_version/cases/case_identities/variants/registry_sha256/units.
    Variants map labels to {'root':absolute path,'files':relative-path:SHA256}.
    Runtime backend/library versions belong to per-job records.
    """
    cases = copy.deepcopy(list(case_list))
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('duplicate case IDs')
    for case in cases:
        validate_case(case)
    manifest = {'schema_version':1, 'cases':cases,
                'case_identities':{c['id']:comparison_identity(c) for c in cases},
                'registry_sha256':digest(REGISTRY), 'units':UNITS,
                'variants':{name:{'root':str(Path(path).resolve()),'files':_source_files(path)}
                            for name,path in variant_paths.items()}}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(manifest, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    return manifest


def freeze_manifest(output_root, variants):
    return write_manifest(load_cases(), variants, Path(output_root) / 'manifest.json')


def verify_manifest(manifest):
    """Reject changed source files, native libraries, registry, or geometries."""
    if digest(REGISTRY) != manifest['registry_sha256']:
        raise ValueError('registry changed')
    for case in manifest['cases']:
        validate_case(case)
        if comparison_identity(case) != manifest['case_identities'][case['id']]:
            raise ValueError('case identity changed')
    for variant in manifest['variants'].values():
        if _source_files(variant['root']) != variant['files']:
            raise ValueError('source/native content changed')
    return True


def select_capacity_water_rungs(inventory, targets=(1500, 2500, 4000)):
    """Choose distinct increasing water rungs from measured CPU AO inventory.

    Input rows contain geometry and actual nao; names never establish size.
    Selection proposes inputs only. Resource admission and source freezing must
    still happen before execution. Insufficient candidates yield fewer rungs.
    """
    remaining = [dict(row) for row in inventory]
    selected, previous = [], 0
    for target in targets:
        candidates = [row for row in remaining if row['nao'] > previous]
        if not candidates:
            break
        chosen = min(candidates, key=lambda row: (abs(row['nao'] - target), row['nao'], row['geometry']))
        remaining.remove(chosen)
        selected.append({**chosen, 'target_nao':target, 'admission_required':True})
        previous = chosen['nao']
    return selected
