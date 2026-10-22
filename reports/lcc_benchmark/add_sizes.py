"""Add system size (basis functions, atoms) to data/campaign.json from the registry, without any SCF."""
import json
from pathlib import Path
from pyscf import gto
from pyscf.pbc import gto as pgto

ROOT = Path('/scratch/qsh226/gpu4pyscf')
cases = {c['id']: c for c in json.loads((ROOT / 'benchmarks/single_gpu/qm_suite/cases.json').read_text())}
size = {}
for cid, c in cases.items():
    try:
        if c['task'] == 'pbc_energy':
            spec = c['cell']
            cell = pgto.Cell(a=spec['a'], atom=spec['atom'], basis=spec['basis'], pseudo=spec['pseudo'], unit=spec.get('unit', 'Angstrom'),
                             verbose=0, precision=spec.get('precision', 1e-8))
            if 'mesh' in spec:
                cell.mesh = spec['mesh']
            cell.build()
            size[cid] = dict(nao=cell.nao_nr(), natm=cell.natm)
        else:
            mol = gto.M(atom=str(ROOT / c['geometry']), basis=c['basis'], charge=c['charge'], spin=c['spin'],
                        ecp=(c.get('task_options') or {}).get('ecp', {}), unit=c['settings'].get('unit', 'Angstrom'), verbose=0)
            size[cid] = dict(nao=mol.nao_nr(), natm=mol.natm, nelectron=mol.nelectron)
    except Exception as error:
        print('size failed', cid, repr(error)[:100])
data = json.loads((Path(__file__).parent / 'data/campaign.json').read_text())
for r in data['per_case']:
    s = size.get(r['case'])
    if s:
        r['nao'] = s['nao']
        r['natm'] = s['natm']
        if 'nelectron' in s:
            r['nelectron'] = s['nelectron']
(Path(__file__).parent / 'data/campaign.json').write_text(json.dumps(data, indent=1))
print('sizes added for', len(size), 'cases; max AOs', max(s['nao'] for s in size.values()))
