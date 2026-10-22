"""Audit only exchange_sweep.json; never discover bounded-build records by glob."""
import json
import math
from pathlib import Path
import statistics

import numpy as np

from original_comparison import BASE, ROOT, digest, python_sources

OUT = BASE/'resident_workspace_results'
HELPERS = {'test_resident_exchange_sweep.py','original_comparison.py',
           'comprehensive.py','molecular.py','gradient_snapshot.py','seeded_reference.py'}
LIMITATIONS = ('Fixed-density J/K only: no SCF, forces or full-workflow gain is '
    'measured. Workspace is an estimate of dense AO unpack plus one occupied '
    'transform buffer, not measured total RAM; it excludes persistent CDERI, '
    'outputs, factors, allocator reservation and native-library workspaces. '
    'Three timing pairs do not establish a hardware-wide guarantee. The audit '
    'rechecks recorded finite full-matrix maximum errors; J/K matrices were not '
    'persisted for a separate numerical recomputation.')


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest(path))


def expected_settings(root):
    return dict(atom=str(root/'benchmarks/molecules/organic/113_Taxol.xyz'),
        unit='Angstrom',basis='def2-svp',auxbasis='def2-universal-jkfit',
        conv_tol=1e-10,conv_tol_grad=1e-7,grid_level=3,grid_pruning='nwchem',
        small_rho_cutoff=0.,grid_response=True,auxbasis_response=True,
        cpu_threads=4,max_cycle=100)


def validate(r,root=ROOT,base=BASE):
    """Check raw evidence and current files without importing PySCF or CuPy."""
    assert r['status']=='passed' and r['bindings_restored'] is True
    assert r['case']=='taxol_svp_b3lyp' and r['method']=='B3LYP'
    assert r['settings']==expected_settings(root)
    assert r['control_block_size']==256 and r['requested_block_sizes']==[128,256,512,1024]
    assert r['tolerance']==dict(absolute=1e-9,relative=0.,require_finite=True)
    assert set(r['helpers'])==HELPERS
    assert all(digest(base/name)==sha for name,sha in r['helpers'].items())
    assert r['source_hashes'] and all(digest(root/name)==sha
                                     for name,sha in r['source_hashes'].items())
    assert r['algorithm_sources'] and r['algorithm_sources']==python_sources(root/'gpu4pyscf')
    native_sources = {str(p.relative_to(root)):digest(p)
        for p in sorted((root/'gpu4pyscf/lib').rglob('*'))
        if p.is_file() and 'deps' not in p.parts and (
            p.suffix in ('.cu','.cuh','.c','.h','.cpp','.cmake') or p.name=='CMakeLists.txt')}
    assert native_sources and r['native_source_hashes']==native_sources
    files = sorted(set((root/'gpu4pyscf/lib').glob('*.so*')) |
                   set((root/'gpu4pyscf/lib/deps/lib').glob('*.so*')))
    native = {str(p.relative_to(root)):identity(p) for p in files if p.is_file()}
    assert native and r['native_file_identities']==native
    assert set(r['native_libraries'])=={'cupy_helper','libxc','gdft','vhf_rys'}
    assert all(value in native.values() for value in r['native_libraries'].values())
    inputs = dict(geometry=Path(r['settings']['atom']),
        cpu_reference=base/'comprehensive_results/taxol_svp_b3lyp_seeded_cpu.json',
        corrected_profile=base/'optimization_results/corrected_profile.json',
        orbitals=base/'optimization_results/corrected_profile.npz')
    assert r['inputs']=={name:identity(path) for name,path in inputs.items()}
    assert r['geometry_sha256']==digest(inputs['geometry'])
    cpu = json.loads(inputs['cpu_reference'].read_text())
    profile = json.loads(inputs['corrected_profile'].read_text())
    assert cpu['status']==profile['status']=='passed'
    assert profile['case']==r['case']
    assert cpu['geometry_sha256']==profile['geometry_sha256']==r['geometry_sha256']
    assert profile['settings']==r['settings']==profile['result']['settings']
    assert profile['orbitals']==identity(inputs['orbitals'])
    assert profile['cpu_reference_sha256']==digest(inputs['cpu_reference'])
    left,right = cpu['result'],profile['result']
    assert left['converged'] is True and right['converged'] is True
    assert left['method']==right['method']=='B3LYP'
    assert {k:v for k,v in left['settings'].items() if k!='cpu_threads'}=={
            k:v for k,v in r['settings'].items() if k!='cpu_threads'}
    energies = [left['energy_hartree'],right['energy_hartree']]
    gradients = [np.asarray(v['gradient_hartree_per_bohr']) for v in [left,right]]
    assert all(math.isfinite(v) for v in energies)
    assert gradients[0].shape==gradients[1].shape==(r['natm'],3)
    assert all(np.isfinite(v).all() for v in gradients)
    comparison = dict(energy_error_hartree=abs(energies[0]-energies[1]),
                     gradient_max_error=float(np.max(np.abs(gradients[0]-gradients[1]))))
    comparison['passed'] = (comparison['energy_error_hartree']<=1e-8
                            and comparison['gradient_max_error']<=1e-6)
    assert comparison['passed'] and r['snapshot_cpu_comparison']==comparison
    nao,nocc,naux = [r[k] for k in ['nao','nocc','naux']]
    assert all(type(n) is int and n>0 for n in [nao,nocc,naux]) and nocc<=nao
    assert r['unpatched_control_block_size']==min(256,naux)
    with np.load(inputs['orbitals'],allow_pickle=False) as archive:
        coeff,occ,energy = [archive[k] for k in ['mo_coeff','mo_occ','mo_energy']]
        assert coeff.shape==(nao,len(occ)) and energy.shape==occ.shape
        assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in [coeff,occ,energy])
        assert np.isin(occ,[0.,2.]).all() and np.count_nonzero(occ)==nocc
        assert abs(occ.sum()-r['nelectron'])<1e-10
        assert math.isfinite(float(archive['energy_hartree']))
        assert float(archive['energy_hartree'])==r['snapshot_energy_hartree']
        assert abs(r['snapshot_energy_hartree']-right['energy_hartree'])<1e-10
    assert len(r['density_sha256'])==64 and int(r['density_sha256'],16)>=0
    assert math.isfinite(r['df_build_seconds']) and r['df_build_seconds']>0
    assert r['resident_cderi']['shape'][0]==naux
    assert r['resident_cderi']['nbytes']==math.prod(r['resident_cderi']['shape'])*8
    assert np.asarray(r['geometry_bohr']).shape==(r['natm'],3)
    assert np.isfinite(r['geometry_bohr']).all() and len(r['atom_symbols'])==r['natm']
    plan = [(size,'warmup',None) for size in [256,128,512,1024]]
    pairs = {size:[] for size in [128,512,1024]}
    for candidate in pairs:
        for index in range(3):
            order = [256,candidate] if index%2==0 else [candidate,256]
            tag = dict(candidate=candidate,index=index,order=order)
            start = len(plan)
            plan.extend((size,'paired',tag) for size in order)
            pairs[candidate].append((start+order.index(candidate),start+order.index(256)))
    assert len(r['samples'])==len(plan)==22
    assert set(r['variants'])=={'128','256','512','1024'}
    for row,(size,phase,pair) in zip(r['samples'],plan):
        assert (row['block_size'],row['phase'],row['pair'])==(size,phase,pair)
        assert math.isfinite(row['seconds']) and row['seconds']>0 and row['passed'] is True
        block = min(size,naux)
        lengths = [min(block,naux-p) for p in range(0,naux,block)]
        assert row['observed_block_lengths']==lengths
        assert row['observed_get_blksize'] and all(v==block for v in row['observed_get_blksize'])
        assert set(row['comparison'])=={'J','K'}
        for check in row['comparison'].values():
            assert check['finite'] is True and check['passed'] is True
            assert math.isfinite(check['max_abs_error']) and 0<=check['max_abs_error']<=1e-9
    variants = {}
    for size in [128,256,512,1024]:
        variant = r['variants'][str(size)]
        block = min(size,naux)
        lengths = [min(block,naux-p) for p in range(0,naux,block)]
        assert variant['effective_block_size']==block
        assert variant['block_lengths']==lengths and variant['blocks_per_call']==len(lengths)
        assert variant['passed'] is True
        rows = [row for row in r['samples'] if row['block_size']==size]
        errors = {k:max(row['comparison'][k]['max_abs_error'] for row in rows) for k in ['J','K']}
        assert variant['max_abs_error']==errors
        times = [row['seconds'] for row in rows if row['phase']=='paired']
        controls = [] if size==256 else [r['samples'][b]['seconds'] for a,b in pairs[size]]
        ratios = [1.]*9 if size==256 else [r['samples'][a]['seconds']/r['samples'][b]['seconds']
                                          for a,b in pairs[size]]
        assert variant['seconds']==times and variant['paired_control_seconds']==controls
        assert variant['normalized_seconds']==ratios
        median,normalized = statistics.median(times),statistics.median(ratios)
        assert variant['median_seconds']==median and variant['median_normalized_seconds']==normalized
        workspace = dict(dense_ao_unpack=8*block*nao*nao,
                         occupied_transform=8*block*nao*nocc,sum=8*block*nao*(nao+nocc))
        assert variant['workspace_bytes']==workspace
        variants[str(size)] = dict(median_jk_seconds=median,
            median_paired_normalized_seconds=normalized,paired_jk_speed_ratio=1/normalized,
            timing_samples_seconds=times,paired_control_seconds=controls,
            paired_normalized_seconds=ratios,estimated_workspace_bytes=workspace,
            blocks_per_call=len(lengths),maximum_jk_error=errors)
    return dict(status='passed',case=r['case'],sample_count=22,
        variants=variants,snapshot_cpu_comparison=comparison,limitations=LIMITATIONS)


def main(root=ROOT,base=BASE,out=OUT):
    source = out/'exchange_sweep.json'
    targets = [out/'exchange_summary.json',out/'exchange_REPORT.md']
    for path in targets:
        if path.exists():
            raise FileExistsError('Archive previous summary explicitly: '+str(path))
    summary = validate(json.loads(source.read_text()),root,base)
    summary.update(input_report=identity(source),validator=identity(__file__))
    lines = ['# Resident Taxol DF J/K block-size sweep','',
        '| Auxiliary block | Median J/K, s | Paired J/K speed ratio vs 256 | Estimated unpack + transform, GiB |',
        '| ---: | ---: | ---: | ---: |']
    for size,row in summary['variants'].items():
        lines.append(f"| {size} | {row['median_jk_seconds']:.6f} | "
            f"{row['paired_jk_speed_ratio']:.4f} | {row['estimated_workspace_bytes']['sum']/1024**3:.3f} |")
    lines += ['', 'Speed ratio is the reciprocal of the median of three candidate/control '
        'time ratios. The 256 median uses all nine paired control samples; warmups '
        'are excluded. Each of 22 calls passes the finite full-J/K component '
        'error gate of absolute 1e-9, relative zero.','',LIMITATIONS,'',
        'Source, native libraries, helpers, inputs, original scientific settings '
        'and saved-orbital CPU comparison were independently rechecked.','']
    with targets[0].open('x') as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
        stream.write('\n')
    with targets[1].open('x') as stream:
        stream.write('\n'.join(lines))
    print('Audited 22 J/K calls:',targets[0],targets[1])


if __name__=='__main__':
    main()
