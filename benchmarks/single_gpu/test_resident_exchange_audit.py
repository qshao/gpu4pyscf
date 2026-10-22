"""CPU-only synthetic evidence checks; these are not Taxol measurements."""
import copy
import json
import statistics

import numpy as np
import pytest

import summarize_resident_exchange_sweep as audit


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path
    base = root/'benchmarks/single_gpu'
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(value)
        return path
    write(root/'gpu4pyscf/__init__.py','# synthetic source\n')
    write(root/'gpu4pyscf/lib/kernel.cu','// synthetic native source\n')
    for name in audit.HELPERS:
        write(base/name,'# synthetic helper\n')
    native = {}
    for name in ['cupy_helper','libxc','gdft','vhf_rys']:
        path = write(root/f'gpu4pyscf/lib/{name}.so','synthetic library')
        native[name] = audit.identity(path)
    geometry = write(root/'benchmarks/molecules/organic/113_Taxol.xyz','1\nsynthetic\nHe 0 0 0\n')
    config = audit.expected_settings(root)
    result = dict(settings=config,method='B3LYP',converged=True,
                  energy_hartree=-1.,gradient_hartree_per_bohr=[[0.,0.,0.]])
    cpu_path = base/'comprehensive_results/taxol_svp_b3lyp_seeded_cpu.json'
    write(cpu_path,json.dumps(dict(status='passed',result=result,
                                  geometry_sha256=audit.digest(geometry))))
    orbital_path = base/'optimization_results/corrected_profile.npz'
    orbital_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(orbital_path,mo_coeff=np.eye(2),mo_occ=np.array([2.,0.]),
             mo_energy=np.array([-1.,0.]),energy_hartree=-1.)
    comparison = dict(passed=True,energy_error_hartree=0.,gradient_max_error=0.)
    profile_path = orbital_path.with_suffix('.json')
    write(profile_path,json.dumps(dict(status='passed',case='taxol_svp_b3lyp',
        settings=config,result=result,geometry_sha256=audit.digest(geometry),
        cpu_reference_sha256=audit.digest(cpu_path),orbitals=audit.identity(orbital_path))))
    report = dict(status='passed',case='taxol_svp_b3lyp',method='B3LYP',settings=config,
        bindings_restored=True,control_block_size=256,unpatched_control_block_size=256,
        requested_block_sizes=[128,256,512,1024],nao=2,nocc=1,naux=1100,
        nelectron=2,natm=1,charge=0,spin=0,geometry_bohr=[[0.,0.,0.]],atom_symbols=['He'],
        density_sha256='0'*64,snapshot_energy_hartree=-1.,df_build_seconds=1.,
        resident_cderi=dict(shape=[1100,3],nbytes=1100*3*8),
        tolerance=dict(absolute=1e-9,relative=0.,require_finite=True),
        snapshot_cpu_comparison=comparison,geometry_sha256=audit.digest(geometry),
        inputs={k:audit.identity(p) for k,p in [('geometry',geometry),
            ('cpu_reference',cpu_path),('corrected_profile',profile_path),('orbitals',orbital_path)]},
        algorithm_sources=audit.python_sources(root/'gpu4pyscf'),
        source_hashes={'gpu4pyscf/__init__.py':audit.digest(root/'gpu4pyscf/__init__.py')},
        native_source_hashes={'gpu4pyscf/lib/kernel.cu':audit.digest(root/'gpu4pyscf/lib/kernel.cu')},
        native_file_identities={str((root/f'gpu4pyscf/lib/{name}.so').relative_to(root)):v
                                for name,v in native.items()},native_libraries=native,
        helpers={name:audit.digest(base/name) for name in audit.HELPERS},variants={},samples=[])
    plan = [(size,'warmup',None) for size in [256,128,512,1024]]
    for candidate in [128,512,1024]:
        for pair in range(3):
            order = [256,candidate] if pair%2 == 0 else [candidate,256]
            plan.extend((size,'paired',dict(candidate=candidate,index=pair,order=order)) for size in order)
    for index,(size,phase,pair) in enumerate(plan):
        block = min(size,1100)
        report['samples'].append(dict(block_size=size,phase=phase,pair=pair,
            seconds=1.+size/1024+index/1000,observed_get_blksize=[block],
            observed_block_lengths=[min(block,1100-p) for p in range(0,1100,block)],
            comparison={k:dict(finite=True,max_abs_error=0.,passed=True) for k in ['J','K']},passed=True))
    for size in [128,256,512,1024]:
        rows = [r for r in report['samples'] if r['block_size']==size and r['phase']=='paired']
        times = [r['seconds'] for r in rows]
        controls = [] if size==256 else [next(r['seconds'] for r in report['samples']
            if r['block_size']==256 and r['pair']==row['pair']) for row in rows]
        ratios = [1.]*9 if size==256 else [a/b for a,b in zip(times,controls)]
        report['variants'][str(size)] = dict(effective_block_size=size,
            block_lengths=[min(size,1100-p) for p in range(0,1100,size)],
            blocks_per_call=(1100+size-1)//size,seconds=times,paired_control_seconds=controls,
            normalized_seconds=ratios,median_seconds=statistics.median(times),
            median_normalized_seconds=statistics.median(ratios),passed=True,
            max_abs_error=dict(J=0.,K=0.),
            workspace_bytes=dict(dense_ao_unpack=8*size*4,occupied_transform=8*size*2,sum=8*size*6))
    return root,base,report


def test_complete_synthetic_evidence(evidence):
    root,base,report = evidence
    summary = audit.validate(report,root,base)
    assert summary['status']=='passed' and summary['sample_count']==22
    assert summary['variants']['256']['paired_jk_speed_ratio']==1.
    assert 'no SCF' in summary['limitations']


@pytest.mark.parametrize('corruption',[
    'running','missing_sample','pair_order','pair_index','warmup','nonfinite_time',
    'nonfinite_matrix','large_error','failed_sample','blocks','block_choice','raw_time',
    'normalization','median','workspace','maximum_error','helper','source','native',
    'input','cpu_comparison','bindings'])
def test_rejects_corrupted_evidence(evidence,corruption):
    root,base,original = evidence
    report = copy.deepcopy(original)
    row = report['samples'][4]
    variant = report['variants']['128']
    if corruption=='running': report['status']='running'
    if corruption=='missing_sample': report['samples'].pop()
    if corruption=='pair_order': report['samples'][4:6]=report['samples'][4:6][::-1]
    if corruption=='pair_index': row['pair']['index']=8
    if corruption=='warmup': report['samples'][0]['phase']='paired'
    if corruption=='nonfinite_time': row['seconds']=float('nan')
    if corruption=='nonfinite_matrix': row['comparison']['J']['finite']=False
    if corruption=='large_error': row['comparison']['K']['max_abs_error']=1.01e-9
    if corruption=='failed_sample': row['passed']=False
    if corruption=='blocks': row['observed_block_lengths'][-1]-=1
    if corruption=='block_choice': row['observed_get_blksize']=[1]
    if corruption=='raw_time': variant['seconds'][0]+=1
    if corruption=='normalization': variant['normalized_seconds'][0]+=1
    if corruption=='median': variant['median_seconds']+=1
    if corruption=='workspace': variant['workspace_bytes']['sum']+=1
    if corruption=='maximum_error': variant['max_abs_error']['J']=1e-10
    if corruption=='helper': report['helpers'].pop('gradient_snapshot.py')
    if corruption=='source': (root/'gpu4pyscf/__init__.py').write_text('changed')
    if corruption=='native': (root/'gpu4pyscf/lib/gdft.so').write_text('changed')
    if corruption=='input': (base/'optimization_results/corrected_profile.npz').write_bytes(b'changed')
    if corruption=='cpu_comparison': report['snapshot_cpu_comparison']['energy_error_hartree']=1.
    if corruption=='bindings': report['bindings_restored']=False
    with pytest.raises((AssertionError,ValueError,KeyError)):
        audit.validate(report,root,base)


@pytest.mark.parametrize('complete',[True,False])
def test_writes_outputs_only_after_audit(evidence,complete):
    root,base,report = evidence
    out = base/'resident_workspace_results'
    out.mkdir()
    if not complete:
        report['samples'].pop()
    (out/'exchange_sweep.json').write_text(json.dumps(report))
    if complete:
        audit.main(root,base,out)
        summary = json.loads((out/'exchange_summary.json').read_text())
        assert summary['sample_count']==22
        assert 'not measured total RAM' in (out/'exchange_REPORT.md').read_text()
        assert summary['input_report']==audit.identity(out/'exchange_sweep.json')
    else:
        with pytest.raises(AssertionError):
            audit.main(root,base,out)
        assert not (out/'exchange_summary.json').exists()
        assert not (out/'exchange_REPORT.md').exists()


def test_existing_summary_is_never_overwritten(evidence):
    root,base,report = evidence
    out = base/'resident_workspace_results'
    out.mkdir()
    (out/'exchange_sweep.json').write_text(json.dumps(report))
    existing = out/'exchange_REPORT.md'
    existing.write_text('preserve this artifact')
    with pytest.raises(FileExistsError):
        audit.main(root,base,out)
    assert existing.read_text()=='preserve this artifact'
    assert not (out/'exchange_summary.json').exists()
