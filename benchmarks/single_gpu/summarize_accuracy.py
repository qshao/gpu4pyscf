"""Independently recompute Taxol accuracy gates and verify source identities."""
import json

import numpy as np

from comprehensive import BASE, ROOT, RESULTS, SOURCES, settings
from molecular import _file_identity
from seeded_reference import compare_execution_cases

OUT=BASE/'accuracy_results'
ARCHIVE=BASE/'runs/2026-10-18-pre-accuracy'


def digest(path):
    return _file_identity(path)['sha256']


def summarize():
    hashes={}
    def load(path):
        hashes[str(path.relative_to(BASE))]=digest(path)
        return json.loads(path.read_text())
    def sources(report,historical=False):
        for name,sha in report['source_hashes'].items(): assert sha==digest(ROOT/name),name
        for name,sha in report.get('algorithm_source_hashes',{}).items():
            allowed={digest(ROOT/name)}
            archive=ARCHIVE/(name.replace('/','__')+'.source')
            if historical and archive.exists(): allowed.add(digest(archive))
            assert sha in allowed,name
        for identity in report.get('native_libraries',{}).values():
            assert _file_identity(identity['path'])==identity
    cpu_path=RESULTS/'taxol_svp_b3lyp_seeded_cpu.json'
    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    cpu=load(cpu_path); parent=load(parent_path)
    assert cpu['status']=='passed' and cpu['result']['converged']
    assert parent['result']['converged']
    assert _file_identity(parent['orbitals']['path'])==parent['orbitals']
    assert cpu['seed_report_sha256']==digest(parent_path)
    assert cpu['geometry_sha256']==digest(settings('taxol_svp_b3lyp')['atom'])
    assert cpu['benchmark_sha256']==digest(BASE/'seeded_reference.py')
    assert cpu['configuration_sha256']==digest(BASE/'comprehensive.py')
    sources(cpu)
    fixed_path=OUT/'taxol_cpu_fixed_components.json'
    fixed=load(fixed_path)
    assert fixed['status']=='captured' and fixed['parent_sha256']==digest(parent_path)
    assert fixed['benchmark_sha256']==digest(BASE/'test_taxol_accuracy.py')
    assert fixed['xc_sha256']==digest(RESULTS/'taxol_fixed_orbital_xc_cpu.json')
    original=load(RESULTS/'taxol_fixed_orbital_jk_diagnostic.json')
    assert original['status']=='captured' and original['parent_sha256']==digest(parent_path)
    assert original['benchmark_sha256'] in {
        digest(BASE/'test_large_gradient_diagnostic.py'),
        digest(BASE/'runs/2026-10-17-pre-comprehensive/large-gradient-jk-diagnostic.py.source')}
    sources(original,historical=True)
    hypothesis=load(OUT/'taxol_factored_metric.json')
    sources(hypothesis,historical=True)
    assert hypothesis['status']=='captured' and hypothesis['cpu_sha256']==digest(fixed_path)
    assert hypothesis['parent_sha256']==digest(parent_path)
    assert hypothesis['benchmark_sha256']==digest(BASE/'test_taxol_metric.py')
    assert hypothesis['candidate_sha256']==digest(BASE/'factored_metric.py')
    def error(a,b):
        left,right=np.asarray(a),np.asarray(b)
        assert left.shape==right.shape and np.isfinite(left).all() and np.isfinite(right).all()
        return float(np.max(np.abs(left-right)))
    before=error(fixed['jk'],original['first'])
    after=error(fixed['jk'],hypothesis['jk'])
    assert before==fixed['jk_error'] and after==hypothesis['cpu_error']
    summary=dict(status='partial',scope='Taxol force accuracy correction only; original eight-case scaling matrix remains partial',
        baseline_jk_error=before,factored_jk_error=after,
        cpu_state_effect=error(fixed['full'],cpu['result']['gradient_hartree_per_bohr']),
        metric=hypothesis['metric'],workflows={},report_hashes=hashes,
        production_sources={name:digest(ROOT/name) for name in SOURCES},
        validator_sha256=digest(__file__))
    complete=True
    for storage in ['resident','disk']:
        path=OUT/('taxol_'+storage+'_acceptance.json')
        if not path.exists(): complete=False; continue
        report=load(path)
        entry=dict(status=report['status']); summary['workflows'][storage]=entry
        if report['status']!='passed': complete=False; continue
        sources(report)
        assert set(report['algorithm_source_hashes'])==set(SOURCES+['gpu4pyscf/df/disk.py'])
        assert report['cpu_reference_sha256']==digest(cpu_path)
        assert report['seed_report_sha256']==digest(parent_path) and report['seed']==parent['orbitals']
        assert report['benchmark_sha256']==digest(BASE/'test_taxol_accuracy_acceptance.py')
        assert report['configuration_sha256']==digest(BASE/'comprehensive.py')
        assert report['comparison_sha256']==digest(BASE/'seeded_reference.py')
        result=report['result']
        assert result['settings']==settings('taxol_svp_b3lyp')
        comparison=compare_execution_cases(cpu['result'],result)
        assert comparison==report['comparison'] and comparison['passed'],comparison
        assert report['scratch_clean'] and report['integrals']
        for integral in report['integrals']:
            assert integral['disk']==(storage=='disk')
            if storage=='disk':
                stats=integral['stats']; budget=report['buffer_bytes']
                assert stats['peak_host_buffer_bytes']<=budget and stats['peak_device_buffer_bytes']<=budget
                assert stats['d2h_bytes']==integral['nbytes'] and stats['cache_device_bytes']==0
        entry.update(comparison=comparison,iterations=result['iterations'],
                     scf_seconds=result['scf_seconds'],gradient_seconds=result['gradient_seconds'])
    if complete:
        summary['status']='passed'
        resident=load(OUT/'taxol_resident_acceptance.json')['result']
        disk=load(OUT/'taxol_disk_acceptance.json')['result']
        summary['storage_comparison']=dict(
            energy_error_hartree=abs(resident['energy_hartree']-disk['energy_hartree']),
            gradient_max_error=error(resident['gradient_hartree_per_bohr'],disk['gradient_hartree_per_bohr']))
    elif any(entry['status']=='failed' for entry in summary['workflows'].values()):
        summary['status']='failed'
    return summary


if __name__=='__main__':
    result=summarize()
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],result['workflows'])
