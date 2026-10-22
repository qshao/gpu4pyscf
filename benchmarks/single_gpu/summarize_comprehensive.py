"""Recompute scientific gates and measurement provenance from immutable raw reports."""
import json
import math
from pathlib import Path
import statistics

from comprehensive import BASE,ROOT,RESULTS,CASES,SOURCES,settings
from molecular import _file_identity,_metadata,compare_cases


def digest(path):
    return _file_identity(path)['sha256']


def candidate_version(sha):
    versions={digest(BASE/'occupied_pair.py'):'symmetric_average_factored_v3',
              digest(BASE/'runs/2026-10-18-pre-accuracy/benchmarks__single_gpu__occupied_pair.py.source'):'symmetric_average_v2',
              digest(BASE/'runs/2026-10-17-pre-comprehensive/occupied-pair-lower-only.py.source'):'lower_triangle_v1'}
    assert sha in versions,'Unknown occupied-pair source'
    return versions[sha]


def summarize():
    summary=dict(status='partial',cases=[],occupied_pairs=[],large_cases=[],jk_capacity=[],diagnostics={},report_hashes={},report_directory=str(RESULTS),
                 validator_sha256=digest(__file__),
                 note='Single controls establish scaling only. Profiles synchronize inclusive nested scopes. Memory is requested CuPy storage and separate process RSS, not hardware-wide RAM. Original 4-case controls/profiles shared a Python process; process-cold claims are not made. Superseded shared-process memory reports are excluded. Capacity estimates at2228/2958 AOs are not completed calculations.')
    archived=BASE/'runs/2026-10-17-pre-comprehensive'
    drivers={digest(BASE/'comprehensive.py'),digest(archived/'comprehensive.py.source'),
             digest(archived/'comprehensive-no-reuse.py.source')}
    def load(path):
        summary['report_hashes'][path.name]=digest(path)
        return json.loads(path.read_text())
    def check(report,cpu,passed_only=True):
        if passed_only: assert report['status']=='passed'
        for key in ['pyscf','numpy','cpu_libxc']: assert report[key]==cpu[key]
        for name,sha in report['source_hashes'].items(): assert digest(ROOT/name)==sha,name
        for name,sha in report.get('algorithm_source_hashes',{}).items():
            assert sha in {digest(ROOT/name),digest(archived/(name.replace('/','__')+'.source'))},name
        for identity in report.get('native_libraries',{}).values():
            assert _file_identity(identity['path'])==identity,identity['path']
        assert report['geometry_sha256']==digest(cpu['settings']['atom'])
        for name,sha in report.get('helper_hashes',{}).items(): assert digest(BASE/name)==sha,name
        if 'driver_sha256' in report: assert report['driver_sha256'] in drivers
        if 'configuration_sha256' in report: assert report['configuration_sha256'] in drivers
        if 'memory_profiler_sha256' in report: assert report['memory_profiler_sha256']==digest(BASE/'memory_profile.py')
        if 'candidate_sha256' in report: candidate_version(report['candidate_sha256'])
        if 'benchmark_sha256' in report:
            if 'selected_cpu_j' in report:
                assert report['benchmark_sha256']==digest(BASE/'test_large_jk_capacity.py')
            elif 'configuration' in report:
                assert report['benchmark_sha256'] in {digest(BASE/'capacity.py'),digest(archived/'capacity-initial.py.source')}
            else:
                name='test_occupied_pair_acceptance.py' if 'pairs' in report else 'test_occupied_pair_workflow.py'
                assert report['benchmark_sha256']==digest(BASE/name)
    def comparison(cpu,result):
        for key in ['scf_seconds','gradient_seconds']:
            assert math.isfinite(result[key]) and result[key]>0
        actual=compare_cases(cpu,result)
        assert actual['passed'],actual
        return actual
    complete=True
    for case in CASES:
        row=dict(case=case,status='pending',modes={})
        summary['cases'].append(row)
        cpu_path=RESULTS/(case+'_cpu.json')
        if not cpu_path.exists():
            complete=False
            seeded_path=RESULTS/(case+'_seeded_cpu.json')
            if seeded_path.exists():
                seeded=load(seeded_path)
                row['supplementary_cpu_status']=seeded['status']
                if seeded['status']=='passed':
                    assert seeded['result']['converged']
                    row['status']='seeded_cpu_complete'
            continue
        oracle=load(cpu_path)
        if oracle['status']=='running': complete=False; row['status']='cpu_running'; continue
        assert oracle['status']=='passed',oracle.get('error')
        cpu=oracle['result']
        assert cpu['converged'] and cpu['settings']==settings(case)
        check(oracle,cpu)
        row['status']='cpu_passed'
        for mode in ['gpu','profile','memory']:
            path=RESULTS/(case+'_'+mode+'.json')
            if not path.exists(): complete=False; row['modes'][mode]='pending'; continue
            report=load(path)
            if report['status']=='running': complete=False; row['modes'][mode]='running'; continue
            check(report,cpu)
            assert report['cpu_reference_sha256']==digest(cpu_path)
            assert report['comparison']==comparison(cpu,report['result'])
            result=report['result']
            entry=dict(status='passed',comparison=report['comparison'],iterations=result['iterations'],
                       scf_seconds=result['scf_seconds'],gradient_seconds=result['gradient_seconds'])
            if mode=='profile':
                assert report['fine']['bindings_restored']
                entry.update(coarse=report['coarse'],fine=report['fine'])
            if mode=='memory':
                assert report['memory']['initial_pool_live_bytes']==0
                assert 'helper_hashes' in report
                entry['memory']=report['memory']
            row['modes'][mode]=entry
        if all(isinstance(x,dict) and x['status']=='passed' for x in row['modes'].values()): row['status']='passed'
        for variant in ['baseline','packed']:
            path=RESULTS/(case+'_occupied_'+variant+'_memory.json')
            if not path.exists(): continue
            report=load(path)
            if report['status']=='running': continue
            check(report,cpu)
            assert report['reference_sha256']==digest(cpu_path)
            assert report['comparison']==comparison(cpu,report['result'])
            assert report['memory']['initial_pool_live_bytes']==0
            row[variant+'_memory']=dict(
                candidate_version=candidate_version(report['candidate_sha256']),
                whole_cupy_bytes=report['memory']['peak_tracked_live_bytes'],
                jk_new_cupy_bytes=max(x['peak_tracked_live_bytes'] for x in report['jk_memory']),
                process_peak_rss_bytes=report['memory']['host_address_space_peak_rss_bytes'],
                comparison=report['comparison'])
        if all(v+'_memory' in row for v in ['baseline','packed']):
            row['memory_reductions']={key:1-row['packed_memory'][key]/row['baseline_memory'][key]
                                      for key in ['whole_cupy_bytes','jk_new_cupy_bytes','process_peak_rss_bytes']}
        path=RESULTS/(case+'_occupied_pairs.json')
        if not path.exists(): continue
        report=load(path)
        if report['status']=='running': continue
        check(report,cpu)
        assert report['reference_sha256']==digest(cpu_path)
        assert len(report['warmups'])==2 and len(report['pairs'])==3
        for entry in report['warmups']: assert entry['comparison']==comparison(cpu,entry['result'])
        for pair in report['pairs']:
            assert pair['baseline']['result']['iterations']==pair['packed']['result']['iterations']
            for entry in pair.values(): assert entry['comparison']==comparison(cpu,entry['result'])
        medians={v:statistics.median(p[v]['result']['scf_seconds']+p[v]['result']['gradient_seconds'] for p in report['pairs']) for v in ['baseline','packed']}
        assert medians==report['combined_medians']
        ratios=[(p['baseline']['result']['scf_seconds']+p['baseline']['result']['gradient_seconds'])/(p['packed']['result']['scf_seconds']+p['packed']['result']['gradient_seconds']) for p in report['pairs']]
        speedup=medians['baseline']/medians['packed']
        assert speedup==report['speedup'] and ratios==report['paired_ratios']
        summary['occupied_pairs'].append(dict(case=case,combined_medians=medians,speedup=speedup,paired_ratios=ratios))
    # These supplementary allocation runs do not complete the original matrix.
    # A CPU pass must never erase a stricter, failed preservation experiment.
    from seeded_reference import compare_execution_cases
    for path in sorted(RESULTS.glob('*_capacity_*_*.json')):
        report=load(path)
        case=report['case'] if 'case' in report else path.name.split('_capacity_')[0]
        entry=dict(report=path.name,case=case,status=report['status'])
        summary['large_cases'].append(entry)
        if report['status']=='running': continue
        result=report.get('result')
        if result is None:
            entry['error']=report.get('error')
            continue
        check(report,result,passed_only=False)
        entry.update(iterations=result['iterations'],scf_seconds=result['scf_seconds'],
                     gradient_seconds=result['gradient_seconds'],error=report.get('error'),
                     cpu_comparison=None)
        for cpu_path in [RESULTS/(case+'_cpu.json'),RESULTS/(case+'_seeded_cpu.json')]:
            if not cpu_path.exists(): continue
            cpu_report=load(cpu_path)
            if cpu_report['status']!='passed': continue
            cpu=cpu_report['result']
            # Seeded references use a different CPU thread count, but retain
            # every scientific setting and the original energy/force gates.
            for key in ['pyscf','numpy','cpu_libxc']: assert cpu_report[key]==report[key]
            assert cpu_report['geometry_sha256']==report['geometry_sha256']
            if 'seed' in cpu_report:
                assert _file_identity(cpu_report['seed']['path'])==cpu_report['seed']
                parent=RESULTS/(case+'_capacity_disk_packed.json')
                assert cpu_report['seed_report_sha256']==digest(parent)
                assert cpu_report['benchmark_sha256']==digest(BASE/'seeded_reference.py')
                assert cpu_report['configuration_sha256'] in drivers
            entry.update(cpu_report=cpu_path.name,cpu_comparison=compare_execution_cases(cpu,result))
            break
        if 'memory' in report:
            entry['whole_cupy_bytes']=report['memory']['peak_tracked_live_bytes']
            entry['process_peak_rss_bytes']=report['memory']['host_address_space_peak_rss_bytes']
        if report.get('jk_memory'):
            entry['jk_new_cupy_bytes']=max(m['peak_tracked_live_bytes'] for m in report['jk_memory'])
        if report.get('variant')=='packed':
            entry['candidate_version']=candidate_version(report['candidate_sha256'])
            baseline_path=RESULTS/(case+'_capacity_'+report.get('gpu_reference_storage',report['configuration']['storage'])+'_baseline.json')
            assert report['gpu_reference_sha256']==digest(baseline_path),'GPU reference report changed'
            baseline_report=load(baseline_path)
            assert baseline_report['status']=='captured_original_gpu'
            for key in ['geometry_sha256','native_libraries','algorithm_source_hashes',
                        'memory_profiler_sha256','pyscf','numpy','cpu_libxc','cupy']:
                assert baseline_report[key]==report[key],key
            for key in ['scientific','buffer_bytes','cache_bytes']:
                assert baseline_report['configuration'][key]==report['configuration'][key],key
            baseline=baseline_report['result']
            assert baseline['settings']==result['settings'] and baseline['method']==result['method']
            import numpy as np
            error=dict(energy_hartree=abs(result['energy_hartree']-baseline['energy_hartree']),
                       gradient_max=float(np.max(np.abs(np.asarray(result['gradient_hartree_per_bohr'])-np.asarray(baseline['gradient_hartree_per_bohr'])))))
            entry['gpu_comparison']=dict(error,passed=(error['energy_hartree']<=1e-10 and error['gradient_max']<=1e-9 and result['iterations']==baseline['iterations']))
    for path in sorted(RESULTS.glob('*_jk_capacity.json')):
        report=load(path)
        entry=dict(report=path.name,case=report['case'],status=report['status'])
        summary['jk_capacity'].append(entry)
        if report['status']!='passed': continue
        check(report,dict(_metadata('cpu'),settings=report['settings']))
        assert report['oracle_sha256']==digest(BASE/'selected_jk.py')
        import numpy as np
        errors={}
        for label in ['j','k']:
            cpu=np.asarray(report['selected_cpu_'+label]); gpu=np.asarray(report['selected_gpu_'+label])
            assert cpu.shape==gpu.shape==(len(report['selected']),len(report['selected']))
            errors[label+'_max']=float(np.max(np.abs(cpu-gpu)))
        assert errors==report['errors'] and all(x<=1e-8 for x in errors.values())
        assert report['scratch_clean'] and report['oracle_read_bytes']==report['cderi_bytes']
        if report['mode']=='disk':
            stats=report['gpu_jk_stats']
            assert stats['d2h_bytes']==report['cderi_bytes'] and stats['cache_device_bytes']==0
            assert stats['peak_host_buffer_bytes']<=report['buffer_bytes']
            assert stats['peak_device_buffer_bytes']<=report['buffer_bytes']
            assert report['cderi_file_removed']
        entry.update(nao=report['nao'],mode=report['mode'],cderi_bytes=report['cderi_bytes'],
                     whole_cupy_bytes=report['memory']['peak_tracked_live_bytes'],
                     process_peak_rss_bytes=report['memory']['host_address_space_peak_rss_bytes'],
                     errors=errors,scope='Single J/K; no converged SCF/force qualification')
    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    if parent_path.exists():
        parent=load(parent_path)
        for name,script in [('taxol_original_gradient_profile.json','test_taxol_original_profile.py'),
                            ('taxol_fixed_orbital_xc_cpu.json','test_taxol_xc_diagnostic.py'),
                            ('taxol_fixed_orbital_xc_gpu.json','test_taxol_xc_diagnostic.py')]:
            path=RESULTS/name
            if not path.exists(): continue
            report=load(path)
            if report['status'] in ['running','failed']:
                summary['diagnostics'][name]=dict(status=report['status'])
                continue
            assert report['parent_sha256']==digest(parent_path)
            assert report['benchmark_sha256']==digest(BASE/script)
            for source,sha in report['algorithm_source_hashes'].items():
                assert sha in {digest(ROOT/source),digest(archived/(source.replace('/','__')+'.source'))},source
            for identity in report.get('native_libraries',{}).values(): assert _file_identity(identity['path'])==identity
            for helper,sha in report.get('helper_hashes',{}).items(): assert sha==digest(BASE/helper)
            if 'orbitals' in report: assert report['orbitals']==parent['orbitals'] and _file_identity(report['orbitals']['path'])==report['orbitals']
            if 'coarse' in report:
                oracle_path=RESULTS/'taxol_fixed_orbital_full_diagnostic.json'
                assert report['oracle_sha256']==digest(oracle_path)
                oracle=load(oracle_path)
                import numpy as np
                error=float(np.max(np.abs(np.asarray(oracle['gradient'])-np.asarray(report['gradient']))))
                assert error==report['replay_error'] and error<=1e-9 and report['fine']['bindings_restored']
                summary['diagnostics'][name]=dict(status=report['status'],coarse=report['coarse'],fine=report['fine'],replay_error=error)
            else:
                assert report['configuration_sha256'] in drivers
                summary['diagnostics'][name]=dict(status=report['status'],seconds=report['seconds'])
        xc_paths=[RESULTS/('taxol_fixed_orbital_xc_'+backend+'.json') for backend in ['cpu','gpu']]
        if all(path.exists() for path in xc_paths):
            xc=[load(path) for path in xc_paths]
            if all(report['status']=='captured' for report in xc):
                assert xc[0]['orbitals']==xc[1]['orbitals'] and xc[0]['settings']==xc[1]['settings']
                for key in ['pyscf','numpy','cpu_libxc']: assert xc[0][key]==xc[1][key]
                import numpy as np
                error=float(np.max(np.abs(np.asarray(xc[0]['force'])-np.asarray(xc[1]['force']))))
                assert math.isfinite(error)
                summary['diagnostics']['matched_orbital_xc_max_error']=error
    summary['status']='passed' if complete else 'partial'
    return summary


if __name__=='__main__':
    result=summarize()
    (BASE/'comprehensive_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],[(r['case'],r['status']) for r in result['cases']])
