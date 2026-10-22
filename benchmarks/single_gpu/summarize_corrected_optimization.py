"""Independently recheck raw observations, numerical gates and source identities."""
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

import numpy as np

from corrected_optimization import BASE,ROOT,OUT,oracle,digest,save
from seeded_reference import compare_execution_cases

ARCHIVE=BASE/'runs/2026-10-18-pre-next-optimization'
REJECTED=BASE/'runs/2026-10-18-rejected-short-exchange'
INVALID=BASE/'runs/2026-10-18-invalid-acceptance'


def source_matches(source,sha):
    paths=[ROOT/source,ARCHIVE/(source.replace('/','__')+'.source'),
           REJECTED/(source.replace('/','__')+'.source'),
           INVALID/(source.replace('/','__')+'.source')]
    assert any(path.exists() and digest(path)==sha for path in paths),(source,sha)


def load(name,script,passed=True):
    path=OUT/name
    report=json.loads(path.read_text())
    assert report['status']==('passed' if passed else 'failed'),name
    source_matches('benchmarks/single_gpu/'+script,report['benchmark_sha256'])
    assert report['shared_helper_sha256']==digest(BASE/'corrected_optimization.py')
    assert report['configuration_sha256']==digest(BASE/'comprehensive.py')
    assert report['comparison_helper_sha256']==digest(BASE/'seeded_reference.py')
    for source,sha in report['algorithm_source_hashes'].items(): source_matches(source,sha)
    for source,sha in report['source_hashes'].items(): source_matches(source,sha)
    for identity in report['native_libraries'].values():
        assert digest(identity['path'])==identity['sha256']
        assert Path(identity['path']).stat().st_size==identity['size_bytes']
    if report.get('xc_source_sha256'): source_matches('gpu4pyscf/grad/rks.py',report['xc_source_sha256'])
    return report


def maximum_difference(a,b):
    a,b=map(np.asarray,(a,b))
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a-b)))


def paired(report,first,second):
    pairs=report['pairs']; assert len(pairs)==3
    for pair in pairs:
        for variant in [first,second]:
            assert math.isfinite(pair[variant]['seconds']) and pair[variant]['seconds']>0
    medians={v:statistics.median(p[v]['seconds'] for p in pairs) for v in [first,second]}
    ratios=[p[first]['seconds']/p[second]['seconds'] for p in pairs]
    assert medians==report['medians'] and ratios==report['paired_ratios']
    speedup=medians[first]/medians[second]; assert speedup==report['speedup']
    return dict(medians=medians,speedup=speedup,paired_ratios=ratios)


def main():
    profile=load('corrected_profile.json','test_corrected_profile.py')
    cpu_path,cpu=oracle()
    assert profile['cpu_reference_sha256']==digest(cpu_path)
    assert compare_execution_cases(cpu['result'],profile['result'])['passed']
    assert digest(profile['orbitals']['path'])==profile['orbitals']['sha256']
    profile_sha=digest(OUT/'corrected_profile.json')
    short=[]
    for layout in ['tight','padded']:
        for size in [32,64,128]:
            report=load(f'short_exchange_{size}_{layout}.json','test_short_exchange.py')
            assert report['candidate_sha256']==digest(BASE/'exchange_layout.py')
            assert math.isfinite(report['maximum_difference']) and report['maximum_difference']<=1e-9
            for samples in report['samples'].values():
                assert len(samples)==7 and all(math.isfinite(t) and t>0 for t in samples)
            medians={v:statistics.median(report['samples'][v]) for v in ['baseline','strided']}
            assert medians==report['medians']
            ratio=medians['baseline']/medians['strided']
            assert ratio==report['component_ratio']
            short.append(dict(rows=size,layout=layout,speedup=ratio))
    streamed=load('streamed_exchange_pairs.json','test_streamed_exchange_pairs.py')
    assert streamed['candidate_sha256']==digest(BASE/'short_exchange.py')
    assert streamed['transform_sha256']==digest(BASE/'exchange_layout.py')
    assert streamed['oracle_sha256']==digest(BASE/'selected_jk.py')
    assert streamed['geometry_sha256']==digest(streamed['settings']['atom'])
    stream_speed=paired(streamed,'baseline','strided')
    rows=streamed['warmups']+[v for p in streamed['pairs'] for v in p.values()]
    for row in rows:
        assert all(math.isfinite(e) and e<=1e-9 for e in row['errors'])
        assert row['h2d_bytes']==streamed['cderi_bytes'] and row['h2d_calls']==235
    assert all(math.isfinite(e) and e<=1e-8 for e in streamed['cpu_contraction_errors'])
    for key in ['peak_host_buffer_bytes','peak_device_buffer_bytes']:
        assert streamed['stream_stats'][key]<=streamed['stream_buffer_bytes']
        assert streamed['build_stats'][key]<=streamed['build_buffer_bytes']
    assert streamed['stream_stats']['cache_device_bytes']==0 and streamed['scratch_clean']

    baseline=load('gradient_baseline_memory.json','test_corrected_packing.py')
    rejected=load('gradient_packed_memory.json','test_corrected_packing.py',False)
    assert baseline['profile_sha256']==rejected['profile_sha256']==profile_sha
    assert baseline['memory_profiler_sha256']==digest(BASE/'memory_profile.py')
    assert maximum_difference(baseline['comparison']['gradient'],profile['result']['gradient_hartree_per_bohr'])<=1e-9
    assert '2.4597284209448844e-09' in rejected['error']
    xc={v:load(f'xc_{v}_memory.json','test_xc_workspace.py') for v in ['baseline','capped','full_capped']}
    for report in xc.values():
        assert report['profile_sha256']==profile_sha
        assert report['candidate_sha256']==digest(BASE/'xc_workspace.py')
        assert report['memory_profiler_sha256']==digest(BASE/'memory_profile.py')
    error=maximum_difference(xc['baseline']['gradient'],xc['capped']['gradient'])
    assert error<=1e-9 and error==xc['capped']['error_to_baseline']
    assert xc['capped']['baseline_sha256']==digest(OUT/'xc_baseline_memory.json')
    for variant in ['capped','full_capped']:
        assert max(b['estimated_bytes'] for b in xc[variant]['batches'])<=512*1024**2
    full=xc['full_capped']
    force_error=maximum_difference(full['gradient'],profile['result']['gradient_hartree_per_bohr'])
    assert force_error<=1e-9
    full_result=dict(profile['result'],gradient_hartree_per_bohr=full['gradient'])
    assert compare_execution_cases(cpu['result'],full_result)['passed']
    xc_pairs=load('xc_pairs.json','test_xc_workspace.py')
    assert xc_pairs['candidate_sha256']==digest(BASE/'xc_workspace.py')
    assert xc_pairs['baseline_sha256']==digest(OUT/'xc_baseline_memory.json')
    assert xc_pairs['capped_sha256']==digest(OUT/'xc_capped_memory.json')
    xc_speed=paired(xc_pairs,'baseline','capped')
    for row in xc_pairs['warmups']+[v for p in xc_pairs['pairs'] for v in p.values()]:
        assert math.isfinite(row['error_to_baseline']) and row['error_to_baseline']<=1e-9

    invalid_reports=[]
    for name in ['production_resident_acceptance.json','production_resident_xc_acceptance.json']:
        invalid=load(name,'test_corrected_acceptance.py',False)
        assert 'actual_factory_settings' not in invalid
        invalid_reports.append(dict(name=name,status='excluded: factory bypassed NWChem pruning',
                                    sha256=digest(OUT/name)))
    acceptance={}
    for storage in ['resident','disk']:
        report=load(f'production_{storage}_qualified_acceptance.json','test_corrected_acceptance.py')
        assert report['storage']==storage
        actual=report['actual_factory_settings']
        assert actual and all(value==profile['settings'][key] for key,value in actual.items())
        assert set(actual)=={'grid_pruning','grid_level','max_cycle','small_rho_cutoff','conv_tol','conv_tol_grad'}
        for source,sha in report['algorithm_source_hashes'].items():
            assert digest(ROOT/source)==sha,source
        assert report['cpu_reference_sha256']==digest(cpu_path)
        assert report['xc_source_sha256']==digest(ROOT/'gpu4pyscf/grad/rks.py')
        assert report['disk_source_sha256']==digest(ROOT/'gpu4pyscf/df/disk.py')
        assert report['algorithm_source_hashes']['gpu4pyscf/df/df_jk.py']==digest(ROOT/'gpu4pyscf/df/df_jk.py')
        assert report['seed_parent_sha256']==digest(BASE/'accuracy_results/taxol_resident_acceptance.json')
        assert digest(report['seed']['path'])==report['seed']['sha256']
        comparison=compare_execution_cases(cpu['result'],report['result']); assert comparison['passed']
        assert report['scratch_clean']
        assert report['integrals'] and all(s['disk']==(storage=='disk') for s in report['integrals'])
        if storage=='disk':
            for integral in report['integrals']:
                stats=integral['stats']
                assert stats['peak_host_buffer_bytes']<=report['buffer_bytes']
                assert stats['peak_device_buffer_bytes']<=report['buffer_bytes']
                assert stats['d2h_bytes']==integral['nbytes'] and stats['cache_device_bytes']==0
        acceptance[storage]=comparison
    before=baseline['memory']['peak_tracked_live_bytes']
    after=full['memory']['peak_tracked_live_bytes']
    assert after<before
    tests={}
    for name in ['qualified_regressions.xml','configuration.xml']:
        path=OUT/name
        cases=list(ET.parse(path).getroot().iter('testcase'))
        assert cases and all(c.find('failure') is None and c.find('error') is None
                             and c.find('skipped') is None for c in cases),name
        tests[name]=dict(passed=len(cases),sha256=digest(path))
    red=OUT/'production_red.xml'
    assert len(list(ET.parse(red).getroot().iter('failure')))==5
    protocol_red=OUT/'protocol_red.xml'
    assert len(list(ET.parse(protocol_red).getroot().iter('failure')))==1
    summary=dict(status='passed',validator_sha256=digest(__file__),
                 short_exchange=short,streamed_jk=stream_speed,xc=xc_speed,
                 gradient_memory=dict(before_bytes=before,after_bytes=after,reduction_fraction=1-after/before),
                 xc_memory=dict(before_bytes=xc['baseline']['memory']['peak_tracked_live_bytes'],
                                after_bytes=xc['capped']['memory']['peak_tracked_live_bytes']),
                 force_error_to_corrected_gpu=force_error,xc_error_to_baseline=error,
                 production_cpu_acceptance=acceptance,packing='rejected: strict accuracy gate failed',
                 short_exchange_promotion='qualified by corrected full resident/disk CPU acceptance',
                 excluded_protocol_reports=invalid_reports,
                 regression_tests=tests,tdd_red_sha256=digest(red),protocol_red_sha256=digest(protocol_red),
                 note='Ratios apply to separate measured components; do not multiply them or claim a whole-workflow speedup. Allocation peaks exclude preexisting/native storage; RSS is a separate metric. Streamed J/K keeps identical H2D volume. Packing remains benchmark-only.',
                 input_sha256={p.name:digest(p) for p in OUT.glob('*.json') if p.name!='summary.json'})
    save(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
