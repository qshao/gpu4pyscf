"""Independent original/current full-workflow and memory comparison audit."""
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import tarfile
import xml.etree.ElementTree as ET

import numpy as np

from original_comparison import (BASE,ROOT,OUT,REVISION,CASES,manifest,digest,save,
    python_sources,case_settings,cpu_reference)
from seeded_reference import compare_execution_cases


def audit_frozen():
    frozen=manifest()
    assert frozen['original_revision']==REVISION
    for package in frozen['packages'].values():
        assert python_sources(Path(package['path']))==package['sources']
    for relative,sha in frozen['packages']['current']['sources'].items():
        assert digest(ROOT/'gpu4pyscf'/relative)==sha,relative
    archive=subprocess.check_output(['git','-C',str(ROOT),'archive',REVISION,'gpu4pyscf'])
    original={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            if member.isfile() and member.name.endswith('.py'):
                key=str(Path(member.name).relative_to('gpu4pyscf'))
                original[key]=hashlib.sha256(stream.extractfile(member).read()).hexdigest()
    assert original==frozen['packages']['original']['sources']
    for identity in frozen['native_libraries'].values():
        assert digest(identity['path'])==identity['sha256']
        assert Path(identity['path']).stat().st_size==identity['bytes']
    for size in CASES:
        path,_=cpu_reference(size)
        assert str(path)==frozen['cpu_references'][size]['path']
        assert digest(path)==frozen['cpu_references'][size]['sha256']
        assert digest(case_settings(size)['atom'])==frozen['geometry_sha256'][size]
    return frozen


def load(size,version,mode,repeat,frozen):
    path=OUT/'results'/f'{size}_{version}_{mode}_{repeat}.json'
    report=json.loads(path.read_text())
    assert report['status'] in ['passed','failed_accuracy'],(path,report['status'])
    assert report['size']==size and report['version']==version and report['mode']==mode and report['repeat']==repeat
    assert report['manifest_sha256']==digest(OUT/'manifest.json')
    assert report['benchmark_sha256']==frozen['runner_sha256']
    assert report['configuration_sha256']==frozen['configuration_sha256']
    assert report['cpu_reference_sha256']==frozen['cpu_references'][size]['sha256']
    assert report['geometry_sha256']==frozen['geometry_sha256'][size]
    assert report['package']==frozen['packages']['original' if version=='original' else 'current']['path']
    native={item['path']:item['sha256'] for item in frozen['native_libraries'].values()}
    for item in report['native_libraries'].values():
        assert native[item['path']]==item['sha256']
    config=case_settings(size)
    assert report['settings']==report['result']['settings']==config
    actual=report['actual_factory_settings']
    assert set(actual)=={'grid_pruning','grid_level','max_cycle','conv_tol','conv_tol_grad','small_rho_cutoff'}
    assert all(config[key]==value for key,value in actual.items())
    _,cpu=cpu_reference(size)
    for key in ['pyscf','numpy','cpu_libxc']: assert report[key]==cpu[key]
    comparison=compare_execution_cases(cpu['result'],report['result'])
    assert comparison==report['comparison']
    assert (report['status']=='passed')==comparison['passed']
    if version!='original': assert comparison['passed'],(size,version,comparison)
    assert report['result']['converged'] and report['scratch_clean']
    gradient=np.asarray(report['result']['gradient_hartree_per_bohr'])
    assert gradient.shape==(report['natm'],3) and np.isfinite(gradient).all()
    for key in ['workflow_seconds','setup_seconds','scf_seconds','gradient_seconds']:
        assert math.isfinite(report[key]) and report[key]>0
    subtotal=sum(report[key] for key in ['setup_seconds','scf_seconds','gradient_seconds'])
    assert 0<=report['workflow_seconds']-subtotal<.1
    assert report['integrals'] and all(store['is_disk']==(version=='disk') for store in report['integrals'])
    if version=='disk':
        for store in report['integrals']:
            stats=store['stats']
            assert stats['peak_host_buffer_bytes']<=4*1024**3
            assert stats['peak_device_buffer_bytes']<=4*1024**3
            assert stats['cache_device_bytes']==0 and stats['d2h_bytes']==store['nbytes']
    else: assert all(store['gpu_array'] for store in report['integrals'])
    if mode=='memory':
        memory=report['memory']
        assert memory['initial_pool_live_bytes']==0
        assert memory['allocation_observations']>0 and memory['peak_tracked_live_bytes']>0
        assert memory['host_address_space_peak_rss_bytes']>0
    else: assert report['memory']=={}
    return report


def main():
    frozen=audit_frozen()
    summary=dict(status='passed',original_revision=REVISION,validator_sha256=digest(__file__),
                 manifest_sha256=digest(OUT/'manifest.json'),cases=[])
    environmental=[]
    for size in CASES:
        versions=['original','current']+(['disk'] if size=='large' else [])
        row=dict(size=size,case=CASES[size],
                 molecule={'small':'Vitamin C','medium':'Tamoxifen','large':'Taxol'}[size],versions={})
        seeds=set()
        for version in versions:
            timed=[load(size,version,'time',repeat,frozen) for repeat in range(3)]
            measured=load(size,version,'memory',0,frozen)
            all_reports=timed+[measured]
            seeds.update(r['initial_density_sha256'] for r in all_reports)
            for r in all_reports:
                environmental.append(tuple(r[k] for k in ['device_name','compute_capability','cupy',
                                                          'cuda_driver_version','cuda_runtime_version','pyscf','numpy','cpu_libxc']))
            row.update(natm=measured['natm'],nao=measured['nao'])
            medians={key:statistics.median(r[key] for r in timed)
                     for key in ['workflow_seconds','setup_seconds','scf_seconds','gradient_seconds']}
            row['versions'][version]=dict(medians=medians,samples=[r['workflow_seconds'] for r in timed],
                scf_iterations=[len(r['iterations']) for r in timed],
                peak_tracked_cupy_bytes=measured['memory']['peak_tracked_live_bytes'],
                host_address_space_peak_rss_bytes=measured['memory']['host_address_space_peak_rss_bytes'],
                cpu_accuracy_passed=all(r['comparison']['passed'] for r in all_reports),
                maximum_cpu_energy_error=max(r['comparison']['energy_error_hartree'] for r in all_reports),
                maximum_cpu_force_error=max(r['comparison']['gradient_max_error'] for r in all_reports))
        assert len(seeds)==1,(size,seeds)
        row['initial_density_sha256']=seeds.pop()
        baseline=row['versions']['original']
        for version in versions[1:]:
            current=row['versions'][version]
            current['speedup_vs_original']=baseline['medians']['workflow_seconds']/current['medians']['workflow_seconds']
            current['elapsed_time_reduction_fraction']=1-1/current['speedup_vs_original']
            current['cupy_reduction_fraction']=1-current['peak_tracked_cupy_bytes']/baseline['peak_tracked_cupy_bytes']
            current['host_rss_reduction_fraction']=1-current['host_address_space_peak_rss_bytes']/baseline['host_address_space_peak_rss_bytes']
            current['accuracy_qualified_comparison']=baseline['cpu_accuracy_passed'] and current['cpu_accuracy_passed']
        original=[load(size,'original','time',i,frozen) for i in range(3)]
        current=[load(size,'current','time',i,frozen) for i in range(3)]
        assert original[0]['timestamp_utc']<current[0]['timestamp_utc']
        assert current[1]['timestamp_utc']<original[1]['timestamp_utc']
        assert original[2]['timestamp_utc']<current[2]['timestamp_utc']
        row['paired_current_speed_ratios']=[a['workflow_seconds']/b['workflow_seconds'] for a,b in zip(original,current)]
        summary['cases'].append(row)
    assert len(set(environmental))==1
    summary['audit_status']='passed'
    summary['current_accuracy_passed']=all(
        data['cpu_accuracy_passed'] for row in summary['cases']
        for version,data in row['versions'].items() if version!='original')
    summary['original_accuracy_passed']=all(
        row['versions']['original']['cpu_accuracy_passed'] for row in summary['cases'])
    if not summary['original_accuracy_passed']:
        summary['status']='completed_with_original_accuracy_failures'
    summary['environment']=dict(zip(['device','compute_capability','cupy','cuda_driver','cuda_runtime','pyscf','numpy','cpu_libxc'],environmental[0]))
    files=sorted((OUT/'results').glob('*.json'));assert len(files)==28,len(files)
    tests=OUT/'harness_tests_final.xml'
    testcases=list(ET.parse(tests).getroot().iter('testcase'))
    assert len(testcases)==11 and all(c.find('failure') is None and c.find('error') is None for c in testcases)
    summary.update(input_sha256={p.name:digest(p) for p in files},
        harness_checks=dict(passed=11,xml_sha256=digest(tests),
            sources={name:digest(BASE/name) for name in ['test_original_comparison_config.py','test_original_comparison_summary.py']}),
        note='Three fresh-process cold atomic-guess timing samples; medians cover setup plus fresh DF/SCF/full gradient. Separate fresh-process allocation observations are not timing samples. Original/current share native GB10-compatible libraries. CuPy peaks exclude native/preexisting storage and cannot be added to host RSS on unified memory. OS/CuPy caches are uncontrolled. An original accuracy failure is explicitly unqualified, not an equivalent-accuracy speed claim.')
    save(OUT/'summary.json',summary)
    render_report(summary)
    print(json.dumps(summary,indent=2))


def render_report(summary):
    lines=['# Original versus current GPU4PySCF benchmark','',
        f"Original GPU source: `{REVISION}`. Current code is the immutable campaign snapshot.",
        'GB10, shared SM121-compatible native libraries, FP64, four CPU threads.',
        'CuPy contraction fallback: this environment lacks the CuPy cuTENSOR interface; results do not establish speedups for a cuTENSOR-enabled installation.',
        'B3LYP/def2-SVP, universal-JKfit, level3 NWChem grids, original convergence thresholds, full grid/auxiliary response.',
        'Fresh atomic initial guesses and fresh integral construction in every process. Three timing repetitions; memory measured separately.','',
        'Timings exclude imports, provenance collection and final cleanup.','',
        '| System | Atoms | AOs | Original complete workflow, s | Current, s | Speed ratio | Original/current cycles |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in summary['cases']:
        a,b=[row['versions'][v] for v in ['original','current']]
        qualification='' if b['accuracy_qualified_comparison'] else ' (accuracy unqualified)'
        lines.append(f"| {row['size']}: {row['molecule']} | {row['natm']} | {row['nao']} | {a['medians']['workflow_seconds']:.3f} | {b['medians']['workflow_seconds']:.3f} | {b['speedup_vs_original']:.3f}x{qualification} | {a['scf_iterations']}/{b['scf_iterations']} |")
    lines+=['','| System | Original SCF, s | Current SCF, s | Original gradient, s | Current gradient, s |',
            '| --- | ---: | ---: | ---: | ---: |']
    for row in summary['cases']:
        a,b=[row['versions'][v]['medians'] for v in ['original','current']]
        lines.append(f"| {row['size']} | {a['scf_seconds']:.3f} | {b['scf_seconds']:.3f} | {a['gradient_seconds']:.3f} | {b['gradient_seconds']:.3f} |")
    lines+=['','| System | Original CuPy peak, GB | Current CuPy peak, GB | Reduction | Original RSS, GB | Current RSS, GB |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['cases']:
        a,b=[row['versions'][v] for v in ['original','current']]
        lines.append(f"| {row['size']} | {a['peak_tracked_cupy_bytes']/1e9:.3f} | {b['peak_tracked_cupy_bytes']/1e9:.3f} | {100*b['cupy_reduction_fraction']:.1f}% | {a['host_address_space_peak_rss_bytes']/1e9:.3f} | {b['host_address_space_peak_rss_bytes']/1e9:.3f} |")
    large=summary['cases'][-1];disk=large['versions']['disk']
    lines+=['',f"Large current disk mode: {disk['medians']['workflow_seconds']:.3f}s, {disk['peak_tracked_cupy_bytes']/1e9:.3f}GB tracked CuPy, {disk['host_address_space_peak_rss_bytes']/1e9:.3f}GB host RSS. Staging4GiB, cache0; scratch cleanup verified.",'',
            'CPU acceptance: absolute energy error <=1e-8 Eh and maximum gradient-component error <=1e-6 Eh/Bohr.','',
            '| System/version | CPU energy error, Eh | Maximum force error, Eh/Bohr | Accuracy gate |',
            '| --- | ---: | ---: | --- |']
    for row in summary['cases']:
        for version,data in row['versions'].items():
            lines.append(f"| {row['size']}/{version} | {data['maximum_cpu_energy_error']:.3e} | {data['maximum_cpu_force_error']:.3e} | {'PASS' if data['cpu_accuracy_passed'] else 'FAIL - speed ratio unqualified'} |")
    lines+=['',summary['note'],'',
        'Timing medians and memory peaks are complete-workflow measurements, not products of earlier component speedups.',
        'GB=10^9 bytes; configured staging budgets use binary GiB. Phase medians need not sum exactly to the median complete workflow.',
        'Memory-hooked timings are excluded. Requested CuPy storage and host address-space VmHWM are distinct accounting metrics; neither is total physical RAM.',
        'Disk timings use normal OS file caching without imposed RAM pressure or cache flushing; they do not establish I/O performance when the integral file exceeds available cache.',
        'CPU reference files are preexisting independently converged SCF/full-force results; no CPU reference job overlapped GPU timing.',
        'The setup-only CPU-pruning binding assertion failure is archived under setup_attempts and excluded.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
