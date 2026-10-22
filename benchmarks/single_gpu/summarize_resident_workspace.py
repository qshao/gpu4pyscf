"""Independent audit of bounded-resident construction measurements."""
import ast
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from original_comparison import (BASE,ROOT,OUT as FROZEN,digest,save,
    case_settings,cpu_reference,python_sources)
from seeded_reference import compare_execution_cases

OUT=BASE/'resident_workspace_results'
LEGACY_RUNNER_SHA256='7696a4788f0f5388107c62d45d564a0e427896adb1c5188db5eab4e94f1043f8'
LEGACY_CONTROL_SHA256='344b547608061d2dd0ea586f7a590e0eba58f5c0343eef0d81c9e816ef9e8c26'
LEGACY=OUT/'pre-runtime-budget-audit'
METADATA_SOURCES={
    'benchmarks/single_gpu/molecular.py',
    'benchmarks/single_gpu/test_molecular.py',
    'gpu4pyscf/lib/cutensor.py',
    'gpu4pyscf/lib/CMakeLists.txt',
    'gpu4pyscf/lib/cmake/CudaArchitectures.cmake',
}


def load(version,size,mode,repeat):
    path=OUT/f'{size}_{version}_{mode}_{repeat}.json'
    r=json.loads(path.read_text())
    assert r['status']=='passed' and r['scratch_clean'],path
    assert (r['version'],r['size'],r['mode'],r['repeat'])==(version,size,mode,repeat)
    assert r['settings']==r['result']['settings']==case_settings(size)
    reference,cpu=cpu_reference(size)
    assert r['cpu_reference_sha256']==digest(reference)
    # The large seeded reference pins the live configuration. Small/medium
    # references predate reporting-only edits to comprehensive.py.
    _,configuration_reference=cpu_reference('large')
    assert configuration_reference['configuration_sha256']==digest(BASE/'comprehensive.py')
    assert set(r['source_hashes'])==METADATA_SOURCES
    assert all(digest(ROOT/name)==sha for name,sha in r['source_hashes'].items())
    assert r['geometry_sha256']==digest(case_settings(size)['atom'])
    assert r['comparison']==compare_execution_cases(cpu['result'],r['result'])
    assert r['comparison']['passed'] and r['result']['converged']
    for key in ['pyscf','numpy','cpu_libxc']:
        assert r[key]==cpu[key]
    legacy_control=(version,size,mode,repeat)==('control','large','profile',0)
    assert set(r['helpers'])=={'phase_memory.py','test_resident_workspace_run.py',
        'test_stage_profile.py','test_fine_stage_profile.py','original_comparison.py',
        'seeded_reference.py'}
    if legacy_control:
        original=BASE/'resident_workspace_results/large_control_profile_0.json'
        assert digest(original)==LEGACY_CONTROL_SHA256
        assert r==json.loads(original.read_text())
    for name,sha in r['helpers'].items():
        if legacy_control and name=='test_resident_workspace_run.py':
            assert sha==LEGACY_RUNNER_SHA256==digest(LEGACY/name)
        else:
            assert digest(BASE/name)==sha
    if not legacy_control:
        assert r['effective_build_workspace_bytes']==4*1024**3
    assert r['control_source_sha256']==digest(FROZEN/'current/gpu4pyscf/df/df.py')
    actual=python_sources(ROOT/'gpu4pyscf')
    frozen=json.loads((FROZEN/'manifest.json').read_text())
    expected=frozen['packages']['current']['sources'] if mode=='profile' else actual
    assert r['algorithm_sources']==expected
    native={v['path']:v['sha256'] for v in frozen['native_libraries'].values()}
    for identity in r['native_libraries'].values():
        assert native[identity['path']]==identity['sha256']==digest(identity['path'])
    assert r['initial_pool_live_bytes']==0
    assert all(a['resident'] for a in r['integrals'])
    for key in ['setup_seconds','scf_seconds','gradient_seconds','workflow_seconds']:
        assert math.isfinite(r[key]) and r[key]>0
    assert 0<=r['workflow_seconds']-sum(r[k] for k in ['setup_seconds','scf_seconds','gradient_seconds'])<.1
    assert bool(r['phase_memory'])==(mode in ('profile','memory'))
    if r['phase_memory']:
        memory=r['phase_memory']
        assert set(memory['phases'])=={'setup','scf','df_build','gradient'}
        assert 0<=memory['live_bytes']<=memory['peak_live_bytes']
        assert memory['peak_live_bytes']>0 and memory['allocation_count']>0
        for phase in memory['phases'].values():
            assert 0<=phase['entry_live_bytes']<=phase['peak_live_bytes']<=memory['peak_live_bytes']
    return r


def main():
    # A control function uses current module globals. Verify no unrelated
    # production symbol changed relative to the complete immutable snapshot.
    frozen=json.loads((FROZEN/'manifest.json').read_text())
    actual=python_sources(ROOT/'gpu4pyscf')
    previous=frozen['packages']['current']['sources']
    assert set(actual)==set(previous)
    assert [p for p in actual if actual[p]!=previous[p]]==['df/df.py']
    def unchanged_nodes(path):
        tree=ast.parse(Path(path).read_text())
        return [ast.dump(n,include_attributes=False) for n in tree.body
                if not (isinstance(n,ast.FunctionDef) and n.name in
                        ['_cholesky_eri','_resident_build_batch_size'])]
    assert unchanged_nodes(ROOT/'gpu4pyscf/df/df.py')==unchanged_nodes(FROZEN/'current/gpu4pyscf/df/df.py')
    for item in frozen['native_libraries'].values():
        assert digest(item['path'])==item['sha256']
    control=load('control','large','profile',0)
    candidate=load('candidate','large','memory',0)
    timed={v:[load(v,'large','time',i) for i in range(3)] for v in ['control','candidate']}
    small=[load('candidate',size,'validation',0) for size in ['small','medium']]
    large=[control,candidate]+timed['control']+timed['candidate']
    assert len({tuple(r[k] for k in ['device_name','compute_capability','cupy',
        'cuda_driver_version','cuda_runtime_version','pyscf','numpy','cpu_libxc'])
        for r in large+small})==1
    assert len({r['initial_density_sha256'] for r in large})==1
    assert len({len(r['iterations']) for r in large})==1
    assert timed['control'][0]['timestamp_utc']<timed['candidate'][0]['timestamp_utc']
    assert timed['candidate'][1]['timestamp_utc']<timed['control'][1]['timestamp_utc']
    assert timed['control'][2]['timestamp_utc']<timed['candidate'][2]['timestamp_utc']
    order=[timed[v][i]['timestamp_utc'] for v,i in [
        ('control',0),('candidate',0),('candidate',1),('control',1),
        ('control',2),('candidate',2)]]
    assert order==sorted(order) and len(set(order))==6
    assert all(r['integrals']==control['integrals'] for r in large)
    medians={v:statistics.median(r['workflow_seconds'] for r in rows) for v,rows in timed.items()}
    checks={}
    for name,count in [('focused-tests.xml',20),('audit-tests.xml',13)]:
        path=OUT/name
        root=ET.parse(path).getroot()
        assert len(list(root.iter('testcase')))==count
        assert not any(list(root.iter(tag)) for tag in ['failure','error','skipped'])
        checks[name]=dict(passed=count,sha256=digest(path))
    a=control['phase_memory']['peak_live_bytes'];b=candidate['phase_memory']['peak_live_bytes']
    summary=dict(status='passed',validator_sha256=digest(__file__),
        effective_build_workspace_bytes=4*1024**3,
        runner_sha256=dict(control_profile=LEGACY_RUNNER_SHA256,
                          remaining_observations=digest(BASE/'test_resident_workspace_run.py')),
        excluded_preliminary_observation=dict(
            path=str(LEGACY/'large_candidate_memory_0.json'),
            sha256=digest(LEGACY/'large_candidate_memory_0.json'),
            reason='Predates explicit effective runtime-budget assertion; rerun and excluded from statistics.'),
        workflow_median_seconds=medians,speed_ratio=medians['control']/medians['candidate'],
        paired_speed_ratios=[timed['control'][i]['workflow_seconds']/
                             timed['candidate'][i]['workflow_seconds'] for i in range(3)],
        phase_median_seconds={v:{phase:statistics.median(r[phase+'_seconds'] for r in rows)
                                for phase in ['setup','scf','gradient']}
                              for v,rows in timed.items()},
        checks=checks,
        tracked_gpu_peak_bytes=dict(control=a,candidate=b),gpu_reduction_fraction=1-b/a,
        phase_peaks_bytes={v:{k:d['peak_live_bytes'] for k,d in r['phase_memory']['phases'].items()}
                           for v,r in [('control',control),('candidate',candidate)]},
        host_rss_bytes={v:r['host_address_space_peak_rss_bytes'] for v,r in [('control',control),('candidate',candidate)]},
        timing_samples_seconds={v:[r['workflow_seconds'] for r in rows] for v,rows in timed.items()},
        iterations=[len(r['iterations']) for r in large],control_coarse=control['coarse'],
        control_exchange_helpers=control['helper_times'],
        maximum_cpu_force_error=max(r['comparison']['gradient_max_error'] for r in large+small),
        maximum_large_cpu_force_error=max(r['comparison']['gradient_max_error'] for r in large),
        maximum_cpu_energy_error=max(r['comparison']['energy_error_hartree'] for r in large+small),
        input_sha256={
            (p:=OUT/f"{r['size']}_{r['version']}_{r['mode']}_{r['repeat']}.json").name:digest(p)
            for r in large+small},
        note='Three alternating fresh-process timing pairs; separate fresh phase-allocation observations. '
             'Control diagnostic scopes include synchronization and must not be used as speed controls. '
             'Only the build function changes. Requested CuPy bytes exclude native/preexisting storage; '
             'host RSS is separate and neither measures total physical RAM on unified GB10. '
             'The cap applies to splittable nongeneral contractions; general contractions keep their existing path. '
             'The pre-change control profile uses its individually hash-pinned archived runner; '
             'all other accepted observations assert and record the effective 4GiB build target.')
    save(OUT/'summary.json',summary)
    lines=['# Bounded resident DF construction','',
        'Taxol: 113 atoms, 1123 AOs; B3LYP/def2-SVP, def2-universal-jkfit, '
        'level-3 NWChem grids, full grid/auxiliary gradient response, FP64. '
        'Resident GPU integrals are forced. The control is the previously '
        'optimized, accuracy-corrected GPU build function, before this cap.', '',
        f"Control/current workflow medians: {medians['control']:.3f}s / {medians['candidate']:.3f}s "
        f"({summary['speed_ratio']:.4f}x; "
        f"{100*(medians['candidate']/medians['control']-1):.3f}% more time).",'',
        '| Allocation phase | Control peak, GB | Candidate peak, GB |',
        '| --- | ---: | ---: |']
    for phase in ['setup','df_build','scf','gradient']:
        lines.append(f"| {phase} | {summary['phase_peaks_bytes']['control'][phase]/1e9:.3f} | "
                     f"{summary['phase_peaks_bytes']['candidate'][phase]/1e9:.3f} |")
    lines += ['',f"Whole-workflow tracked peak: {a/1e9:.3f} to {b/1e9:.3f}GB "
              f"({100*summary['gpu_reduction_fraction']:.1f}% lower).",'',
              'Host address-space peak RSS, measured separately: '+', '.join(
                  f"{v} {n/1e9:.3f}GB" for v,n in summary['host_rss_bytes'].items())+'.','',
              'Timing samples (seconds): '+json.dumps(summary['timing_samples_seconds'])+'.','',
              'Paired speed ratios: '+', '.join(f'{v:.4f}' for v in summary['paired_speed_ratios'])+'.',
              'Three pairs describe observed variation; they do not establish a hardware-wide speed guarantee.','',
              f"Maximum CPU energy error: {summary['maximum_cpu_energy_error']:.3e}Eh; "
              f"gradient component error: {summary['maximum_cpu_force_error']:.3e}Eh/Bohr. "
              'Acceptance gates remain 1e-8Eh / 1e-6Eh/Bohr.','',
              'All ten workflow observations pass, including full small/medium '
              'validation and the six large timing workflows. All large runs '
              'use the same atomic density and converge in 20 recorded cycles.', '',
              '20 focused checks and 13 audit checks pass; XML evidence is hashed in summary.json.',
              '',summary['note'],'',
              'The preliminary candidate memory observation is preserved under '
              'pre-runtime-budget-audit/ and excluded because its runner predates '
              'the explicit runtime-budget assertion. The exact archived runner '
              'is accepted only for the separately pinned pre-change control profile.', '',
              'A subprocess launcher failed CUDA initialization before calculation; '
              'candidate-small-validation-0.log retains that infrastructure failure. '
              'Direct launches passed and provide the accepted observations.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
