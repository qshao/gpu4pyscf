"""Independently validate saved speed/memory evidence; uses only the stdlib."""
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    result = json.loads((BASE / name).read_text())
    assert result.get('status') == 'passed', name
    return result


def flat(value):
    if isinstance(value, list):
        return [x for item in value for x in flat(item)]
    return [value]


def difference(a, b):
    a, b = flat(a), flat(b)
    assert len(a) == len(b) and all(math.isfinite(x) for x in a + b)
    return max((abs(x - y) for x, y in zip(a, b)), default=0.)


def function(source, name):
    return next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def summarize():
    accept = read('speed_memory_acceptance_results.json')
    experiment = read('speed_memory_workflow_results.json')
    components = read('speed_memory_components.json')
    memory = read('speed_memory_usage_current.json')
    old_name = 'runs/2026-10-17-pre-promotion-speed-memory/speed_memory_usage_current.json'
    old_memory = read(old_name)
    refs = read('stage_profile_results.json')
    snapshot_path = BASE / 'speed_memory_baseline.json'
    snapshot = json.loads(snapshot_path.read_text())
    for report in [accept, memory]:
        for path, sha in report['algorithm_source_hashes'].items():
            assert digest(ROOT / path) == sha, path
        for path, sha in report['source_hashes'].items():
            assert digest(ROOT / path) == sha, path
    for report in [accept, memory, old_memory, experiment]:
        for identity in report['native_libraries'].values():
            assert digest(identity['path']) == identity['sha256']
        assert report['baseline_sha256'] == digest(snapshot_path)
    for name, report in [('acceptance', accept), ('workflow', experiment)]:
        assert digest(BASE / f'test_speed_memory_{name}.py') == report['benchmark_sha256']
        assert digest(BASE / 'stage_profile_results.json') == report['reference_sha256']
    assert digest(BASE / 'test_speed_memory_qualification.py') == memory['benchmark_sha256']
    assert digest(BASE / 'memory_profile.py') == memory['memory_profiler_sha256']
    assert digest(BASE / 'speed_memory_candidates.py') == experiment['candidate_sha256'] == components['candidate_sha256']
    for module in snapshot['modules'].values():
        archived = (ROOT / module['archive']).read_text()
        assert digest(ROOT / module['archive']) == module['sha256']
        for name, source in module['functions'].items():
            assert ast.dump(function(archived, name)) == ast.dump(function(source, name))
    old_df = (ROOT / snapshot['modules']['df_jk']['archive']).read_text()
    new_df = (ROOT / 'gpu4pyscf/df/df_jk.py').read_text()
    for name in ['_exchange_transform', '_symmetric_gram']:
        assert ast.dump(function(old_df, name)) == ast.dump(function(new_df, name))
    assert digest(ROOT / snapshot['modules']['cutensor']['path']) == snapshot['modules']['cutensor']['sha256']
    for path, sha in old_memory['algorithm_source_hashes'].items():
        module = next(m for m in snapshot['modules'].values() if m['path'] == path)
        assert sha == module['sha256']

    errors = dict(records=0, cpu_energy_hartree=0., cpu_gradient_hartree_per_bohr=0.,
                  paired_energy_hartree=0., paired_gradient_hartree_per_bohr=0.)

    def check(case, entry):
        gpu = entry['result']
        cpu = next(r['cpu_reference'] for r in refs['cases']
                   if r['case'] == case and r['method'] == gpu['method'])
        assert cpu['settings'] == gpu['settings']
        assert cpu['converged'] and gpu['converged'] and entry['comparison']['passed']
        energy = difference(cpu['energy_hartree'], gpu['energy_hartree'])
        force = difference(cpu['gradient_hartree_per_bohr'], gpu['gradient_hartree_per_bohr'])
        assert energy <= 1e-8 and force <= 1e-6
        errors['records'] += 1
        errors['cpu_energy_hartree'] = max(errors['cpu_energy_hartree'], energy)
        errors['cpu_gradient_hartree_per_bohr'] = max(errors['cpu_gradient_hartree_per_bohr'], force)

    def assess(row, versions):
        for item in row['warmups']:
            check(row['case'], item)
        ratios = []
        for pair in row['pairs']:
            for entry in pair.values():
                check(row['case'], entry)
            a, b = (pair[v]['result'] for v in versions)
            assert a['iterations'] == b['iterations']
            errors['paired_energy_hartree'] = max(errors['paired_energy_hartree'], difference(a['energy_hartree'], b['energy_hartree']))
            errors['paired_gradient_hartree_per_bohr'] = max(errors['paired_gradient_hartree_per_bohr'], difference(a['gradient_hartree_per_bohr'], b['gradient_hartree_per_bohr']))
            ratios.append((a['scf_seconds'] + a['gradient_seconds']) / (b['scf_seconds'] + b['gradient_seconds']))
        medians = {v: statistics.median(p[v]['result']['scf_seconds'] + p[v]['result']['gradient_seconds']
                                      for p in row['pairs']) for v in versions}
        speedup = medians[versions[0]] / medians[versions[1]]
        assert math.isclose(speedup, row['speedup'], rel_tol=1e-12)
        assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(ratios, row['paired_ratios']))
        if 'instrumented' in row:
            check(row['case'], row['instrumented'])
            assert row['fine']['bindings_restored'] and row['actual_gemv_calls'] > 0
            assert all(p[versions[1]]['result']['iterations'] == row['instrumented']['result']['iterations'] for p in row['pairs'])
        return dict(case=row['case'], pairs=len(row['pairs']), medians_seconds=medians,
                    speedup=speedup, time_reduction_percent=100 * (1 - 1 / speedup),
                    paired_ratios=ratios,
                    production_phase_medians={phase: statistics.median(p[versions[1]]['result'][phase + '_seconds']
                                                                      for p in row['pairs']) for phase in ['scf', 'gradient']})

    production = [assess(row, ['previous', 'production']) for row in accept['cases']]
    isolated = [assess(row, ['current', 'combined']) for row in experiment['cases']]
    for report in [old_memory, memory]:
        check('tamoxifen_svp', report)
        assert report['memory']['initial_pool_live_bytes'] == 0
    before, after = (r['memory']['peak_tracked_live_bytes'] for r in [old_memory, memory])
    names = ['speed_memory_acceptance_results.json', 'speed_memory_workflow_results.json',
             'speed_memory_components.json', 'speed_memory_usage_current.json', old_name]
    result = dict(status='passed', production=production, experimental=isolated,
                  accuracy=errors,
                  summary_script_sha256=digest(__file__),
                  memory=dict(previous_peak_tracked_live_bytes=before,
                              production_peak_tracked_live_bytes=after,
                              reduction_percent=100 * (1 - after / before),
                              previous_host_address_space_peak_rss_bytes=old_memory['memory']['host_address_space_peak_rss_bytes'],
                              production_host_address_space_peak_rss_bytes=memory['memory']['host_address_space_peak_rss_bytes']),
                  production_source_hashes=accept['algorithm_source_hashes'],
                  raw_report_hashes={n: digest(BASE / n) for n in names},
                  note='Single shared GB10, CuPy fallback, regression grid level 1. Saved approximately42-second GPU baseline; unchanged native build and scientific settings. Memory is live CuPy-requested storage, excludes native/library/preexisting allocations. Host RSS unchanged; unified-memory metrics must not be added. Workspace cap is not a full-memory cap. No arbitrary large chemical calculation or production grid qualified.')
    (BASE / 'speed_memory_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    summarize()
