"""CPU-only, fail-closed audit of the fresh code-review evidence.

Never executes SCF, imports CuPy, or substitutes historical campaign results.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from original_comparison import BASE, ROOT, case_settings, cpu_reference, digest, python_sources, save
from seeded_reference import compare_execution_cases

OUT = BASE / 'code_review_results'
CHANGED = {'gpu4pyscf/df/df_jk.py', 'gpu4pyscf/df/grad/rhf.py',
           'gpu4pyscf/grad/rks.py', 'gpu4pyscf/lib/cutensor.py', 'setup.py'}
CASES = {'matrix_64_float64': [[64, 64], [64, 64]],
         'matrix_512_float64': [[512, 512], [512, 512]],
         'matrix_1024_float64': [[1024, 1024], [1024, 1024]],
         'matrix_512_complex128': [[512, 512], [512, 512]],
         'grid_3_4096': [[3, 3, 4096], [4096]],
         'grid_113_32768': [[113, 3, 32768], [32768]]}
COMPONENT_SOURCES = {'gpu4pyscf/lib/cutensor.py', 'gpu4pyscf/grad/rks.py',
                     'benchmarks/single_gpu/test_code_review_components.py',
                     'benchmarks/single_gpu/phase_memory.py',
                     'benchmarks/single_gpu/molecular.py'}
METADATA_SOURCES = {'benchmarks/single_gpu/molecular.py',
                    'benchmarks/single_gpu/test_molecular.py',
                    'gpu4pyscf/lib/cutensor.py', 'gpu4pyscf/lib/CMakeLists.txt',
                    'gpu4pyscf/lib/cmake/CudaArchitectures.cmake'}
HELPERS = {'test_code_review_workflows.py', 'original_comparison.py',
           'molecular.py', 'comprehensive.py', 'seeded_reference.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite numeric evidence')
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def load(path):
    data = json.loads(Path(path).read_text())
    finite_tree(data)
    return data


def close(actual, expected, label):
    require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), label)


def hashes(record, root, expected_names=None):
    if expected_names is not None:
        require(set(record) == set(expected_names), 'Source manifest coverage changed')
    for name, expected in record.items():
        require(digest(Path(root) / name) == expected, 'Stale source: ' + name)


def audit_metadata(record):
    hashes(record['source_hashes'], ROOT, METADATA_SOURCES)
    require(set(record['native_libraries']) == {'cupy_helper', 'libxc', 'gdft', 'vhf_rys'},
            'Native metadata coverage changed')
    for library in record['native_libraries'].values():
        path = Path(library['path'])
        require(path.stat().st_size == library['size_bytes'] and digest(path) == library['sha256'],
                'Stale native library: ' + str(path))
    require(record['backend'] == 'gpu' and record['compute_capability'] == '121',
            'Unexpected execution backend')


def audit_baseline(directory=OUT):
    manifest = load(directory / 'before_manifest.json')
    sources = manifest['sources']
    require(len(sources) == 549, 'Incomplete before snapshot')
    hashes(sources, directory / 'before')
    current = {name: digest(ROOT / name) for name in sources}
    changed = {name for name in sources if current[name] != sources[name]}
    require(changed == CHANGED, 'Unexpected changed production sources: ' + str(sorted(changed)))
    return dict(manifest_sha256=digest(directory / 'before_manifest.json'),
                verified_file_count=len(sources),
                source_hashes={name: sources[name] for name in
                               ['gpu4pyscf/lib/cutensor.py', 'gpu4pyscf/grad/rks.py']})


def audit_error(check):
    require(check['finite'] is True and check['passed'] is True, 'Failed numeric check')
    require(check['atol'] == 1e-9 and check['rtol'] == 1e-12, 'Relaxed component error gate')
    absolute, relative = check['max_absolute_error'], check['relative_to_reference_max']
    require(isinstance(absolute, (float, int)) and isinstance(relative, (float, int)), 'Missing numeric error')
    require(absolute >= 0 and relative >= 0 and math.isfinite(absolute) and math.isfinite(relative),
            'Invalid numeric error')
    require((absolute == 0) == (relative == 0), 'Inconsistent numeric error')
    scale = absolute / relative if relative else 0
    require(absolute <= 1e-9 + 1e-12 * scale, 'Error exceeds recorded gate')


def audit_allocation(record, name):
    for field in ['live_bytes', 'peak_live_bytes', 'allocation_count']:
        require(type(record[field]) is int and record[field] >= 0, 'Invalid allocation ' + field)
    require(record['peak_live_bytes'] >= record['live_bytes'], 'Peak below live allocation')
    require(set(record['phases']) == {name}, 'Invalid allocation phase')
    phase = record['phases'][name]
    for field in ['entry_live_bytes', 'peak_live_bytes', 'allocation_count', 'entry_count']:
        require(type(phase[field]) is int and phase[field] >= 0, 'Invalid phase allocation')
    require(phase['entry_count'] == 1 and phase['peak_live_bytes'] >= phase['entry_live_bytes']
            and phase['peak_live_bytes'] <= record['peak_live_bytes']
            and phase['allocation_count'] == record['allocation_count'], 'Inconsistent allocation phase')


def audit_components(report, baseline=None):
    finite_tree(report)
    require(report['status'] == 'passed', 'Components did not pass')
    require(len(report['cases']) == len(CASES)
            and {row['case'] for row in report['cases']} == set(CASES), 'Missing or repeated component case')
    require(report['source_hashes_before'] == report['source_hashes_after'], 'Sources changed during components')
    hashes(report['source_hashes_before'], ROOT, COMPONENT_SOURCES)
    require(report['baseline_identity_before'] == report['baseline_identity_after'], 'Baseline changed during components')
    if baseline is not None:
        require(report['baseline_identity_before'] == baseline, 'Stale baseline identity')
    for side in ['before', 'after']:
        audit_metadata(report['metadata_' + side])
    for key in ['native_libraries', 'source_hashes']:
        require(report['metadata_before'][key] == report['metadata_after'][key], 'Metadata changed during components')
    # Re-extract the production statements; no CUDA import is involved.
    from test_code_review_components import _grid_updates
    for side, root in [('baseline', OUT / 'before'), ('current', ROOT)]:
        _, statement = _grid_updates((root / 'gpu4pyscf/grad/rks.py').read_text(), side)
        require(statement == report['grid_update_statements'][side], 'Stale grid statements')
    summaries = []
    for row in report['cases']:
        name = row['case']
        require(row['shapes'] == CASES[name], 'Incorrect component shape')
        require(row['input_sha256_before'] == row['input_sha256_after']
                and len(row['input_sha256_before']) == 64
                and all(c in '0123456789abcdef' for c in row['input_sha256_before']), 'Input mutation or invalid identity')
        repeats = 30 if name in ['matrix_64_float64', 'grid_3_4096'] else 5
        require(row['repeats_per_sample'] == repeats, 'Incorrect repeats')
        require(set(row['checks']) == {'baseline_reference', 'current_reference', 'current_baseline'}, 'Missing numeric check')
        for check in row['checks'].values():
            audit_error(check)
        if name.startswith('matrix'):
            require(set(row['cpu_oracle_checks']) == {'baseline', 'current'}, 'Missing CPU oracle')
            for check in row['cpu_oracle_checks'].values():
                audit_error(check)
        else:
            require(row['derivative_input_bytes'] == math.prod(row['shapes'][0]) * 8,
                    'Incorrect derivative size')
        require(set(row['samples']) == {'baseline', 'current'}, 'Missing paired samples')
        medians = {}
        for side in ['baseline', 'current']:
            observations = row['samples'][side]
            require(len(observations) == 7, 'Need seven paired samples')
            for observation in observations:
                for clock in ['wall_us', 'gpu_us']:
                    require(observation[clock] > 0, 'Invalid timing')
                if name.startswith('grid'):
                    require(len(observation['calls']) == repeats, 'Incomplete per-call observations')
                    for clock in ['wall_us', 'gpu_us']:
                        calls = [call[clock] for call in observation['calls']]
                        require(all(value > 0 for value in calls), 'Invalid per-call timing')
                        close(observation[clock], statistics.median(calls), 'Stale per-call median')
            medians[side] = {clock: statistics.median(o[clock] for o in observations)
                             for clock in ['wall_us', 'gpu_us']}
            for clock, value in medians[side].items():
                close(row['median_us'][side][clock], value, 'Stale median')
            audit_allocation(row['requested_allocation'][side], side)
        ratios = {clock: medians['baseline'][clock] / medians['current'][clock]
                  for clock in ['wall_us', 'gpu_us']}
        for clock, value in ratios.items():
            close(row['speedup'][clock], value, 'Stale speed ratio')
        summaries.append(dict(case=name, median_us=medians, speedup=ratios,
                              requested_peak_bytes={side: row['requested_allocation'][side]['peak_live_bytes']
                                                    for side in ['baseline', 'current']}))
    return summaries


def audit_workflow(report, size, baseline):
    finite_tree(report)
    require(report['status'] == 'passed' and report['size'] == size, 'Workflow not passed: ' + size)
    config = case_settings(size)
    require(report['settings'] == config and report['result']['settings'] == config, 'Stale scientific settings')
    require(report['baseline_manifest_sha256'] == baseline['manifest_sha256'], 'Stale workflow baseline')
    hashes(report['algorithm_sources'], ROOT / 'gpu4pyscf', python_sources(ROOT / 'gpu4pyscf'))
    hashes(report['helpers'], BASE, HELPERS)
    audit_metadata(report)
    require(report['geometry_sha256'] == digest(config['atom']), 'Stale geometry')
    reference_path, cpu = cpu_reference(size)
    finite_tree(cpu)
    require(report['cpu_reference_sha256'] == digest(reference_path), 'Stale CPU reference')
    for key in ['pyscf', 'numpy', 'cpu_libxc']:
        require(report[key] == cpu[key], 'CPU/GPU environment mismatch: ' + key)
    # Pin the CPU runner/configuration metadata without rerunning the oracle.
    for key in ['source_hashes']:
        if key in cpu:
            for name, sha in cpu[key].items():
                if name.startswith('benchmarks/'):
                    require(digest(ROOT / name) == sha, 'Stale CPU reference helper: ' + name)
    previous_path = BASE / 'resident_workspace_results' / (
        f'{size}_candidate_' + ('memory' if size == 'large' else 'validation') + '_0.json')
    previous = load(previous_path)
    require(previous['status'] == 'passed' and previous['settings'] == config,
            'Invalid previous initial-density record')
    require(report['previous_input_report_sha256'] == digest(previous_path)
            and report['initial_density_sha256'] == previous['initial_density_sha256'], 'Changed initial density')
    require(report['effective_build_workspace_bytes'] == 4 * 1024**3, 'Changed workspace control')
    comparison = compare_execution_cases(cpu['result'], report['result'])
    finite_tree(comparison)
    require(comparison['passed'] is True, 'Independent CPU comparison failed')
    require(report['comparison'] == comparison, 'Stale recorded CPU comparison')
    for key in ['setup_seconds', 'scf_seconds', 'gradient_seconds', 'workflow_seconds']:
        require(report[key] > 0, 'Invalid workflow elapsed time')
    return comparison


def audit_regression(path):
    root = ET.parse(path).getroot()
    cases = root.findall('.//testcase')
    require(len(cases) == 178, 'Incomplete final regression suite')
    require(not root.findall('.//failure') and not root.findall('.//error')
            and not root.findall('.//skipped'), 'Regression suite has nonpassing cases')
    return len(cases)


def main():
    baseline = audit_baseline()
    components = audit_components(load(OUT / 'components.json'), baseline)
    workflows = {size: audit_workflow(load(OUT / f'{size}_workflow.json'), size, baseline)
                 for size in ['medium', 'large']}
    count = audit_regression(OUT / 'regression-suite.xml')
    inputs = ['before_manifest.json', 'components.json', 'medium_workflow.json',
              'large_workflow.json', 'regression-suite.xml']
    summary = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   baseline=baseline, changed_production_sources=sorted(CHANGED),
                   component_cases=components, independent_cpu_comparisons=workflows,
                   regression_tests=count, input_sha256={name: digest(OUT / name) for name in inputs},
                   auditor_sha256=digest(__file__),
                   scope='Component timing and requested allocation only. Fresh medium/large workflows establish accuracy, not a workflow speed ratio. No physical RAM bound or 20 percent speedup claim.')
    save(OUT / 'summary.json', summary)
    lines = ['# Code-review component evidence', '',
             'Control: pinned optimized working tree before review, not original HEAD. Seven alternating pairs; wall and CUDA-event medians recomputed from raw observations.', '',
             '| Component | Wall ratio | GPU ratio | Baseline requested peak bytes | Current requested peak bytes |',
             '|---|---:|---:|---:|---:|']
    for row in components:
        peaks = row['requested_peak_bytes']
        lines.append(f"| {row['case']} | {row['speedup']['wall_us']:.4f}x | {row['speedup']['gpu_us']:.4f}x | {peaks['baseline']} | {peaks['current']} |")
    lines += ['', 'Ratios below 1 show regressions. Grid timings exclude derivative construction, input restoration, accumulator reset and synchronization before each call; per-call observations are aggregated by median. Allocation hooks run separately from timing. The grid optimization consumes a fresh owned derivative in place, then retains the original unary reduction.', '',
              'Requested allocation records exclude preexisting inputs/destinations, pool reservation, native allocation and physical RAM; grid derivative input storage is still required. These measurements establish no full-workflow acceleration or 20 percent speedup. Medium and large scientific acceptance independently recomputes CPU-reference errors at the unchanged 1e-8 Eh and 1e-6 Eh/Bohr gates.', '',
              f'Final focused regression suite: {count} passed. Complete source snapshot, current sources, helpers, geometry, native libraries and reference records were rehashed by the CPU-only auditor.', '']
    (OUT / 'component_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    main()
