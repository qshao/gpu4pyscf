"""Validate production identities, accuracy, memory and transfer evidence."""
import hashlib
import json
import math
from pathlib import Path
import statistics


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(value):
    if isinstance(value, list):
        return [x for item in value for x in flatten(item)]
    return [value]


def difference(a, b):
    a, b = flatten(a), flatten(b)
    assert len(a) == len(b) and all(math.isfinite(x) for x in a+b)
    return max(abs(x-y) for x, y in zip(a, b))


def summarize():
    refs_path = BASE/'stage_profile_results.json'
    refs = json.loads(refs_path.read_text())
    assert refs['status'] == 'passed'
    reports = {}
    errors = dict(records=0, max_energy_hartree=0., max_gradient_hartree_per_bohr=0.)
    def identity(report, benchmark):
        assert report['status'] == 'passed'
        assert sha(BASE/benchmark) == report['benchmark_sha256']
        assert sha(refs_path) == report['reference_sha256']
        for path, digest in report['algorithm_source_hashes'].items():
            assert sha(ROOT/path) == digest, path
        for path, digest in report['source_hashes'].items():
            assert sha(ROOT/path) == digest, path
        for item in report['native_libraries'].values():
            assert sha(item['path']) == item['sha256']

    def check(case, entry):
        result = entry['result']
        ref = next(r for r in refs['cases'] if r['case']==case and r['method']=='B3LYP')
        cpu = ref['cpu_reference']
        assert sha(cpu['settings']['atom']) == ref['geometry_sha256']
        assert result['settings'] == cpu['settings']
        for key in ['pyscf','numpy','cpu_libxc']:
            assert result[key] == cpu[key]
        assert result['native_libraries'] == ref['instrumented']['native_libraries']
        assert result['converged'] and cpu['converged'] and entry['comparison']['passed']
        energy = difference(cpu['energy_hartree'], result['energy_hartree'])
        force = difference(cpu['gradient_hartree_per_bohr'], result['gradient_hartree_per_bohr'])
        assert energy <= 1e-8 and force <= 1e-6
        errors['records'] += 1
        errors['max_energy_hartree'] = max(errors['max_energy_hartree'], energy)
        errors['max_gradient_hartree_per_bohr'] = max(errors['max_gradient_hartree_per_bohr'], force)

    modes = ['resident','host','disk','cached']
    capacity = {}
    for case in ['vitamin_svp','tamoxifen_svp']:
        rows = {}
        for mode in modes:
            name = f'disk_df_memory_{case}_{mode}.json'
            report = reports[name] = json.loads((BASE/name).read_text())
            identity(report, 'test_disk_df_qualification.py')
            assert report['cderi_released_by_gradient']
            assert report['memory_profiler_sha256'] == sha(BASE/'memory_profile.py')
            check(case, report)
            memory = report['memory']
            assert memory['initial_pool_live_bytes'] == 0
            assert len(report['storage']) == 1
            tensor = report['storage'][0]
            row = dict(peak_tracked_cupy_bytes=memory['peak_tracked_live_bytes'],
                       host_address_space_peak_rss_bytes=memory['host_address_space_peak_rss_bytes'],
                       tensor_bytes=tensor['nbytes'], iterations=report['result']['iterations'])
            if mode in ['disk','cached']:
                stats = tensor['stats']
                assert stats['d2h_bytes'] == stats['write_bytes'] == tensor['nbytes']
                assert stats['h2d_bytes'] == stats['read_bytes']
                assert stats['peak_host_buffer_bytes'] <= report['storage_buffer_bytes']
                assert stats['peak_device_buffer_bytes'] <= report['storage_buffer_bytes']
                assert stats['cache_device_bytes'] <= report['cache_budget_bytes']
                assert (stats['h2d_bytes']+stats['cache_hit_bytes']) % tensor['nbytes'] == 0
                row.update(stats=stats, staging_budget_bytes=report['storage_buffer_bytes'],
                           cache_budget_bytes=report['cache_budget_bytes'])
            elif mode=='host':
                row['h2d_bytes_inferred'] = report['host_streamed_bytes_inferred']
            rows[mode] = row
        assert len({rows[m]['iterations'] for m in ['resident','host','disk']}) == 1
        disk, cached = rows['disk']['stats'], rows['cached']['stats']
        assert rows['host']['h2d_bytes_inferred'] == disk['h2d_bytes']
        disk_passes = disk['h2d_bytes']/rows['disk']['tensor_bytes']
        cache_passes = (cached['h2d_bytes']+cached['cache_hit_bytes'])/rows['cached']['tensor_bytes']
        rows['tensor_passes'] = dict(disk=disk_passes,cached=cache_passes)
        used_cache_bytes = cached['cache_hit_bytes']/cache_passes
        assert used_cache_bytes <= cached['cache_device_bytes']
        rows['used_cache_bytes_per_pass'] = used_cache_bytes
        rows['cache_h2d_reduction_percent'] = 100*(1-cached['h2d_bytes']/disk['h2d_bytes'])
        rows['cache_h2d_per_pass_reduction_percent'] = 100*(1-(cached['h2d_bytes']/cache_passes)/(disk['h2d_bytes']/disk_passes))
        rows['disk_vs_resident_cupy_reduction_percent'] = 100*(1-rows['disk']['peak_tracked_cupy_bytes']/rows['resident']['peak_tracked_cupy_bytes'])
        rows['disk_vs_host_rss_reduction_percent'] = 100*(1-rows['disk']['host_address_space_peak_rss_bytes']/rows['host']['host_address_space_peak_rss_bytes'])
        capacity[case] = rows

    name = 'disk_df_speed_results.json'
    speed = reports[name] = json.loads((BASE/name).read_text())
    identity(speed, 'test_disk_df_speed.py')
    for entry in speed['warmups']:
        check(speed['case'], entry)
    for row in speed['rounds']:
        assert len({row[m]['result']['iterations'] for m in ['resident','host','disk']}) == 1
        for entry in row.values():
            check(speed['case'], entry)
    medians = {mode:statistics.median(r[mode]['result']['scf_seconds']+r[mode]['result']['gradient_seconds']
                                     for r in speed['rounds']) for mode in modes}
    assert medians == speed['medians_seconds']
    speed_transfers = {m:dict(median_h2d_bytes=statistics.median(
        r[m]['transfers'][0]['h2d_bytes'] for r in speed['rounds']),
        median_iterations=statistics.median(r[m]['result']['iterations'] for r in speed['rounds']))
        for m in ['disk','cached']}
    prior = json.loads((BASE/'speed_memory_summary.json').read_text())
    for path,digest in prior['production_source_hashes'].items():
        assert sha(ROOT/path) == digest  # prior contraction/gradient changes preserved
    archive = BASE/'runs/2026-10-17-pre-disk-df'
    for path,digest in json.loads((archive/'identity.json').read_text()).items():
        assert sha(archive/(Path(path).name+'.source')) == digest
    diagnostic_name='disk_df_repeated_jk.json'
    diagnostic=reports[diagnostic_name]=json.loads((BASE/diagnostic_name).read_text())
    assert diagnostic['status']=='passed'
    assert all(math.isfinite(v) and v<=1e-10 for row in diagnostic['rows']
               for pair in row['errors'] for v in pair)
    result = dict(status='passed', capacity=capacity, accuracy=errors,
                  speed_case=speed['case'], speed_rounds=len(speed['rounds']),
                  speed_medians_seconds=medians,
                  cache_performance_status=speed['cache_performance_status'],
                  speed_iteration_counts=speed['iteration_counts'],
                  speed_transfer_totals=speed_transfers,
                  production_source_hashes=speed['algorithm_source_hashes'],
                  raw_report_hashes={name:sha(BASE/name) for name in reports},
                  summary_script_sha256=sha(__file__),
                  diagnostic_benchmark_sha256=sha(BASE/'test_disk_df_diagnostics.py'),
                  note='Single shared GB10/ext4 NVMe, gridlevel1. Logical H2D bytes, not physical bus traffic. Small3round timings are separate from instrumented fresh-process memory runs. Native/library memory, OS cache and persistent metric/factor tensors are excluded from staging budget; cache has separate explicit budget. Do not add GPU and host figures on unified memory. Largecase uses1GiB staging,128MiB default insufficient for its995840640B indivisible construction workspace. Disk mode is capacity feature, resident default retained.')
    (BASE/'disk_df_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    summarize()
