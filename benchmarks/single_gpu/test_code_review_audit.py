"""CPU-only rejection tests using copies of the fresh raw evidence."""
import copy

import pytest

import summarize_code_review as audit


@pytest.fixture
def components():
    return copy.deepcopy(audit.load(audit.OUT / 'components.json'))


def test_actual_component_evidence(components):
    result = audit.audit_components(components, audit.audit_baseline())
    assert len(result) == 6
    assert result[-1]['speedup']['wall_us'] < 1
    assert result[-1]['requested_peak_bytes']['current'] < result[-1]['requested_peak_bytes']['baseline']


@pytest.mark.parametrize('corruption', ['source', 'ratio', 'median', 'missing_case',
    'duplicate_case', 'shape', 'negative_peak', 'phase_peak', 'input', 'nonfinite',
    'gate', 'oracle', 'failed_check', 'missing_pair', 'call_median', 'statements'])
def test_component_corruptions_rejected(components, corruption):
    row = components['cases'][0]
    if corruption == 'source':
        components['source_hashes_before']['gpu4pyscf/lib/cutensor.py'] = '0' * 64
        components['source_hashes_after'] = dict(components['source_hashes_before'])
    elif corruption == 'ratio':
        row['speedup']['wall_us'] *= 1.1
    elif corruption == 'median':
        row['median_us']['current']['wall_us'] *= 1.1
    elif corruption == 'missing_case':
        components['cases'].pop()
    elif corruption == 'duplicate_case':
        components['cases'][-1] = copy.deepcopy(row)
    elif corruption == 'shape':
        row['shapes'][0][0] += 1
    elif corruption == 'negative_peak':
        row['requested_allocation']['baseline']['peak_live_bytes'] = -1
    elif corruption == 'phase_peak':
        row['requested_allocation']['baseline']['phases']['baseline']['peak_live_bytes'] = 10**10
    elif corruption == 'input':
        row['input_sha256_after'] = '0' * 64
    elif corruption == 'nonfinite':
        row['samples']['current'][0]['wall_us'] = float('nan')
    elif corruption == 'gate':
        row['checks']['baseline_reference']['atol'] = 1e-3
    elif corruption == 'oracle':
        del row['cpu_oracle_checks']['current']
    elif corruption == 'failed_check':
        row['checks']['current_baseline']['passed'] = False
    elif corruption == 'missing_pair':
        row['samples']['current'].pop()
    elif corruption == 'call_median':
        components['cases'][-1]['samples']['current'][0]['calls'][0]['wall_us'] = -1
    elif corruption == 'statements':
        components['grid_update_statements']['current'] = 'pass'
    with pytest.raises(ValueError):
        audit.audit_components(components)


def test_actual_medium_cpu_comparison():
    report = audit.load(audit.OUT / 'medium_workflow.json')
    assert audit.audit_workflow(report, 'medium', audit.audit_baseline())['passed']


@pytest.mark.parametrize('corruption', ['settings', 'recorded_error', 'force', 'reference',
                                       'density', 'helper', 'running', 'workspace'])
def test_workflow_corruptions_rejected(corruption):
    report = copy.deepcopy(audit.load(audit.OUT / 'medium_workflow.json'))
    if corruption == 'settings':
        report['settings']['grid_level'] = 1
    elif corruption == 'recorded_error':
        report['comparison']['gradient_max_error'] = 0
    elif corruption == 'force':
        report['result']['gradient_hartree_per_bohr'][0][0] += 1e-3
    elif corruption == 'reference':
        report['cpu_reference_sha256'] = '0' * 64
    elif corruption == 'density':
        report['initial_density_sha256'] = '0' * 64
    elif corruption == 'helper':
        report['helpers']['molecular.py'] = '0' * 64
    elif corruption == 'running':
        report['status'] = 'running'
    elif corruption == 'workspace':
        report['effective_build_workspace_bytes'] = 0
    baseline = dict(manifest_sha256=audit.digest(audit.OUT / 'before_manifest.json'))
    with pytest.raises(ValueError):
        audit.audit_workflow(report, 'medium', baseline)


@pytest.mark.parametrize('bad', ['failure', 'error', 'skipped', 'incomplete'])
def test_final_xml_must_be_complete_and_passing(tmp_path, bad):
    import xml.etree.ElementTree as ET
    root = ET.Element('testsuites')
    suite = ET.SubElement(root, 'testsuite')
    for index in range(177 if bad == 'incomplete' else 178):
        case = ET.SubElement(suite, 'testcase', name=str(index))
        if index == 0 and bad != 'incomplete':
            ET.SubElement(case, bad)
    path = tmp_path / 'suite.xml'
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError):
        audit.audit_regression(path)
