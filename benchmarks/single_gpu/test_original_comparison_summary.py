"""Independent audit must reject altered references, settings and observations."""
import copy
import json

import pytest


@pytest.mark.parametrize('corruption',[None,'reference','native','force','comparison','pruning','time'])
def test_comparison_audit_rejects_corruption(tmp_path,monkeypatch,corruption):
    import summarize_original_comparison as validator
    from original_comparison import manifest,OUT
    frozen=manifest()
    (tmp_path/'results').mkdir()
    (tmp_path/'manifest.json').write_bytes((OUT/'manifest.json').read_bytes())
    name='small_original_time_0.json'
    report=json.loads((OUT/'results'/name).read_text())
    report=copy.deepcopy(report)
    if corruption=='reference': report['cpu_reference_sha256']='forged'
    if corruption=='native': report['native_libraries']['libxc']['sha256']='forged'
    if corruption=='force': report['result']['gradient_hartree_per_bohr'][0][0]+=.001
    if corruption=='comparison': report['comparison']['energy_error_hartree']=0.
    if corruption=='pruning': report['actual_factory_settings']['grid_pruning']=None
    if corruption=='time': report['workflow_seconds']=-1.
    (tmp_path/'results'/name).write_text(json.dumps(report))
    monkeypatch.setattr(validator,'OUT',tmp_path)
    if corruption:
        with pytest.raises(AssertionError): validator.load('small','original','time',0,frozen)
    else:
        assert validator.load('small','original','time',0,frozen)['status']=='passed'
