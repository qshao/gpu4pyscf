"""Scientific configuration must match across CPU and GPU benchmark modes."""
import pytest


def test_comprehensive_matrix():
    from comprehensive import CASES, settings
    assert len(CASES)==8
    for name in CASES:
        s=settings(name)
        assert s['grid_level']==3 and s['grid_pruning']=='nwchem'
        assert s['conv_tol']==1e-10 and s['conv_tol_grad']==1e-7
        assert s['auxbasis_response'] and s['grid_response']
    with pytest.raises(KeyError): settings('unknown')


def test_configured_cpu_grid():
    from comprehensive import configured
    import molecular
    with configured('vitamin_svp_b3lyp') as settings:
        mf,dm0=molecular._mean_field('B3LYP','cpu')
        assert mf.grids.level==3
        assert mf.grids.prune.__name__=='nwchem_prune'
        assert mf.max_cycle==100
        assert dm0.shape==(208,208)


def test_reject_invalid_mode_before_running():
    from comprehensive import execute
    with pytest.raises(ValueError): execute('vitamin_svp_b3lyp','invalid')


def test_existing_report_requires_explicit_archive(tmp_path,monkeypatch):
    import json
    import comprehensive as driver
    monkeypatch.setattr(driver,'RESULTS',tmp_path)
    case='vitamin_svp_pbe'
    report=dict(status='passed',settings=driver.settings(case),
                driver_sha256=driver._file_identity(driver.__file__)['sha256'],
                geometry_sha256='stale-geometry')
    (tmp_path/(case+'_cpu.json')).write_text(json.dumps(report))
    with pytest.raises(FileExistsError):
        driver.execute(case,'cpu')
