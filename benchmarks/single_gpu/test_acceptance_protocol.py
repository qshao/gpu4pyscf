"""Acceptance must execute the actual scientific factory configuration."""
from types import SimpleNamespace
import numpy as np
import pytest


def test_acceptance_factory_keeps_grid_pruning(monkeypatch,tmp_path):
    import molecular
    import test_corrected_acceptance as driver
    from gpu4pyscf.dft.gen_grid import nwchem_prune
    class Inspected(Exception): pass
    def factory(method,backend):
        return SimpleNamespace(grids=SimpleNamespace(level=3,prune=None),
            max_cycle=50,small_rho_cutoff=0.,conv_tol=1e-10,conv_tol_grad=1e-7,
            with_df=SimpleNamespace()),None
    def inspect(method,backend):
        mf,_=molecular._mean_field(method,backend)
        assert mf.grids.prune is nwchem_prune
        assert mf.max_cycle==100
        raise Inspected
    monkeypatch.setattr(molecular,'_mean_field',factory)
    monkeypatch.setattr(molecular,'run_case',inspect)
    monkeypatch.setattr(driver,'seed_density',lambda:(np.eye(1),{},'seed'))
    monkeypatch.setattr(driver,'oracle',lambda:(None,{'result':{}}))
    monkeypatch.setattr(driver,'new_report',lambda *args:(tmp_path/'report.json',{}))
    monkeypatch.setattr(driver,'digest',lambda *args:'identity')
    monkeypatch.setattr(driver,'save',lambda *args:None)
    with pytest.raises(Inspected):
        driver.test_corrected_optimization_acceptance('resident',tmp_path)
