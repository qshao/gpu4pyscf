"""Scientific and import invariants for the original/current comparison."""
import pytest


@pytest.mark.parametrize('size,nao',[('small',208),('medium',537),('large',1123)])
def test_actual_scientific_factory(size,nao):
    from original_comparison import configure_cpu,case_settings
    from pyscf import lib
    from pyscf.dft.gen_grid import nwchem_prune
    with lib.with_omp_threads(4):
        mf=configure_cpu(size)
    config=case_settings(size)
    assert mf.mol.nao==nao and mf.grids.prune is nwchem_prune
    assert mf.grids.level==3 and mf.max_cycle==100
    assert mf.conv_tol==1e-10 and mf.conv_tol_grad==1e-7
    assert mf.small_rho_cutoff==0. and mf.with_df.auxbasis==config['auxbasis']


def test_unknown_comparison_size_rejected():
    from original_comparison import case_settings
    with pytest.raises(KeyError): case_settings('unknown')
