"""Run each capacity node in a fresh process; baselines precede candidates."""
import pytest
from capacity import LARGE_CASES,execute


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('case',LARGE_CASES)
@pytest.mark.parametrize('variant',['baseline','packed'])
def test_capacity(case,variant):
    print('capacity',case,variant,flush=True)
    report=execute(case,variant)
    result=report['result']
    print(case,variant,report['status'],result['iterations'],
          result['scf_seconds'],result['gradient_seconds'],flush=True)


@pytest.mark.special
@pytest.mark.benchmark
def test_taxol_combined_disk_packed():
    report=execute('taxol_svp_b3lyp','packed',storage='disk',reference_storage='resident')
    result=report['result']
    print('taxol combined disk/packed',report['status'],result['iterations'],
          result['scf_seconds'],result['gradient_seconds'],flush=True)
