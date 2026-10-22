"""Call each memory case in a fresh process; timing runs avoid CPU contention."""
import pytest
from comprehensive import CASES,execute


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('case',list(CASES))
@pytest.mark.parametrize('mode',['cpu','gpu','profile','memory'])
def test_case(case,mode):
    print('comprehensive',case,mode,flush=True)
    report=execute(case,mode)
    result=report['result']
    print(case,mode,'passed',result['iterations'],'iterations',
          result['scf_seconds'],'SCF seconds',result['gradient_seconds'],
          'gradient seconds',flush=True)
