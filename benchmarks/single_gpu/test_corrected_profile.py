"""Current-code full SCF/gradient profile with exchange-shape observations."""
import time
import traceback
from unittest.mock import patch

import pytest

from corrected_optimization import BASE, CASE, OUT, new_report, seed_density, oracle, digest, save


@pytest.mark.special
@pytest.mark.benchmark
def test_corrected_profile():
    import cupy as cp
    import molecular
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.df.grad import rhf
    from gpu4pyscf.grad import rhf as grad_rhf
    from comprehensive import configured
    from molecular import run_case,_file_identity
    from gradient_snapshot import write_orbitals
    from seeded_reference import compare_execution_cases
    from test_stage_profile import stage_profile
    from test_fine_stage_profile import fine_profile
    output,report=new_report('corrected_profile.json',__file__)
    density,seed,parent_sha=seed_density()
    report.update(seed=seed,seed_qualification_sha256=parent_sha,exchange_shapes={},metric_calls=[],
        helper_hashes={p:digest(BASE/p) for p in ['gradient_snapshot.py','test_stage_profile.py','test_fine_stage_profile.py']},
        note='Corrected production baseline. Full SCF/full-response forces at original scientific settings, initial density from previously converged orbitals only. Inclusive synchronized diagnostic scopes; do not sum parents/children. Orbital snapshot saved before gradient; I/O is not a performance control. Exchange shape counts include initial setup and three converged-state iterations; no whole-workflow gain inferred.')
    original_transform=df_jk._exchange_transform
    def transform(a,b,out):
        key=str((list(a.shape),list(b.shape),int(a.strides[2]//8)))
        report['exchange_shapes'][key]=report['exchange_shapes'].get(key,0)+1
        return original_transform(a,b,out)
    original_metric=rhf._apply_auxiliary_metric
    def metric(*args,**kwargs):
        cp.cuda.Device().synchronize(); start=time.perf_counter()
        value=original_metric(*args,**kwargs)
        cp.cuda.Device().synchronize()
        report['metric_calls'].append(dict(seconds=time.perf_counter()-start,rhs_shape=list(args[2].shape)))
        return value
    original_kernel=grad_rhf.GradientsBase.kernel
    def kernel(grad,*args,**kwargs):
        if 'orbitals' not in report:
            snapshot=OUT/'corrected_profile.npz'
            write_orbitals(grad.base,snapshot); report['orbitals']=_file_identity(snapshot)
        return original_kernel(grad,*args,**kwargs)
    try:
        with configured(CASE):
            factory=molecular._mean_field
            def initialize(method,backend):
                mf,_=factory(method,backend)
                return mf,density.copy()
            with patch.object(molecular,'_mean_field',initialize), \
                 patch.object(df_jk,'_exchange_transform',transform), \
                 patch.object(rhf,'_apply_auxiliary_metric',metric), \
                 patch.object(grad_rhf.GradientsBase,'kernel',kernel), \
                 stage_profile() as coarse,fine_profile() as fine:
                result=run_case('B3LYP','gpu')
        comparison=compare_execution_cases(oracle()[1]['result'],result)
        report.update(result=result,comparison=comparison,coarse=coarse,fine=fine)
        assert comparison['passed'] and fine['bindings_restored']
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    save(output,report)
    print('Corrected profile',comparison,coarse,report['metric_calls'],report['exchange_shapes'],flush=True)
