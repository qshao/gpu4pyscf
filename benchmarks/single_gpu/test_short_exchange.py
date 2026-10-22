"""Short exchange blocks with actual tight strides and padded controls."""
import statistics
import time

import numpy as np
import pytest

from corrected_optimization import BASE, OUT, digest, new_report, save


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('naux',[32,64,128])
@pytest.mark.parametrize('layout',['tight','padded'])
def test_short_exchange(naux,layout):
    import cupy as cp
    from gpu4pyscf.df import df_jk
    from exchange_layout import transform
    output,report=new_report(f'short_exchange_{naux}_{layout}.json',__file__)
    report.update(case='synthetic_exchange_1542',settings=None,
                  geometry_sha256=None,cpu_reference_sha256=None)
    save(output,report)
    nao,nocc=1542,300
    capacity=naux if layout=='tight' else 256
    storage=cp.arange(nao*nao*capacity,dtype=cp.float64).reshape(nao,nao,capacity)
    storage/=storage.size
    a=storage[:,:,:naux].transpose(2,0,1)
    b=cp.arange(nao*nocc,dtype=cp.float64).reshape(1,nao,nocc)/float(nao*nocc)
    old=cp.empty((1,nao,nocc,naux)); new=cp.empty_like(old)
    functions=dict(baseline=lambda:df_jk._exchange_transform(a,b,old),
                   strided=lambda:transform(a,b,new))
    for _ in range(2):
        for function in functions.values(): function()
    maximum=float(cp.max(cp.abs(old-new)))
    assert maximum<=1e-9,maximum
    rows=cp.asarray([0,nao//2,nao-1]); columns=cp.asarray([0,nocc//2,nocc-1])
    expected=np.einsum('Lij,jk->Lik',cp.asnumpy(a[:,rows,:]),cp.asnumpy(b[0][:,columns]))
    selected=cp.take(cp.take(new[0],rows,axis=0),columns,axis=1).transpose(2,0,1)
    np.testing.assert_allclose(cp.asnumpy(selected),expected,rtol=1e-12,atol=1e-10)
    samples={key:[] for key in functions}
    for repeat in range(7):
        for version in (['baseline','strided'] if repeat%2==0 else ['strided','baseline']):
            cp.cuda.Device().synchronize(); start=time.perf_counter()
            functions[version]()
            cp.cuda.Device().synchronize()
            samples[version].append(time.perf_counter()-start)
    medians={key:statistics.median(value) for key,value in samples.items()}
    report.update(nao=nao,nocc=nocc,naux=naux,capacity=capacity,layout=layout,
        candidate_sha256=digest(BASE/'exchange_layout.py'),samples=samples,medians=medians,
        maximum_difference=maximum,component_ratio=medians['baseline']/medians['strided'],status='passed',
        note='Component only; seven alternating samples after both warmups. Tight storage matches a short DF.loop allocation; padded256 is a separate layout control. Selected elements independently checked with NumPy. No CPU reference job or allocation instrumentation inside timings. Does not establish full molecular acceleration or qualify converged valinomycin forces.')
    save(output,report)
    print('short exchange',naux,layout,medians,report['component_ratio'],flush=True)
