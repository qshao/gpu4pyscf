"""Measure large exchange transforms when bounded streaming shortens blocks."""
import json
from pathlib import Path
import statistics
import time

import pytest


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('naux',[128,214,256])
def test_large_stream_exchange(naux):
    import cupy as cp
    from gpu4pyscf.df import df_jk
    from exchange_layout import transform
    from molecular import _metadata,_file_identity
    base=Path(__file__).parent
    output=base/('large_stream_exchange_'+str(naux)+'.json')
    if output.exists(): raise FileExistsError('Archive existing report explicitly')
    nao,nocc,capacity=1542,300,256
    storage=cp.arange(nao*nao*capacity,dtype=cp.float64).reshape(nao,nao,capacity)
    storage/=storage.size
    a=storage[:,:,:naux].transpose(2,0,1)
    b=cp.arange(nao*nocc,dtype=cp.float64).reshape(1,nao,nocc)
    b/=b.size
    old=cp.empty((1,nao,nocc,naux)); new=cp.empty_like(old)
    functions=dict(current=lambda:df_jk._exchange_transform(a,b,old),
                   strided=lambda:transform(a,b,new))
    for _ in range(2):
        for function in functions.values(): function()
    error=float(cp.max(cp.abs(old-new)))
    assert error<=1e-9,error
    samples={v:[] for v in functions}
    for repeat in range(7):
        for version in (['current','strided'] if repeat%2==0 else ['strided','current']):
            cp.cuda.Device().synchronize()
            begin,end=cp.cuda.Event(),cp.cuda.Event()
            wall=time.perf_counter(); begin.record()
            functions[version]()
            end.record(); end.synchronize()
            samples[version].append(dict(wall_seconds=time.perf_counter()-wall,
                                         event_seconds=cp.cuda.get_elapsed_time(begin,end)/1000))
    assert float(cp.max(cp.abs(old-new)))<=1e-9
    medians={v:statistics.median(x['wall_seconds'] for x in rows) for v,rows in samples.items()}
    report=dict(_metadata('gpu'),status='passed',nao=nao,nocc=nocc,naux=naux,
                benchmark_sha256=_file_identity(__file__)['sha256'],
                candidate_sha256=_file_identity(base/'exchange_layout.py')['sha256'],
                production_sha256=_file_identity(base.parent.parent/'gpu4pyscf/df/df_jk.py')['sha256'],
                max_error=error,samples=samples,medians=medians,
                component_ratio=medians['current']/medians['strided'],
                note='Component only, no concurrent CPU-reference jobs. Seven alternating samples after two warmups, padded256-row storage. naux214 models a large-system disk staging cap. naux256 is a same-algorithm control. No whole-workflow speed or reduced-RAM claim. Do not change production thresholds without molecular qualification.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(naux,medians,report['component_ratio'],flush=True)
