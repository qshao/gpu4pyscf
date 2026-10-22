"""Actual-stride components; independent of memory-instrumented timings."""
import json
from pathlib import Path
import statistics
import time

import cupy as cp
import pytest

import speed_memory_candidates as candidate
from molecular import _metadata,_file_identity


@pytest.mark.special
@pytest.mark.benchmark
def test_components():
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.df.grad import rhf
    output=Path(__file__).with_name('speed_memory_components.json')
    report=dict(_metadata('gpu'),status='running',cases=[],
        candidate_sha256=_file_identity(candidate.__file__)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'])
    rng=cp.random.RandomState(92)
    def measure(label,pattern,a,b,out,original,proposed,beta):
        functions=dict(current=lambda:original(pattern,a,b,alpha=.3,beta=beta,out=out),
                       candidate=lambda:proposed(pattern,a,b,out,.3,beta))
        out.fill(0); ref=functions['current']().copy(); out.fill(0)
        cp.testing.assert_allclose(functions['candidate'](),ref,rtol=2e-11,atol=1e-9)
        for f in functions.values(): f()
        samples={k:[] for k in functions}
        for repeat in range(7):
            keys=list(functions)
            if repeat%2: keys.reverse()
            for k in keys:
                cp.cuda.Device().synchronize(); start=time.perf_counter()
                functions[k]()
                cp.cuda.Device().synchronize()
                samples[k].append(time.perf_counter()-start)
        row=dict(component=label,pattern=pattern,a_shape=list(a.shape),a_strides=list(a.strides),
                 timings={k:dict(samples_seconds=v,median_seconds=statistics.median(v)) for k,v in samples.items()})
        report['cases'].append(row)
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(label,pattern,list(a.shape),{k:round(v['median_seconds']*1000,3) for k,v in row['timings'].items()},flush=True)
    for layout in ['C','padded_F']:
        storage=cp.empty((92642,2626)) if layout=='padded_F' else None
        for naux in [256,66]:
            b=storage.T[:naux] if storage is not None else cp.empty((naux,92642))
            b[:]=rng.normal(size=b.shape)
            for pattern in ['np,Lp->nL','nL,Lp->np']:
                a=rng.normal(size=(1,92642 if pattern.startswith('np') else naux))
                out=cp.empty((1,naux if pattern.startswith('np') else 92642))
                measure('coulomb_'+layout,pattern,a,b,out,df_jk.contract,candidate.coulomb,1.)
            del a,b,out
        del storage
    for naux in [28,281,1008]:
        b=rng.normal(size=(565,100))
        for pattern in ['pqr,pi->iqr','iqr,pi->pqr']:
            rows=565 if pattern.startswith('pqr') else 100
            capacity=3080 if pattern.startswith('pqr') else naux
            raw=cp.empty((rows,565,capacity))
            a=raw[:,:,:naux]; a[:]=rng.normal(size=a.shape)
            out=cp.empty((100 if rows==565 else 565,565,naux))
            measure('gradient',pattern,a,b,out,rhf.contract,candidate.gradient,0. if rows==565 else 1.)
            del raw,a,out
        del b
    report['status']='passed'; output.write_text(json.dumps(report,indent=2)+'\n')
