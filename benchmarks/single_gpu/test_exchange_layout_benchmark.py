"""Current optimized GPU versus layout-aware exchange transformation."""
import json
from pathlib import Path
import statistics
import time

import cupy as cp
import pytest

import exchange_layout as candidate
from molecular import _metadata, _file_identity


@pytest.mark.special
@pytest.mark.benchmark
def test_layout_components():
    from gpu4pyscf.df import df_jk
    output = Path(__file__).with_name('exchange_layout_components.json')
    report = dict(_metadata('gpu'), status='running', cases=[],
                  candidate_sha256=_file_identity(candidate.__file__)['sha256'],
                  benchmark_sha256=_file_identity(__file__)['sha256'],
                  note='Actual DF strides including partial-block padding; shared GB10. Seven alternating synchronized wall-time samples after warmup. Direct strided GEMM uses no tensor-sized temporary. Other variants may allocate internally.')
    def save(): output.write_text(json.dumps(report,indent=2)+'\n')
    rng = cp.random.RandomState(71)
    for nao,nocc,naux,pitch in [(208,44,256,256),(208,78,66,256),
                               (512,100,256,256),(537,100,256,256),(537,169,256,256),
                               (537,100,66,256),(537,169,66,256)]:
        a = rng.normal(size=(nao,nao,pitch))[:,:,:naux].transpose(2,0,1)
        b = rng.normal(size=(1,nao,nocc))
        out = cp.empty((1,nao,nocc,naux))
        def matmul():
            cp.matmul(a.transpose(1,0,2),b[0],out=out[0].transpose(0,2,1))
            return out
        functions = dict(current=lambda:df_jk.contract('Lij,njk->nikL',a,b,out=out),
                         strided=lambda:candidate.transform(a,b,out), matmul=matmul)
        reference = functions['current']().copy()
        for f in functions.values():
            cp.testing.assert_allclose(f(),reference,rtol=2e-11,atol=1e-10)
        samples = {k:[] for k in functions}
        for repeat in range(7):
            keys = list(functions)
            if repeat%2: keys.reverse()
            for k in keys:
                cp.cuda.Device().synchronize(); start = time.perf_counter()
                functions[k]()
                cp.cuda.Device().synchronize()
                samples[k].append(time.perf_counter()-start)
        row = dict(nao=nao,nocc=nocc,naux=naux,pitch=pitch,
                   input_strides=list(a.strides),output_strides=list(out.strides),
                   timings={k:dict(samples_seconds=v,median_seconds=statistics.median(v)) for k,v in samples.items()})
        report['cases'].append(row); save()
        print(nao,nocc,naux,{k:round(v['median_seconds']*1000,3) for k,v in row['timings'].items()},flush=True)
        del a,b,out,reference
    report['status']='passed'; save()
