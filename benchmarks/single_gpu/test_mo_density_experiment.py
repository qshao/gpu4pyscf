"""Reversible GGA density experiment; production implementation is untouched."""
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import cupy as cp
import numpy as np
import pytest

from df_profile import ROOT
from molecular import SETTINGS, _file_identity, _metadata, compare_cases, run_case


def batched_gga(original, fused=True):
    from gpu4pyscf.dft import numint
    from gpu4pyscf.lib.cupy_helper import ndarray

    def evaluate(ao, cpos, xctype, with_lapl=False, buf=None, rho=None):
        if xctype not in ('GGA', 'NLC') or ao.dtype != cp.float64 or cpos.dtype != cp.float64:
            return original(ao, cpos, xctype, with_lapl, buf=buf, rho=rho)
        ngrids, nmo = ao.shape[-1], cpos.shape[1]
        required = 4*nmo*ngrids*8
        if buf is not None and buf.nbytes < required:
            buf = None
        c = ndarray((4, nmo, ngrids), buffer=buf)
        cp.matmul(cpos.T, ao[:4], out=c)
        if fused:
            return numint._contract_rho_gga(c, c, rho=rho)
        rho = ndarray((4, ngrids), buffer=rho)
        for i in range(4):
            numint._contract_rho(c[0], c[i], rho=rho[i])
        rho[1:] *= 2
        return rho
    return evaluate


@pytest.mark.special
@pytest.mark.benchmark
def test_density_components():
    from gpu4pyscf.dft import numint
    original = numint._eval_rho2
    rng = np.random.default_rng(102)
    report = dict(_metadata('gpu'), status='running', cases=[],
                  experiment_sha256=_file_identity(__file__)['sha256'])
    output = Path(__file__).with_name('mo_density_component_results.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    # Small/large AO and occupied spaces; odd grid size and strided AO included.
    for nao, nocc, ngrids, strided, ncomp in [(208, 44, 4096, False, 4), (537, 104, 4096, False, 4),
                                      (113, 104, 1023, True, 4), (16, 0, 37, False, 4),
                                      (24, 8, 97, False, 10)]:
        ao = cp.asarray(rng.normal(size=(ncomp, nao, ngrids*(2 if strided else 1))))
        if strided:
            ao = ao[:, :, ::2]
        cpos = cp.asarray(rng.normal(size=(nao, nocc)))
        ref = original(ao, cpos, 'GGA')
        row = dict(nao=nao, nocc=nocc, ngrids=ngrids, strided=strided, ncomp=ncomp, variants={})
        report['cases'].append(row)
        for label, evaluate in [('original', original), ('batched', batched_gga(original, False)),
                                ('batched_fused', batched_gga(original))]:
            # Both sufficient and insufficient caller scratch buffers, plus output reuse.
            for buf_size in (2, 4):
                buf = cp.empty((buf_size, nocc, ngrids))
                out = cp.empty((4, ngrids))
                actual = evaluate(ao, cpos, 'GGA', buf=buf, rho=out)
                assert actual is out or actual.data.ptr == out.data.ptr
                cp.testing.assert_allclose(actual, ref, rtol=2e-12, atol=2e-11)
            samples=[]
            buf = cp.empty((4, nocc, ngrids))
            for _ in range(9):
                cp.cuda.Device().synchronize()
                start=time.perf_counter()
                evaluate(ao, cpos, 'GGA', buf=buf)
                cp.cuda.Device().synchronize()
                samples.append(time.perf_counter()-start)
            row['variants'][label]=dict(samples_seconds=samples, median_seconds=statistics.median(samples))
        print('density', nao, nocc, ngrids, {k:round(v['median_seconds']*1000,3)
              for k,v in row['variants'].items()}, flush=True)
    report['status']='passed'
    output.write_text(json.dumps(report, indent=2)+'\n')


@pytest.mark.special
@pytest.mark.benchmark
def test_density_workflows():
    from gpu4pyscf.dft import numint
    original = numint._eval_rho2
    reference_path=Path(__file__).with_name('stage_profile_results.json')
    references=json.loads(reference_path.read_text())
    assert references['status']=='passed'
    report=dict(_metadata('gpu'), status='running', cases=[],
                experiment_sha256=_file_identity(__file__)['sha256'],
                reference_sha256=_file_identity(reference_path)['sha256'],
                note='Shared single GB10. Benchmark-only GGA batched/fused experiment. Five alternating paired runs after both warmups, same build/settings; CPU gates on every run. Per-iteration synchronization retained. No production edits.')
    output=Path(__file__).with_name('mo_density_workflow_results.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    for case, method in [('vitamin_svp','PBE'), ('tamoxifen_svp','PBE')]:
        ref=next(c for c in references['cases'] if c['case']==case and c['method']==method)
        cpu=ref['cpu_reference']
        for key in ('pyscf', 'numpy', 'cpu_libxc'):
            assert cpu[key]==report[key],key
        assert ref['geometry_sha256']==_file_identity(cpu['settings']['atom'])['sha256']
        assert ref['instrumented']['source_hashes']==report['source_hashes']
        assert ref['instrumented']['native_libraries']==report['native_libraries']
        row=dict(case=case, method=method, pairs=[], warmups=[], status='running')
        report['cases'].append(row)
        with patch.dict(SETTINGS, cpu['settings']):
            for label, implementation in [('original',original),('candidate',batched_gga(original))]:
                print(case, label, 'warmup', flush=True)
                with patch.object(numint, '_eval_rho2', implementation):
                    result=run_case(method,'gpu')
                comparison=compare_cases(cpu,result)
                assert comparison['passed'],comparison
                row['warmups'].append(dict(variant=label,result=result,comparison=comparison))
            for repeat in range(5):
                pair={}
                order=['original','candidate'] if repeat%2==0 else ['candidate','original']
                for label in order:
                    implementation=original if label=='original' else batched_gga(original)
                    with patch.object(numint, '_eval_rho2', implementation):
                        result=run_case(method,'gpu')
                    comparison=compare_cases(cpu,result)
                    assert comparison['passed'],comparison
                    pair[label]=dict(result=result,comparison=comparison)
                assert pair['original']['result']['iterations']==pair['candidate']['result']['iterations']
                row['pairs'].append(pair)
                output.write_text(json.dumps(report,indent=2)+'\n')
                print(case, 'pair',repeat+1,'passed',flush=True)
            for phase in ['scf','gradient']:
                row[phase+'_medians']={label:statistics.median(p[label]['result'][phase+'_seconds']
                    for p in row['pairs']) for label in ('original','candidate')}
            row['combined_medians']={label:statistics.median(p[label]['result']['scf_seconds']+
                p[label]['result']['gradient_seconds'] for p in row['pairs'])
                for label in ('original','candidate')}
            row['status']='passed'
        assert numint._eval_rho2 is original
        output.write_text(json.dumps(report,indent=2)+'\n')
    report['status']='passed'
    output.write_text(json.dumps(report,indent=2)+'\n')
