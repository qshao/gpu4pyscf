"""Diagnostic kernel comparison on an actual general-contraction layout."""
import ctypes
import inspect
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import pytest

from molecular import _metadata,_file_identity


@pytest.mark.special
@pytest.mark.benchmark
def test_component():
    import cupy as cp
    from pyscf import gto,scf,lib
    from gpu4pyscf.df import df,int3c2e_bdiv
    from recontract_gather import gather,csr_mapping
    base=Path(__file__).parent
    output=base/'recontract_gather_diagnostic.json'
    if output.exists(): raise FileExistsError('Archive existing report explicitly')
    geometry=base.parent/'molecules/organic/020_Vitamin_C.xyz'
    saved={}
    original=int3c2e_bdiv._create_pair_recontractor
    def observed(*args):
        function,counts,contracted,pairs=original(*args)
        params=inspect.getclosurevars(function).nonlocals['recontraction_params']
        def capture(batch,data,out=None):
            if not saved and contracted[batch]:
                inp,destination,coef=params[batch]
                saved.update(data=data.copy(),inp=cp.asarray(inp),
                             destination=cp.asarray(destination),coef=cp.asarray(coef),
                             rows=int(contracted[batch]),
                             mapping=csr_mapping(inp,destination,coef,int(contracted[batch])))
            return function(batch,data,out)
        return capture,counts,contracted,pairs
    assert cp.cuda.get_current_stream().ptr==0
    with lib.with_omp_threads(4):
        mol=gto.M(atom=str(geometry),basis='cc-pvdz',verbose=0)
        fitting=df.DF(mol,auxbasis='def2-universal-jkfit')
        fitting.use_gpu_memory=True
        dm=cp.asarray(scf.RHF(mol).get_init_guess())
        with patch.object(int3c2e_bdiv,'_create_pair_recontractor',observed): fitting.get_jk(dm)
        fitting.reset()
    assert saved
    data=saved['data']; mapping=saved['mapping']
    old=cp.empty((saved['rows'],data.shape[1])); new=cp.empty_like(old)
    def atomic():
        old.fill(0.)
        error=int3c2e_bdiv.libvhf_rys.recontract_ao_pair(
            ctypes.c_void_p(old.data.ptr),ctypes.c_void_p(data.data.ptr),
            ctypes.c_void_p(saved['destination'].data.ptr),ctypes.c_void_p(saved['inp'].data.ptr),
            ctypes.c_void_p(saved['coef'].data.ptr),ctypes.c_int(data.shape[1]),
            ctypes.c_int(len(saved['inp'])))
        assert error==0
    def coalesced(): gather(data,mapping,out=new)
    functions=dict(atomic=atomic,coalesced=coalesced)
    for _ in range(3): atomic(); coalesced()
    cp.testing.assert_allclose(new,old,atol=1e-10,rtol=0)
    deterministic=new.copy(); samples={v:[] for v in functions}
    for repeat in range(7):
        for version in (['atomic','coalesced'] if repeat%2==0 else ['coalesced','atomic']):
            cp.cuda.Device().synchronize()
            start,end=cp.cuda.Event(),cp.cuda.Event()
            wall=time.perf_counter(); start.record()
            for _ in range(3): functions[version]()
            end.record(); end.synchronize()
            samples[version].append(dict(wall_seconds=(time.perf_counter()-wall)/3,
                                         event_seconds=cp.cuda.get_elapsed_time(start,end)/3000))
    cp.testing.assert_allclose(new,old,atol=1e-10,rtol=0)
    assert bool(cp.array_equal(new,deterministic))
    medians={v:statistics.median(x['wall_seconds'] for x in rows) for v,rows in samples.items()}
    report=dict(_metadata('gpu'),status='passed',basis='cc-pvdz',case='vitamin_c',
        geometry_sha256=_file_identity(geometry)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        candidate_sha256=_file_identity(base/'recontract_gather.py')['sha256'],
        factory_sha256=_file_identity(base.parent.parent/'gpu4pyscf/df/int3c2e_bdiv.py')['sha256'],
        shape=list(data.shape),output_shape=list(old.shape),contributions=len(saved['inp']),
        candidate_mapping_bytes=sum(x.nbytes for x in mapping[:3]),
        baseline_device_mapping_bytes=sum(saved[k].nbytes for k in ['inp','destination','coef']),
        max_error=float(cp.max(cp.abs(new-old))),bitwise_repeatable=True,
        samples=samples,medians=medians,diagnostic_ratio=medians['atomic']/medians['coalesced'],
        note='DIAGNOSTIC ONLY: concurrent CPU-reference jobs on unified-memory GB10. Not final speed qualification. Actual first recontraction batch; resident metadata on both sides, setup/H2D excluded. Atomic zero-fill included, gather overwrites each output. Seven alternating rounds after three warmups; three calls per sample. Not full-workflow speed or whole-memory measurement. Candidate retains CSR metadata for all factory batches; record here covers this batch only.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['shape'],report['output_shape'],report['medians'],report['diagnostic_ratio'],flush=True)
