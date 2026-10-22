"""Microbenchmark of the DF exchange (K) contractions at the real Taxol shapes, on one GPU.

Compares, per auxiliary block of nL functions:
  cuTENSOR   contract('Lij,njk->nikL')  and  contract('nikL,njkL->nij')    (what runs when cuTENSOR is installed)
  GEMM path  _exchange_transform (strided cuBLAS DGEMM) and _symmetric_gram (cuBLAS SYRK)   (the CuPy-engine fast paths)
and reports achieved TFLOPS against the card's cuBLAS DGEMM ceiling (a large square DGEMM).
Usage: python lcc_gemm_bench.py GPU
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, '/scratch/qsh226/g4p_lcc/current')
import cupy as cp
from gpu4pyscf.df import df_jk
from gpu4pyscf.lib.cutensor import contract

NAO, NOCC, NAUX = 1123, 226, 5598   # Taxol / def2-SVP / def2-universal-jkfit, B3LYP


def timed(fn, repeat=5):
    fn(); cp.cuda.Device().synchronize()
    samples = []
    for _ in range(repeat):
        cp.cuda.Device().synchronize(); t = time.perf_counter()
        fn(); cp.cuda.Device().synchronize()
        samples.append(time.perf_counter() - t)
    return statistics.median(samples)


def main(gpu):
    name = cp.cuda.runtime.getDeviceProperties(0)['name']
    result = dict(gpu=gpu, device=name.decode() if isinstance(name, bytes) else name, nao=NAO, nocc=NOCC, naux=NAUX,
                  free_bytes=int(cp.cuda.runtime.memGetInfo()[0]), blocks=[])
    n = 8192
    a = cp.random.rand(n, n); b = cp.random.rand(n, n)
    t = timed(lambda: cp.matmul(a, b), 3)
    result['dgemm_ceiling_tflops'] = 2 * n**3 / t / 1e12
    del a, b
    rng = cp.random.default_rng(0)
    c = rng.random((1, NAO, NOCC))
    for nL in (128, 256, 512, 1024):
        need = 8 * (nL * NAO * NAO + NAO * NOCC * nL * 2 + NAO * NAO)
        if need > 0.8 * cp.cuda.runtime.memGetInfo()[0]:
            result['blocks'].append(dict(nL=nL, skipped='does not fit')); continue
        cderi = rng.random((nL, NAO, NAO))
        rhok = cp.empty((1, NAO, NOCC, nL)); vk = cp.zeros((1, NAO, NAO))
        flops_transform = 2 * nL * NAO * NAO * NOCC
        flops_gram = 2 * NAO * NAO * NOCC * nL     # general GEMM count; SYRK does half
        row = dict(nL=nL)
        row['transform_cutensor_s'] = timed(lambda: contract('Lij,njk->nikL', cderi, c, out=rhok))
        ref = rhok.copy()
        df_jk.contract_engine = 'cupy'                 # enable the GEMM fast path for this call only
        row['transform_gemm_s'] = timed(lambda: df_jk._exchange_transform(cderi, c, rhok))
        row['transform_agree'] = float(cp.abs(ref - rhok).max())
        row['gram_cutensor_s'] = timed(lambda: contract('nikL,njkL->nij', rhok, rhok, beta=1, out=vk))
        row['gram_syrk_s'] = timed(lambda: df_jk._symmetric_gram(rhok.reshape(1, NAO, -1), vk))
        df_jk.contract_engine = None
        for k, f in (('transform_cutensor', flops_transform), ('transform_gemm', flops_transform),
                     ('gram_cutensor', flops_gram), ('gram_syrk', flops_gram)):
            row[k + '_tflops'] = f / row[k + '_s'] / 1e12
        blocks = NAUX / nL
        row['K_per_cycle_cutensor_s'] = blocks * (row['transform_cutensor_s'] + row['gram_cutensor_s'])
        row['K_per_cycle_gemm_s'] = blocks * (row['transform_gemm_s'] + row['gram_syrk_s'])
        result['blocks'].append(row)
        del cderi, rhok, vk; cp.get_default_memory_pool().free_all_blocks()
    out = Path(__file__).resolve().parent / 'lcc_gemm_results'
    out.mkdir(exist_ok=True)
    (out / (gpu + '.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
