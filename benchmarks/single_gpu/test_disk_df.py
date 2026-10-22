"""Disk integral storage, transfer lifetime and scientific regression checks."""
import numpy as np
import pytest


def test_disk_round_trip_and_cleanup(tmp_path):
    from gpu4pyscf.df.disk import DiskTensor
    source = np.arange(11*17, dtype=np.float64).reshape(11, 17) / 7
    store = DiskTensor(source.shape, directory=tmp_path)
    path = store.path
    with pytest.raises(RuntimeError, match='complete'):
        store.read_rows(0, 2, np.empty((2, 17)))
    for p0, p1 in [(0, 5), (5, 13), (13, 17)]:
        store.write_columns(p0, p1, source[:, p0:p1].copy())
    store.finish()
    for p0, p1 in [(0, 4), (4, 8), (8, 11)]:
        out = np.empty((p1-p0, 17))
        assert store.read_rows(p0, p1, out) is out
        np.testing.assert_array_equal(out, source[p0:p1])
    assert store.nbytes == source.nbytes
    store.close()
    assert not path.exists()
    store.close()


def test_disk_rejects_incomplete_or_invalid_blocks(tmp_path):
    from gpu4pyscf.df.disk import DiskTensor
    store = DiskTensor((4, 9), directory=tmp_path)
    try:
        with pytest.raises(ValueError):
            store.write_columns(1, 3, np.zeros((4, 2)))
        with pytest.raises(ValueError):
            store.write_columns(0, 3, np.zeros((4, 3), dtype=np.float32))
        with pytest.raises(RuntimeError, match='complete'):
            store.finish()
        store.write_columns(0, 9, np.ones((4, 9)))
        store.finish()
        with pytest.raises(ValueError):
            store.read_rows(0, 2, np.empty((2, 10)))
        with pytest.raises(ValueError):
            store.read_rows(-1, 2, np.empty((3, 9)))
    finally:
        store.close()


@pytest.mark.special
@pytest.mark.parametrize('basis', ['def2-svp', 'cc-pvdz'])
@pytest.mark.parametrize('method,spin', [('B3LYP', 0), ('WB97X', 0), ('B3LYP', 1)])
def test_disk_scf_gradient_and_ao2mo(tmp_path, basis, method, spin):
    import cupy as cp
    from pyscf import gto, dft
    from gpu4pyscf import dft as gdft
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                basis=basis, charge=spin, spin=spin, verbose=0)
    cpu = (dft.UKS(mol) if spin else dft.RKS(mol)).density_fit(auxbasis='weigend')
    gpu = (gdft.UKS(mol) if spin else gdft.RKS(mol)).density_fit(auxbasis='weigend')
    for mf in [cpu, gpu]:
        mf.xc = method
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-7
        mf.max_cycle = 100
        mf.grids.level = 1
        mf.grids.prune = None
    gpu.with_df.cderi_storage = 'disk'
    gpu.with_df.cderi_directory = tmp_path
    gpu.with_df.cderi_cache_bytes = 32*1024
    try:
        a, b = cpu.kernel(), gpu.kernel()
        assert cpu.converged and gpu.converged
        assert abs(a-b) < 1e-8
        ga, gb = cpu.nuc_grad_method(), gpu.nuc_grad_method()
        ga.grid_response = gb.grid_response = True
        np.testing.assert_allclose(cp.asnumpy(gb.kernel()), ga.kernel(), atol=1e-6, rtol=0)
        coeff = cp.asarray(cpu.mo_coeff[0] if spin else cpu.mo_coeff)[:, :3]
        expected = cpu.with_df.ao2mo(cp.asnumpy(coeff))
        actual = gpu.with_df.ao2mo(coeff)
        np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-9, rtol=0)
    finally:
        gpu.with_df.reset()
        assert not list(tmp_path.glob('gpu4pyscf-cderi-*.bin'))


@pytest.mark.special
@pytest.mark.parametrize('basis', ['def2-svp', 'cc-pvdz'])
def test_disk_failed_build_cleanup_and_minimum_budget(tmp_path, monkeypatch, basis):
    from pyscf import gto
    from gpu4pyscf.df.df import DF
    from gpu4pyscf.df.disk import DiskTensor
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis=basis, verbose=0)
    obj = DF(mol, 'weigend')
    obj.cderi_storage = 'disk'
    obj.cderi_directory = tmp_path
    obj.cderi_buffer_bytes = 8
    with pytest.raises(MemoryError, match='Disk CDERI .*staging'):
        obj.build()
    assert not list(tmp_path.iterdir())
    obj.cderi_buffer_bytes = 4*1024**2
    def fail(*args):
        raise OSError('injected disk full')
    monkeypatch.setattr(DiskTensor, 'write_columns', fail)
    with pytest.raises(OSError, match='injected disk full'):
        obj.build()
    assert obj._cderi is None
    assert not list(tmp_path.iterdir())


@pytest.mark.special
def test_disk_cache_avoids_repeat_transfers(tmp_path):
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf.df.df import DF
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis='def2-svp', verbose=0)
    obj = DF(mol, 'weigend')
    obj.cderi_storage = 'disk'
    obj.cderi_directory = tmp_path
    obj.cderi_cache_bytes = 32*1024
    obj.build()
    store = obj._cderi[0]
    try:
        assert 0 < store.cache.nbytes <= obj.cderi_cache_bytes
        assert store.stats['h2d_bytes'] == 0
        initial = dict(store.stats)
        for unused in range(2):
            rows = sum(block.shape[0] for _, block in obj.loop(9, unpack=False))
            assert rows == store.shape[0]
        transferred = store.stats['h2d_bytes']-initial['h2d_bytes']
        used = store.cache.shape[0]//9*9*store.shape[1]*8
        assert transferred == 2*(store.nbytes-used)
        assert store.stats['cache_hit_bytes'] == 2*used
        out = np.empty(store.cache.shape)
        store.read_rows(0, len(out), out)
        cp.testing.assert_array_equal(store.cache, cp.asarray(out))
    finally:
        store.close()


def test_disk_handles_short_io(tmp_path, monkeypatch):
    from gpu4pyscf.df import disk
    source = np.arange(35, dtype=np.float64).reshape(5, 7)
    write, read = disk.os.pwrite, disk.os.preadv
    monkeypatch.setattr(disk.os, 'pwrite', lambda fd, data, offset: write(fd, data[:13], offset))
    monkeypatch.setattr(disk.os, 'preadv', lambda fd, buffers, offset: read(fd, [buffers[0][:19]], offset))
    store = disk.DiskTensor(source.shape, directory=tmp_path)
    try:
        store.write_columns(0, 7, source)
        store.finish()
        out = np.empty_like(source)
        store.read_rows(0, 5, out)
        np.testing.assert_array_equal(out, source)
    finally:
        store.close()


@pytest.mark.special
@pytest.mark.parametrize('cached_rows', [0, 5, 23])
def test_disk_nondefault_stream_queued_consumers(tmp_path, cached_rows):
    import cupy as cp
    from gpu4pyscf.df.disk import DiskTensor, device_blocks
    source = np.random.default_rng(17).normal(size=(23, 257))
    store = DiskTensor(source.shape, directory=tmp_path)
    try:
        store.write_columns(0, source.shape[1], source)
        store.finish()
        with cp.cuda.Stream(non_blocking=True):
            if cached_rows:
                store.cache = cp.asarray(source[:cached_rows])
            outputs = []
            for p0, p1, block in device_blocks(store, 4):
                # No host synchronization between reuse of staging slots.
                outputs.append(cp.sum(block*block, axis=1))
            actual = cp.asnumpy(cp.concatenate(outputs))
        np.testing.assert_allclose(actual, np.sum(source*source, axis=1), atol=1e-11, rtol=0)
        used = cached_rows if cached_rows == len(source) else cached_rows//4*4
        assert store.stats['h2d_bytes'] == source[used:].nbytes
    finally:
        store.close()


@pytest.mark.special
def test_cache_preserves_contraction_block_boundaries(tmp_path):
    import cupy as cp
    from gpu4pyscf.df.disk import DiskTensor, device_blocks
    source = np.arange(23*17,dtype=np.float64).reshape(23,17)
    store = DiskTensor(source.shape,directory=tmp_path)
    try:
        store.write_columns(0,17,source)
        store.finish()
        store.cache=cp.asarray(source[:5])
        boundaries=[(p0,p1) for p0,p1,block in device_blocks(store,4)]
        assert boundaries==[(p0,min(p0+4,23)) for p0 in range(0,23,4)]
    finally:
        store.close()


@pytest.mark.special
def test_disk_hf_hessian(tmp_path):
    import gc
    from unittest.mock import patch
    from gpu4pyscf.df.df import DF
    from molecular import run_case, compare_cases
    cpu = run_case('HF', 'cpu', hessian=True)
    with patch.object(DF, 'cderi_storage', 'disk'), \
         patch.object(DF, 'cderi_directory', tmp_path):
        gpu = run_case('HF', 'gpu', hessian=True)
    assert compare_cases(cpu, gpu)['passed']
    gc.collect()
    assert not list(tmp_path.glob('gpu4pyscf-cderi-*.bin'))


@pytest.mark.special
def test_disk_build_jk_and_partial_blocks(tmp_path, monkeypatch):
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf.df import df
    from gpu4pyscf.df.disk import DiskTensor
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                basis='def2-svp', verbose=0)
    resident = df.DF(mol, 'weigend').build()
    disk = df.DF(mol, 'weigend')
    disk.cderi_storage = 'disk'
    disk.cderi_directory = tmp_path
    disk.cderi_buffer_bytes = 4*1024**2
    monkeypatch.setattr(df, 'empty_mapped', lambda *a, **k: pytest.fail('full host allocation'))
    disk.build()
    store = disk._cderi[0]
    try:
        assert isinstance(store, DiskTensor)
        for unpack in [False, True]:
            expected = [packed for _, packed in resident.loop(blksize=7, unpack=False)]
            for (dense, packed), ref_packed in zip(disk.loop(7, unpack), expected, strict=True):
                if unpack:
                    rows = cp.asarray(disk._cderi_idx[0]) // mol.nao
                    cols = cp.asarray(disk._cderi_idx[0]) % mol.nao
                    cp.testing.assert_allclose(dense[:, rows, cols], packed, atol=1e-12)
                cp.testing.assert_allclose(packed, ref_packed, atol=1e-12)
        dm = cp.asarray(np.random.default_rng(4).normal(size=(mol.nao, mol.nao)))
        dm = dm + dm.T
        for hermi in [0, 1]:
            got = disk.get_jk(dm, hermi=hermi)
            expected = resident.get_jk(dm, hermi=hermi)
            for a, b in zip(got, expected):
                cp.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
        iterator = disk.loop(7, unpack=False)
        next(iterator)
        iterator.close()
        assert store.stats['peak_host_buffer_bytes'] <= disk.cderi_buffer_bytes
        assert store.stats['peak_device_buffer_bytes'] <= disk.cderi_buffer_bytes
        assert store.stats['h2d_bytes'] > 0
    finally:
        store.close()
