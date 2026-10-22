"""CPU ledger checks and opt-in, small GPU allocator integration checks."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from phase_memory import _Ledger, phase_memory


def test_nested_scopes_include_retained_allocations_and_worker_threads():
    tracker = _Ledger()
    tracker.allocate(10)
    with tracker.phase('scf'):
        tracker.allocate(20)
        with tracker.phase('build'):
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(tracker.allocate, 7).result()
            tracker.release(20)
        tracker.allocate(5)
    with tracker.phase('gradient'):
        tracker.allocate(30)
        tracker.release(7)
    assert tracker.data['live_bytes'] == 45
    assert tracker.data['peak_live_bytes'] == 52
    assert tracker.data['allocation_count'] == 5
    assert tracker.data['phases']['scf'] == dict(
        entry_live_bytes=10, peak_live_bytes=37, allocation_count=3, entry_count=1)
    assert tracker.data['phases']['build'] == dict(
        entry_live_bytes=30, peak_live_bytes=37, allocation_count=1, entry_count=1)
    assert tracker.data['phases']['gradient'] == dict(
        entry_live_bytes=22, peak_live_bytes=52, allocation_count=1, entry_count=1)
    json.dumps(tracker.data)


def test_phase_exception_restores_enclosing_scope_and_reentry_aggregates():
    tracker = _Ledger()
    with tracker.phase('outer'):
        with pytest.raises(ValueError):
            with tracker.phase('inner'):
                tracker.allocate(8)
                raise ValueError('expected')
        tracker.allocate(3)
    with tracker.phase('inner'):
        tracker.allocate(1)
    assert tracker.data['phases']['inner'] == dict(
        entry_live_bytes=0, peak_live_bytes=12, allocation_count=2, entry_count=2)
    assert tracker.data['phases']['outer']['allocation_count'] == 2
    tracker.release(12)
    assert tracker.data['live_bytes'] == 0
    assert 'preexisting' in tracker.data['note']
    assert 'native' in tracker.data['note'].lower()


def test_same_named_nested_scope_counts_each_allocation_once():
    tracker = _Ledger()
    with tracker.phase('scf'):
        with tracker.phase('scf'):
            tracker.allocate(4)
    assert tracker.data['phases']['scf']['allocation_count'] == 1


def test_shared_accounting_serializes_concurrent_updates():
    tracker = _Ledger()
    barrier = threading.Barrier(8)

    def work():
        for _ in range(100):
            tracker.allocate(3)
        barrier.wait()
        for _ in range(100):
            tracker.release(3)

    with tracker.phase('worker'):
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda _: work(), range(8)))
    assert tracker.data['live_bytes'] == 0
    assert tracker.data['peak_live_bytes'] == 2400
    assert tracker.data['phases']['worker']['allocation_count'] == 800
    assert tracker.data['phases']['worker']['peak_live_bytes'] == 2400


def test_import_does_not_import_cupy():
    result = subprocess.run(
        [sys.executable, '-c',
         "import sys; import phase_memory; assert 'cupy' not in sys.modules"],
        cwd=Path(__file__).parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.special
@pytest.mark.parametrize('bypass_pool', [False, True])
def test_gpu_allocator_restores_and_retains_owners_across_phases(bypass_pool):
    import cupy as cp
    original = cp.cuda.get_allocator()
    allocator = cp.cuda.memory._malloc if bypass_pool else original
    cp.cuda.set_allocator(allocator)
    try:
        with phase_memory() as tracker:
            with tracker.phase('scf'):
                array = cp.empty(1024, dtype=cp.uint8)
                array.fill(19)
            with tracker.phase('gradient'):
                assert tracker.data['live_bytes'] == 1024
                assert tracker.data['phases']['gradient']['entry_live_bytes'] == 1024
                other = cp.empty(512, dtype=cp.uint8)
                other.fill(7)
                # Validate retained storage via a host copy: a GPU reduction
                # allocates its own temporary buffers and changes the peak.
                assert array.get().tolist() == [19] * 1024
                del array
                gc.collect()
                assert tracker.data['live_bytes'] == 512
            assert tracker.data['peak_live_bytes'] == 1536
        assert cp.cuda.get_allocator() is allocator
        assert int(other.sum()) == 7 * 512
        del other
        gc.collect()
        assert tracker.data['live_bytes'] == 0
    finally:
        cp.cuda.set_allocator(original)


@pytest.mark.special
def test_gpu_exception_restores_allocator_and_phase_with_worker_allocation():
    import cupy as cp
    original = cp.cuda.get_allocator()
    device = cp.cuda.Device().id
    with pytest.raises(RuntimeError, match='expected'):
        with phase_memory() as tracker:
            with tracker.phase('scf'):
                def allocate():
                    with cp.cuda.Device(device):
                        array = cp.empty(256, dtype=cp.uint8)
                        array.fill(11)
                        cp.cuda.Device().synchronize()
                        return array
                with ThreadPoolExecutor(max_workers=1) as executor:
                    array = executor.submit(allocate).result()
                assert tracker.data['phases']['scf']['allocation_count'] == 1
                assert tracker.data['live_bytes'] == 256
                raise RuntimeError('expected')
    assert cp.cuda.get_allocator() is original
    assert int(array.sum()) == 256 * 11
    tracker.allocate(1)
    assert tracker.data['phases']['scf']['allocation_count'] == 1
    tracker.release(1)
    del array
    gc.collect()
    assert tracker.data['live_bytes'] == 0
