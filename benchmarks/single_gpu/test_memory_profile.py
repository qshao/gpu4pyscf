import cupy as cp
from concurrent.futures import ThreadPoolExecutor
import memory_profile


def test_pool_peak_includes_worker_allocations():
    assert hasattr(memory_profile,'peak_memory'), 'memory instrumentation missing'
    live=[]
    baseline=cp.get_default_memory_pool().used_bytes()
    with memory_profile.peak_memory() as result:
        live.append(cp.empty(1024*1024,dtype=cp.uint8))
        with ThreadPoolExecutor(max_workers=1) as executor:
            live.append(executor.submit(lambda:cp.empty(2*1024*1024,dtype=cp.uint8)).result())
    assert result['peak_pool_live_bytes']>=baseline+3*1024*1024
    assert result['peak_pool_reserved_bytes']>=result['peak_pool_live_bytes']
    assert result['host_peak_rss_bytes']>0
    assert result['observed_thread_count']>=2


def test_peak_includes_allocations_bypassing_pool():
    original=cp.cuda.get_allocator()
    cp.cuda.set_allocator(cp.cuda.memory._malloc)
    try:
        with memory_profile.peak_memory() as result:
            large=cp.empty(4*1024*1024,dtype=cp.uint8)
        assert 'peak_live_bytes' in result, 'pool-bypassing allocations are not measured'
        assert result['peak_live_bytes']>=large.nbytes
    finally: cp.cuda.set_allocator(original)
