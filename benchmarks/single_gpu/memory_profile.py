"""Benchmark-only process RSS and all-CuPy allocation instrumentation."""
from contextlib import contextmanager
from pathlib import Path
import resource
import threading

import cupy as cp


@contextmanager
def peak_memory():
    pool=cp.get_default_memory_pool()
    data=dict(peak_pool_live_bytes=pool.used_bytes(),
              peak_pool_reserved_bytes=pool.total_bytes(),allocation_observations=0,
              initial_pool_live_bytes=pool.used_bytes(),peak_tracked_live_bytes=0,
              peak_live_bytes=pool.used_bytes())
    live=0
    threads=set()
    lock=threading.Lock()
    original=cp.cuda.get_allocator()
    class Owner:
        def __init__(self,pointer,size):
            self.pointer,self.size=pointer,size
        def __del__(self):
            nonlocal live
            with lock:
                live-=self.size
    def allocate(size):
        nonlocal live
        pointer=original(size)
        used,total=pool.used_bytes(),pool.total_bytes()
        owner=Owner(pointer,size)
        with lock:
            live+=size
            data['peak_tracked_live_bytes']=max(data['peak_tracked_live_bytes'],live)
            data['peak_live_bytes']=data['initial_pool_live_bytes']+data['peak_tracked_live_bytes']
            data['peak_pool_live_bytes']=max(data['peak_pool_live_bytes'],used)
            data['peak_pool_reserved_bytes']=max(data['peak_pool_reserved_bytes'],total)
            data['allocation_observations']+=1
            threads.add(threading.get_ident())
        memory=cp.cuda.UnownedMemory(pointer.ptr,size,owner,device_id=pointer.device_id)
        return cp.cuda.MemoryPointer(memory,0)
    try:
        cp.cuda.set_allocator(allocate)
        yield data
    finally:
        cp.cuda.Device().synchronize()
        cp.cuda.set_allocator(original)
        status=Path('/proc/self/status').read_text().splitlines()
        host_hwm=next(int(s.split()[1])*1024 for s in status if s.startswith('VmHWM:'))
        data.update(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    host_address_space_peak_rss_bytes=host_hwm,
                    observed_thread_count=len(threads),
                    note='Wraps existing CuPy allocator and retains original memory owners across threads, including pool-bypassing allocations. peak_tracked_live_bytes is peak requested live storage allocated inside the context. peak_live_bytes adds the fixed initial_pool_live_bytes offset and can overcount preexisting pool storage later freed. Native/preexisting nonpool storage is excluded. Pool reserved and process RSS are separate; VmHWM is current address-space RSS high-water. On unified-memory GB10 do not sum metrics. Fresh process per variant; timing is diagnostic only.')
