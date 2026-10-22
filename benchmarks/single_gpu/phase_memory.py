"""Benchmark-only requested CuPy storage ledger for one GPU workflow.

Named phases are inclusive, including allocations made by worker threads while
that phase is active. Repeated names aggregate counts and peaks; entry_live_bytes
is the first entry's live storage. Array lifetimes continue across phase exits.
Importing this module does not import CuPy or initialize a GPU.
"""
from contextlib import contextmanager
import threading


class _Ledger:
    """Thread-safe accounting independent of CuPy for CPU-only verification."""

    def __init__(self):
        self.data = dict(
            live_bytes=0, peak_live_bytes=0, allocation_count=0, phases={},
            note='Requested live bytes allocated through the wrapped CuPy allocator '
                 'during this context; retains original memory owners, including '
                 'pool-bypassing allocations. Excludes native and preexisting '
                 'allocations. Phase peaks include all tracked live storage, '
                 'including storage retained from earlier phases. Repeated phase '
                 'names aggregate counts and peaks; entry_live_bytes records the '
                 'first entry. Hooks are intended for one GPU workflow only. '
                 'Requested bytes are distinct from pool reservation and RSS.')
        self._lock = threading.Lock()
        self._active = {}

    def allocate(self, size):
        with self._lock:
            self.data['live_bytes'] += size
            self.data['peak_live_bytes'] = max(
                self.data['peak_live_bytes'], self.data['live_bytes'])
            self.data['allocation_count'] += 1
            for name in set(self._active.values()):
                phase = self.data['phases'][name]
                phase['allocation_count'] += 1
                phase['peak_live_bytes'] = max(
                    phase['peak_live_bytes'], self.data['live_bytes'])

    def release(self, size):
        with self._lock:
            self.data['live_bytes'] -= size

    @contextmanager
    def phase(self, name):
        token = object()
        with self._lock:
            phase = self.data['phases'].setdefault(name, dict(
                entry_live_bytes=self.data['live_bytes'],
                peak_live_bytes=self.data['live_bytes'],
                allocation_count=0, entry_count=0))
            phase['entry_count'] += 1
            phase['peak_live_bytes'] = max(
                phase['peak_live_bytes'], self.data['live_bytes'])
            self._active[token] = name
        try:
            yield
        finally:
            with self._lock:
                del self._active[token]


@contextmanager
def phase_memory():
    """Wrap the current CuPy allocator and yield a tracker with ``phase``/``data``.

    The original MemoryPointer remains alive until the wrapping allocation dies,
    even after the context exits. The original allocator is always restored.
    Run separate profiling workflows sequentially; allocator hooks are global.
    """
    import cupy as cp

    tracker = _Ledger()
    original = cp.cuda.get_allocator()

    class Owner:
        def __init__(self, pointer, size):
            self.pointer = pointer
            self.size = size
            tracker.allocate(size)

        def __del__(self):
            tracker.release(self.size)

    def allocate(size):
        pointer = original(size)
        owner = Owner(pointer, size)
        memory = cp.cuda.UnownedMemory(
            pointer.ptr, size, owner, device_id=pointer.device_id)
        return cp.cuda.MemoryPointer(memory, 0)

    try:
        cp.cuda.set_allocator(allocate)
        yield tracker
    finally:
        try:
            cp.cuda.Device().synchronize()
        finally:
            cp.cuda.set_allocator(original)
