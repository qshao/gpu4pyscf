"""Bounded staging for temporary, single-GPU molecular DF integral storage."""
import os
from pathlib import Path
import tempfile
import weakref

import numpy as np


def _cleanup(fd, path):
    try:
        os.close(fd)
    finally:
        path.unlink(missing_ok=True)


class DiskTensor:
    """Row-major float64 CDERI; build by AO-pair columns, read auxiliary rows.

    No full-tensor mapping or host allocation. Files belong to this object and
    are removed on close or final reference release, including failed builds.
    """

    def __init__(self, shape, directory=None):
        if len(shape) != 2 or any(int(n) != n or n <= 0 for n in shape):
            raise ValueError('Disk CDERI requires two positive dimensions')
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(np.float64)
        self.nbytes = self.shape[0] * self.shape[1] * 8
        self._written = 0
        self._complete = False
        self.cache = None
        fd, path = tempfile.mkstemp(prefix='gpu4pyscf-cderi-', suffix='.bin', dir=directory)
        self._fd, self.path = fd, Path(path)
        self._finalizer = weakref.finalize(self, _cleanup, fd, self.path)
        try:
            os.ftruncate(fd, self.nbytes)
        except BaseException:
            self.close()
            raise
        self.stats = dict(read_bytes=0, write_bytes=0, h2d_bytes=0, h2d_calls=0,
                          d2h_bytes=0, d2h_calls=0, peak_host_buffer_bytes=0,
                          peak_device_buffer_bytes=0, cache_hit_bytes=0,
                          cache_device_bytes=0)

    def _check_open(self):
        if not self._finalizer.alive:
            raise RuntimeError('Disk CDERI is closed')

    def write_columns(self, p0, p1, block):
        self._check_open()
        if (self._complete or p0 != self._written or not p0 < p1 <= self.shape[1]
                or block.shape != (self.shape[0], p1-p0)
                or block.dtype != self.dtype or not block.flags.c_contiguous):
            raise ValueError('Disk CDERI requires consecutive C-contiguous FP64 column blocks')
        for row in range(self.shape[0]):
            data = memoryview(block[row]).cast('B')
            offset = (row*self.shape[1] + p0)*8
            while data:
                count = os.pwrite(self._fd, data, offset)
                if count <= 0:
                    raise OSError('Failed to write disk CDERI')
                data, offset = data[count:], offset+count
                self.stats['write_bytes'] += count
        self._written = p1

    def finish(self):
        self._check_open()
        if self._written != self.shape[1]:
            raise RuntimeError('Disk CDERI construction is not complete')
        self._complete = True

    def read_rows(self, p0, p1, out):
        self._check_open()
        if not self._complete:
            raise RuntimeError('Disk CDERI construction is not complete')
        if (not 0 <= p0 < p1 <= self.shape[0]
                or out.shape != (p1-p0, self.shape[1])
                or out.dtype != self.dtype or not out.flags.c_contiguous
                or not out.flags.writeable):
            raise ValueError('Invalid disk CDERI row buffer')
        data = memoryview(out).cast('B')
        offset = p0*self.shape[1]*8
        while data:
            count = os.preadv(self._fd, [data], offset)
            if count <= 0:
                raise OSError('Unexpected EOF reading disk CDERI')
            data, offset = data[count:], offset+count
            self.stats['read_bytes'] += count
        return out

    def close(self):
        self._finalizer()
        self.cache = None


def pinned_empty(shape):
    import cupy as cp
    size = int(np.prod(shape))
    owner = cp.cuda.alloc_pinned_memory(size*8)
    return np.frombuffer(owner, dtype=np.float64, count=size).reshape(shape)


def device_blocks(store, blksize):
    """Reuse a bounded resident prefix before streaming the remaining rows."""
    import cupy as cp
    cached_rows = 0 if store.cache is None else store.cache.shape[0]
    # Keep contraction/reduction blocks identical to uncached iteration. A
    # partial cache boundary otherwise changes SCF roundoff and DIIS history.
    if cached_rows < store.shape[0]:
        cached_rows = cached_rows // blksize * blksize
    compute = cp.cuda.get_current_stream()
    try:
        for p0 in range(0, cached_rows, blksize):
            p1 = min(cached_rows, p0+blksize)
            block = store.cache[p0:p1]
            store.stats['cache_hit_bytes'] += block.nbytes
            yield p0, p1, block
        streamed = _device_blocks(store, blksize, cached_rows)
        try:
            yield from streamed
        finally:
            streamed.close()
    finally:
        compute.synchronize()


def _device_blocks(store, blksize, start):
    """Yield bounded blocks, protecting host/device reuse with CUDA events."""
    import cupy as cp
    if start >= store.shape[0]:
        return
    host = [pinned_empty((blksize, store.shape[1])) for _ in range(2)]
    device = [cp.empty(host[0].shape, dtype=cp.float64) for _ in range(2)]
    store.stats['peak_host_buffer_bytes'] = max(store.stats['peak_host_buffer_bytes'],
                                                sum(x.nbytes for x in host))
    store.stats['peak_device_buffer_bytes'] = max(store.stats['peak_device_buffer_bytes'],
                                                  sum(x.nbytes for x in device))
    compute = cp.cuda.get_current_stream()
    io = cp.cuda.Stream(non_blocking=True)
    copied = [cp.cuda.Event() for _ in range(2)]
    consumed = [cp.cuda.Event() for _ in range(2)]

    def load(slot, p0):
        p1 = min(store.shape[0], p0+blksize)
        block = store.read_rows(p0, p1, host[slot][:p1-p0])
        device[slot][:p1-p0].set(block, stream=io)
        copied[slot].record(io)
        store.stats['h2d_bytes'] += block.nbytes
        store.stats['h2d_calls'] += 1

    try:
        load(0, start)
        for index, p0 in enumerate(range(start, store.shape[0], blksize)):
            slot = index % 2
            compute.wait_event(copied[slot])
            p1 = min(store.shape[0], p0+blksize)
            # Start the next transfer before yielding this block to the consumer.
            if p1 < store.shape[0]:
                other = 1-slot
                if index:
                    copied[other].synchronize()  # pinned host data no longer in use
                    io.wait_event(consumed[other])
                load(other, p1)
            yield p0, p1, device[slot][:p1-p0]
            consumed[slot].record(compute)
    finally:
        # A consumer can close the generator while transfers/kernels are queued.
        io.synchronize()
        compute.synchronize()
