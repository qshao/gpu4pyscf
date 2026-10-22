"""Benchmark-only cap on the full-grid weight-response batching target."""
from contextlib import contextmanager
import ast
import hashlib
from pathlib import Path
from unittest.mock import patch


@contextmanager
def original_workspace(budget=None):
    from gpu4pyscf.grad import rks
    original = rks.get_exc_full_response
    path = Path(__file__).parent/'runs/2026-10-18-pre-next-optimization/gpu4pyscf__grad__rks.py.source'
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == 'fd4fd2177567e39fef8703fddcf1165397de755bb5f054b889da0ddaafef576e'
    text = data.decode()
    node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)
                and n.name == 'get_exc_full_response')
    source = ast.get_source_segment(text,node)
    if budget is not None:
        old = 'available_gpu_memory = int(available_gpu_memory * 0.1)'
        assert source.count(old) == 1
        source = source.replace(old, f'available_gpu_memory = min(int(available_gpu_memory * 0.1), {int(budget)})')
        old = 'ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16'
        assert source.count(old) == 1
        source = source.replace(old, 'ngrids_per_batch = ngrids_per_batch // 16 * 16')
    namespace = dict(original.__globals__)
    exec(compile(source, '<benchmark XC workspace>', 'exec'), namespace)
    with patch.object(rks, 'get_exc_full_response', namespace['get_exc_full_response']):
        yield


def capped_workspace(budget=512 * 1024**2):
    return original_workspace(budget)
