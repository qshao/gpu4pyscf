"""Repeated identical-density J/K checks for storage and transfer state."""
import json
from pathlib import Path
from unittest.mock import patch

import cupy as cp
import numpy as np
import pytest

from molecular import SETTINGS, _mean_field


@pytest.mark.special
@pytest.mark.benchmark
def test_repeated_disk_jk(tmp_path):
    from gpu4pyscf.df.df import DF
    base = Path(__file__).parent
    refs = json.loads((base/'stage_profile_results.json').read_text())
    cpu = next(r['cpu_reference'] for r in refs['cases'] if r['case']=='vitamin_svp' and r['method']=='B3LYP')
    with patch.dict(SETTINGS,cpu['settings']):
        mf, guess = _mean_field('B3LYP','gpu')
    mf.kernel(dm0=guess)
    assert mf.converged
    dm = mf.make_rdm1()
    original_dm = dm.copy()
    resident = mf.with_df
    expected = tuple(x.copy() for x in resident.get_jk(dm,hermi=1))
    rows = []
    try:
        for cache_bytes in [0,64*1024**2]:
            obj = DF(mf.mol,cpu['settings']['auxbasis'])
            obj.cderi_storage='disk'
            obj.cderi_directory=tmp_path
            obj.cderi_cache_bytes=cache_bytes
            obj.build()
            cache = obj._cderi[0].cache
            snapshot = cache.copy() if cache is not None else None
            if cache is not None:
                stored = np.empty(cache.shape)
                obj._cderi[0].read_rows(0,len(stored),stored)
                cp.testing.assert_array_equal(cache,cp.asarray(stored))
            first = None
            errors=[]
            try:
                for repeat in range(8):
                    actual = obj.get_jk(dm,hermi=1)
                    errors.append([float(cp.max(cp.abs(a-b))) for a,b in zip(actual,expected)])
                    assert max(errors[-1]) < 1e-10
                    if first is None: first=tuple(x.copy() for x in actual)
                    for a,b in zip(actual,first): cp.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
                    cp.testing.assert_array_equal(dm,original_dm)
                    if cache is not None: cp.testing.assert_array_equal(cache,snapshot)
            finally:
                obj.reset()
            rows.append(dict(cache_budget_bytes=cache_bytes,errors=errors))
    finally:
        resident.reset()
    (base/'disk_df_repeated_jk.json').write_text(json.dumps(dict(status='passed',rows=rows),indent=2)+'\n')
