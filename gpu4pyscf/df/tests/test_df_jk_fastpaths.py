# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Low-level DF J/K fast paths (CuPy-engine GEMM/GEMV/SYRK) against plain einsum, including padded leading dimensions.'''

import unittest
import numpy as np
import cupy as cp
from gpu4pyscf.df import df_jk


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.saved_engine = df_jk.contract_engine
        df_jk.contract_engine = 'cupy'   # enables the fast paths regardless of the installed cuTENSOR

    def tearDown(self):
        df_jk.contract_engine = self.saved_engine

    def test_exchange_transform_padded_layout(self):
        rng = np.random.default_rng(3)
        ni, naux, ld, nocc, batch = 512, 256, 300, 6, 2   # ld > naux: padded auxiliary dimension
        base = cp.asarray(rng.random((ni, ni, ld)))
        a = base.transpose(2, 0, 1)[:naux]                  # auxiliary-contiguous DF block
        assert a.strides[0] == 8 and a.strides[2] == ld * 8
        b = cp.asarray(rng.random((batch, ni, nocc)))
        out = cp.empty((batch, ni, nocc, naux))
        expected = cp.einsum('Lij,njk->nikL', a, b)
        result = df_jk._exchange_transform(a, b, out)
        assert result is out
        assert abs(out - expected).max() < 1e-11
        # an unpadded, packed block must give the same answer
        packed = cp.ascontiguousarray(a)
        out2 = cp.empty_like(out)
        df_jk._exchange_transform(packed, b, out2)
        assert abs(out2 - expected).max() < 1e-11

    def test_symmetric_gram_accumulates_and_fills_both_triangles(self):
        rng = np.random.default_rng(5)
        n, nao, k = 2, 300, 50
        x = cp.asarray(rng.random((n, nao, k)))
        start = cp.asarray(rng.random((n, nao, nao)))
        start = start + start.transpose(0, 2, 1)             # symmetric accumulator, as in get_jk
        out = start.copy()
        df_jk._symmetric_gram(x, out)
        expected = start + cp.einsum('nik,njk->nij', x, x)
        assert abs(out - expected).max() < 1e-11
        assert abs(out - out.transpose(0, 2, 1)).max() < 1e-12

    def test_coulomb_gemv_layouts(self):
        rng = np.random.default_rng(9)
        L, p, pad = 40, 300, 17
        dense = cp.asarray(rng.random((L, p)))
        aux_contig = cp.asarray(rng.random((p, L + pad))).T[:L]      # strides (8, ld*8)
        pair_contig = cp.asarray(rng.random((L + pad, p + pad)))[:L, :p]  # strides (ld*8, 8)
        for b in (dense, aux_contig, pair_contig):
            ref_b = cp.asarray(b)
            a1 = cp.asarray(rng.random((1, p)))
            out = df_jk._coulomb_contract('np,Lp->nL', a1, b)
            assert abs(out - a1 @ ref_b.T).max() < 1e-11
            a2 = cp.asarray(rng.random((1, L)))
            start = cp.asarray(rng.random((1, p)))
            out = df_jk._coulomb_contract('nL,Lp->np', a2, b, alpha=.5, beta=1., out=start.copy())
            assert abs(out - (start + .5 * a2 @ ref_b)).max() < 1e-11


if __name__ == "__main__":
    print("Full Tests for the DF J/K fast paths")
    unittest.main()
