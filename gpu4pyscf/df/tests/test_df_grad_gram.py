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

import unittest
import numpy as np
import cupy as cp
from gpu4pyscf.df.grad.rhf import _auxiliary_gram, _apply_auxiliary_metric


class KnownValues(unittest.TestCase):
    def test_gram_matches_einsum(self):
        rng = np.random.default_rng(7)
        for shape in [(37, 5, 5), (300, 4, 4), (2, 3, 6, 6)]:
            dm = cp.asarray(rng.random(shape))
            spin = 'n' if dm.ndim == 4 else ''
            ref_sym = cp.einsum(f'{spin}rij,{spin}sij->rs', dm, dm)
            ref_tr = cp.einsum(f'{spin}rij,{spin}sji->rs', dm, dm)
            naux = shape[-3]
            for transposed, ref in ((False, ref_sym), (True, ref_tr)):
                for block in (256, 7):
                    out = _auxiliary_gram(dm, transposed, -.5, 0., block=block)
                    assert abs(out + .5 * ref).max() < 1e-12
                    start = cp.asarray(rng.random((naux, naux)))
                    expected = 1.5 * start - .5 * ref
                    out = _auxiliary_gram(dm, transposed, -.5, 1.5, out=start.copy(), block=block)
                    assert abs(out - expected).max() < 1e-12

    def test_in_place_metric_matches_out_of_place(self):
        rng = np.random.default_rng(11)
        nao_cart, nao_sph, nocc = 40, 33, 6
        coeff = cp.asarray(rng.random((nao_cart, nao_sph)))
        metric = rng.random((nao_sph, nao_sph))
        metric = cp.asarray(metric @ metric.T + nao_sph * np.eye(nao_sph))
        solve = lambda b: cp.linalg.solve(metric, b)
        rhs = cp.asarray(rng.random((nao_cart, nocc, nocc)))
        expected = _apply_auxiliary_metric(coeff, solve, rhs.copy())
        work = rhs.copy()
        result = _apply_auxiliary_metric(coeff, solve, work, out=work)
        assert result is work
        assert abs(result - expected).max() < 1e-12


if __name__ == "__main__":
    print("Full Tests for the DF gradient Gram and metric helpers")
    unittest.main()
