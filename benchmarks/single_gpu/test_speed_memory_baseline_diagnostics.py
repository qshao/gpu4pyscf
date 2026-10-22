"""Reproduce strict upstream fingerprints with saved pre-change functions."""
import importlib
from contextlib import nullcontext
from unittest.mock import patch
import pytest
from speed_memory_candidates import version


@pytest.mark.parametrize('variant',['current','compact','production'])
def test_rhf_gradient_fingerprint(variant):
    from gpu4pyscf.df.grad import rhf
    upstream=importlib.import_module('gpu4pyscf.df.tests.test_df_rhf_grad')
    context=nullcontext() if variant=='production' else version(variant)
    with context,patch.object(upstream,'_jk_energy_per_atom',rhf._jk_energy_per_atom):
        upstream.setUpModule()
        try:
            upstream.KnownValues('test_jk_energy_per_atom').test_jk_energy_per_atom()
        finally: upstream.tearDownModule()
