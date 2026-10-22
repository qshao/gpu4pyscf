"""Explicit larger-molecule DF memory and contraction benchmark."""
import json
from pathlib import Path

import pytest

from df_profile import profile_df


@pytest.mark.special
@pytest.mark.benchmark
def test_df_profile():
    report = profile_df()
    assert report["natm"] == 20
    assert report["cpu_scf_converged"]
    for mode in report["storage"]:
        assert mode["actual_storage"] == mode["requested_storage"]
        assert mode["j_cpu_max_error"] < 1e-8
        assert mode["k_cpu_max_error"] < 1e-8
        assert mode["tensor_bytes"] > 0
        assert mode["default_pool_total_bytes"] >= mode["default_pool_live_bytes"]
        assert mode["measurements"]
        for row in mode["measurements"]:
            assert row["baseline_current_j_error"] < 1e-10
            assert row["baseline_current_k_error"] < 1e-10
            assert len(row["jk"]["baseline"]["samples_seconds"]) == 7
            assert len(row["jk"]["current"]["samples_seconds"]) == 7
    Path(__file__).with_name("df_profile_results.json").write_text(
        json.dumps(report, indent=2) + "\n")
