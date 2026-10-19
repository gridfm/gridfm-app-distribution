"""
Smoke tests for the contingency_data_gen application.

The config-handling tests run anywhere. The generation test needs pandapower
and runs a tiny N-1 analysis on a built-in pandapower case, so it is skipped
when pandapower is not installed and can be deselected with ``-m "not slow"``.
"""

import os

import pytest

from contingency_data_gen import CONFIG_DIR
from contingency_data_gen.__main__ import (
    list_shipped_configs,
    load_config,
    merge_defaults,
    resolve_config_path,
    resolve_input_path,
)

pandapower = pytest.importorskip(
    "pandapower",
    reason="pandapower is required for data generation",
)


def test_shipped_configs_resolve_and_validate():
    names = list_shipped_configs()
    assert "data_gen_1.yaml" in names
    for name in names:
        path = resolve_config_path(name)
        cfg = load_config(path)
        jobs = merge_defaults(cfg, os.path.dirname(path))
        assert jobs, f"{name} produced no jobs"
        for job in jobs:
            assert job["output_dir"].replace("\\", "/").startswith("data/")


def test_input_paths_resolve_to_package_files():
    cfg_dir = str(CONFIG_DIR)
    xlsx = resolve_input_path("./config/case33radial.xlsx", cfg_dir)
    assert os.path.isfile(xlsx) and xlsx.endswith("case33radial.xlsx")
    yml = resolve_input_path("urban_LV.yaml", cfg_dir)
    assert os.path.isfile(yml) and os.sep + "netbuilder" + os.sep in yml
    missing = resolve_input_path("networks/does_not_exist.m", cfg_dir)
    assert missing == "networks/does_not_exist.m"  # left for a precise error later


@pytest.mark.slow
def test_tiny_n1_generation_writes_datakit_parquet(tmp_path, monkeypatch):
    import pandas as pd

    from contingency_data_gen.__main__ import run_job

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    out = tmp_path / "scenario_case33bw"
    job = {
        "name": "case33bw_smoke",
        "flag": None,
        "builtin_network": "case33bw",
        "network_file_path": None,
        "output_dir": str(out),
        "max_n_level": 1,
        "max_size": 20,
        "n_scenario_per_partition": 10,
        "run_summary": False,
        "run_validation": True,
        "continue_on_error": False,
    }
    result = run_job(job)
    assert result["ok"]
    raw = out / "raw"
    for name in ("bus_data.parquet", "branch_data.parquet"):
        assert (raw / name).exists(), f"{name} missing"
    bus = pd.read_parquet(raw / "bus_data.parquet")
    assert {"scenario", "scenario_partition"} <= set(bus.columns)
    assert bus["scenario"].nunique() > 1
    # Hive partitioning reads scenario_partition back as a categorical.
    partition = bus["scenario_partition"].astype(int)
    assert (bus["scenario"] // 10 == partition).all()
    assert partition.nunique() > 1, "all scenarios fell into one partition"

    # case33bw has no generators (ext_grid only). The writer then leaves
    # gen_data.parquet absent and the model treats the dataset as
    # generator-free, so the file must be missing-or-empty, never partial.
    from contingency_data_gen.generation import datakit_writer as dkw

    gen = dkw.read_dataset(str(raw), "gen")
    assert gen.empty
    if (raw / "gen_data.parquet").exists():
        assert pd.read_parquet(raw / "gen_data.parquet").empty
