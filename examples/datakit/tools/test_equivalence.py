"""Regression test: the NetworkBuilder output must match the legacy S1/S2 scripts.

Usage:
    python tools/test_equivalence.py <legacy_dir> [configs_dir]

For every scenario the legacy input script is imported, the legacy S2 build
steps are executed on it, and the resulting pandapower net is compared element
by element against the net produced by NetworkBuilder from the YAML file.
Also checks that the std type catalogue in netlib reproduces Std_libraries.py.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

from contingency_data_gen.netlib import NetworkBuilder, STD_TYPES  # noqa: E402

SCENARIOS = [
    "rural_LV",
    "rural_MVLV",
    "suburban_LV",
    "suburban_MVLV",
    "urban_LV",
    "urban_MVLV",
]

COMPARE = {
    "bus": ["name", "vn_kv", "type"],
    "line": ["from_bus", "to_bus", "length_km", "std_type"],
    "trafo": ["hv_bus", "lv_bus", "std_type"],
    "switch": ["bus", "element", "et", "type", "closed"],
    "load": ["bus", "p_mw", "q_mvar"],
    "ext_grid": ["bus", "vm_pu"],
}


def legacy_net(legacy_dir: Path, scenario: str):
    mod = importlib.import_module(f"S1_network_model_inputs_{scenario}")
    f = importlib.import_module("S2_network_model_funcs")

    net = mod.net
    f.addbuses_MV(net, mod.MVNetwork_data, mod.MVbus_data)
    f.addbuses_LV(net, mod.LVbus_data)
    f.add_MVCBs(net, mod.MVCB_data, mod.LVbus_data)
    f.addCBs_LV(net, mod.LVbus_data, mod.LVCBs)
    f.add_MVfuses(net, mod.MVfuses)
    f.addfuses_LV(net, mod.LVbus_data, mod.LVfuses)
    f.add_MVlines(net, mod.MVline_data)
    f.add_LVlines(net, mod.LVline_data, mod.LVbus_data)
    f.addMV_trafos(net, mod.MV_trafos)
    f.addLV_trafos(net, mod.LV_trafos, mod.LVbus_data)
    f.addLPs(net, mod.LP_data, mod.LVbus_data)
    return net


def compare(old, new, scenario: str) -> list[str]:
    problems: list[str] = []
    for element, columns in COMPARE.items():
        a, b = getattr(old, element), getattr(new, element)
        if len(a) != len(b):
            problems.append(f"{element}: legacy {len(a)} rows vs builder {len(b)} rows")
            continue
        if len(a) == 0:
            continue
        cols = [c for c in columns if c in a.columns and c in b.columns]
        # compare as unordered multisets of rows (creation order is irrelevant)
        fa = a[cols].sort_values(cols).reset_index(drop=True)
        fb = b[cols].sort_values(cols).reset_index(drop=True)
        if element == "bus":  # builder keeps identical indices, check them too
            if sorted(a.index) != sorted(b.index):
                problems.append("bus: index sets differ")
        try:
            pd.testing.assert_frame_equal(fa, fb, check_dtype=False, atol=1e-9)
        except AssertionError as exc:
            problems.append(f"{element}: {str(exc).splitlines()[0]}")
    return problems


def check_std_types(legacy_dir: Path) -> list[str]:
    std = importlib.import_module("Std_libraries")
    problems: list[str] = []
    for element, names in STD_TYPES.items():
        for name, params in names.items():
            legacy = getattr(std, name, None)
            if legacy is None:
                problems.append(f"{element} {name}: missing in Std_libraries.py")
            elif dict(legacy) != dict(params):
                diff = {
                    k: (legacy.get(k), params.get(k))
                    for k in set(legacy) | set(params)
                    if legacy.get(k) != params.get(k)
                }
                problems.append(f"{element} {name}: parameter mismatch {diff}")
    return problems


def main() -> int:
    legacy_dir = Path(sys.argv[1]).resolve()
    configs = (
        Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else HERE.parent / "configs"
    )
    sys.path.insert(0, str(legacy_dir))

    cwd = Path.cwd()
    os.chdir(legacy_dir)  # legacy scripts write Results/ and Network_data/
    failures = 0
    try:
        problems = check_std_types(legacy_dir)
        print(("[ OK ] " if not problems else "[FAIL] ") + "std type catalogue")
        for p in problems:
            print("        -", p)
        failures += bool(problems)

        for scenario in SCENARIOS:
            old = legacy_net(legacy_dir, scenario)
            new = NetworkBuilder.from_yaml(
                configs / f"{scenario}.yaml",
                options={"run_diagnostic": False, "run_powerflow": False},
            ).build()
            problems = compare(old, new, scenario)
            print(("[ OK ] " if not problems else "[FAIL] ") + scenario)
            for p in problems:
                print("        -", p)
            failures += bool(problems)
    finally:
        os.chdir(cwd)

    print(f"\n{len(SCENARIOS) + 1 - failures}/{len(SCENARIOS) + 1} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
