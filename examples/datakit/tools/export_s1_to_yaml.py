"""
Convert the legacy S1_network_model_inputs_<scenario>.py scripts into YAML
input files for the NetworkBuilder class.

Usage:
    python tools/export_s1_to_yaml.py <legacy_dir> <output_dir>

The legacy directory must contain the S1 input scripts together with
S1_network_model_inputs_funcs.py, S2_network_model_funcs.py, Std_libraries.py
and functions.py.

This script is a one-off migration helper: once the YAML files exist, the
legacy scripts are no longer needed.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections import OrderedDict

import yaml

SCENARIOS = [
    "rural_LV",
    "rural_MVLV",
    "suburban_LV",
    "suburban_MVLV",
    "urban_LV",
    "urban_MVLV",
]


# --------------------------------------------------------------------------- #
# YAML representers: keep dict order, write tuples as lists, plain floats
# --------------------------------------------------------------------------- #
class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    OrderedDict,
    lambda d, data: d.represent_mapping("tag:yaml.org,2002:map", data.items()),
)
_Dumper.add_representer(
    tuple,
    lambda d, data: d.represent_sequence("tag:yaml.org,2002:seq", list(data)),
)


def _conns(connections):
    """[{'from_bus': a, 'to_bus': b}, ...] -> [[a, b], ...]"""
    return [[c["from_bus"], c["to_bus"]] for c in connections]


def _pairs(connections):
    """[(a, b), ...] -> [[a, b], ...]"""
    return [[int(a), int(b)] for a, b in connections]


def _custom_std_types(net):
    """Return the std types that were registered on top of the pandapower defaults."""
    import pandapower as pp

    default = pp.create_empty_network()
    out = {}
    for element in ("line", "trafo"):
        names = [
            n
            for n in net.std_types.get(element, {})
            if n not in default.std_types.get(element, {})
        ]
        out[element] = sorted(names)
    return out


def convert(module_name: str) -> OrderedDict:
    mod = importlib.import_module(module_name)

    mvn = mod.MVNetwork_data
    lvb = mod.LVbus_data
    lp = mod.LP_data

    # ---------------- MV lines: std_type -> length -> section -> conns ------ #
    mv_lines = OrderedDict()
    for std_type, lengths in mod.MVline_data.items():
        entry = OrderedDict()
        for length, sections in lengths.items():
            sec = OrderedDict(
                (section, _conns(data["connections"]))
                for section, data in sections.items()
                if data["connections"]
            )
            if sec:
                entry[float(length)] = sec
        mv_lines[std_type] = entry

    # ---------------- LV lines: std_type -> length -> conns ---------------- #
    lv_lines = OrderedDict()
    for std_type, lengths in mod.LVline_data.items():
        entry = OrderedDict()
        for length, data in lengths.items():
            if data["connections"]:
                entry[float(length)] = _conns(data["connections"])
        lv_lines[std_type] = entry

    cfg = OrderedDict()
    cfg["scenario"] = mvn["Network_scenario"]
    cfg["std_types"] = OrderedDict(
        (k, v) for k, v in _custom_std_types(mod.net).items()
    )

    cfg["mv"] = OrderedDict(
        [
            ("v_ref", float(mvn["V_ref"])),
            ("slack_bus", int(mvn["slack_bus"])),
            ("sections", OrderedDict(mvn["sections"])),
            (
                "voltage_levels",
                OrderedDict((k, float(v)) for k, v in mvn["voltage_levels"].items()),
            ),
            ("bus_counts", OrderedDict(mod.bus_data)),
            ("feeder_buses", [int(b) for b in mvn.get("feeder_buses", [])]),
            (
                "circuit_breakers",
                OrderedDict(
                    (section, _conns(data["connections"]))
                    for section, data in mod.MVCB_data.items()
                ),
            ),
            (
                "fuses",
                OrderedDict(
                    (section, _conns(data["connections"]))
                    for section, data in mod.MVfuses.items()
                ),
            ),
            ("lines", mv_lines),
            (
                "trafos",
                OrderedDict(
                    (std_type, _pairs(data["connections"]))
                    for std_type, data in mod.MV_trafos.items()
                ),
            ),
        ],
    )

    reduced = lvb.get("Sec_substations_reduced") or []
    cfg["lv"] = OrderedDict(
        [
            ("buses_mv_per_network", int(lvb["buses11_LV"])),
            ("buses_lv_per_network", int(lvb["busesLV"])),
            (
                "voltage_levels",
                OrderedDict((k, float(v)) for k, v in lvb["voltage_levels"].items()),
            ),
            ("sec_substations", [int(b) for b in lvb["Sec_substations"]]),
            ("sec_substations_reduced", [int(b) for b in reduced]),
            (
                "sec_substations_reduced_main_trunk_feeder",
                [
                    int(b)
                    for b in lvb.get("Sec_substations_reduced_data_main_trunk_feeder")
                    or []
                ],
            ),
            (
                "feeder_buses_lvnetworks",
                [int(n) for n in lvb.get("feeder_buses_LVnetworks") or []],
            ),
            ("circuit_breakers", _conns(mod.LVCBs["connections"])),
            ("fuses", _conns(mod.LVfuses["connections"])),
            ("lines", lv_lines),
            (
                "trafos",
                OrderedDict(
                    (std_type, _pairs(data["connections"]))
                    for std_type, data in mod.LV_trafos.items()
                ),
            ),
        ],
    )

    cfg["load_points"] = OrderedDict(
        [
            ("lp_per_customer", float(lp["LP_per_customer"])),
            ("power_factor", float(lp["power_factor"])),
            (
                "loads",
                [
                    OrderedDict(
                        [("bus", int(ld["bus"])), ("customers", int(ld["customers"]))],
                    )
                    for ld in lp["loads"]
                ],
            ),
        ],
    )

    cfg["options"] = OrderedDict(
        [
            ("lp_placement", ["lv"]),
            ("run_diagnostic", True),
            ("run_powerflow", True),
        ],
    )
    return cfg


def main() -> None:
    legacy_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, legacy_dir)
    cwd = os.getcwd()
    os.chdir(legacy_dir)  # legacy scripts write Results/ and Network_data/ folders
    try:
        for scenario in SCENARIOS:
            cfg = convert(f"S1_network_model_inputs_{scenario}")
            path = os.path.join(out_dir, f"{scenario}.yaml")
            with open(path, "w") as fh:
                yaml.dump(
                    cfg,
                    fh,
                    Dumper=_Dumper,
                    sort_keys=False,
                    default_flow_style=False,
                )
            print(f"wrote {path}")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
