# ============================================================
#          Integrated MCS Reliability Assessment Tool
# ============================================================
#               (c) 2026 Mikka Kisuule | INESC TEC, Porto
# ============================================================
#
# Self-contained builder class that turns a YAML input file into a pandapower network.
#
#   from netlib import NetworkBuilder
#   net = NetworkBuilder.from_yaml("configs/urban_MVLV.yaml").build()
#
# The class owns every step of the model construction (std types, MV buses,
# LV buses, switches, lines, transformers, load points, diagnostic, power flow
# and export), keeps no module level state and returns the pandapower `net`.
# ============================================================

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandapower as pp
import yaml

try:  # std type catalogue shipped with the library
    from .std_libraries import STD_TYPES as DEFAULT_STD_TYPES
except ImportError:  # pragma: no cover - allows running the file standalone
    from std_libraries import STD_TYPES as DEFAULT_STD_TYPES


class NetworkConfigError(ValueError):
    """Raised when the YAML input data is missing or inconsistent."""


class NetworkDiagnosticError(RuntimeError):
    """Raised when pandapower's diagnostic reports problems in the model."""


@dataclass
class BuildReport:
    """Bookkeeping returned alongside the network."""

    scenario: str = ""
    build_seconds: float = 0.0
    powerflow_converged: bool | None = None
    diagnostic: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "build_seconds": round(self.build_seconds, 3),
            "powerflow_converged": self.powerflow_converged,
            "diagnostic": self.diagnostic,
            "counts": self.counts,
        }


class NetworkBuilder:
    """Build a pandapower MV/LV network from a YAML input file.

    Parameters
    ----------
    config : Mapping
        Parsed YAML content (see ``configs/*.yaml`` for the schema).
    std_types : Mapping, optional
        Catalogue ``{"line": {name: params}, "trafo": {name: params}}`` used to
        resolve the std type names referenced in the config.  Defaults to
        ``netlib.std_libraries.STD_TYPES``.  Definitions given inline in the
        config under ``std_type_definitions`` take precedence.
    name : str, optional
        Label used for the network and the exported files.

    Notes
    -----
    Bus numbering follows the original convention below:

    * MV buses of the *feeder* sections are indexed ``<feeder_bus><n>`` (e.g. feeder bus 7, third bus -> 73);
    all other MV sections use a running       counter starting at ``options.mv_bus_index_start``.
    * Every LV network hanging off secondary substation ``s`` is indexed ``<s><local_bus>``;
    if that index is taken, a suffix equal to the LV bus count (incremented until free) is inserted -> ``<s><suffix><local_bus>``.
    """

    ELEMENTS = ("bus", "line", "trafo", "switch", "load", "ext_grid")

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config: Mapping[str, Any],
        std_types: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise NetworkConfigError("config must be a mapping (parsed YAML).")

        # shallow copy so the data is never mutated
        self.config: dict[str, Any] = dict(config)
        self.source_path: Path | None = None
        self.name = name or str(self.config.get("scenario", "network"))

        catalogue = std_types or DEFAULT_STD_TYPES
        self.std_types: dict[str, dict[str, dict]] = {
            "line": dict(catalogue.get("line", {})),
            "trafo": dict(catalogue.get("trafo", {})),
        }
        for element, defs in (self.config.get("std_type_definitions") or {}).items():
            if element not in self.std_types:
                raise NetworkConfigError(f"std_type_definitions: unknown element '{element}'.")
            self.std_types[element].update(defs)

        self.mv = dict(self.config.get("mv") or {})
        self.lv = dict(self.config.get("lv") or {})
        self.load_points = dict(self.config.get("load_points") or {})
        self.options = dict(self.config.get("options") or {})

        self.net: pp.pandapowerNet | None = None
        self.mv_buses: dict[str, dict[str, dict[str, list[int]]]] = {}
        self.lv_buses: dict[int, dict[str, list[int]]] = {}
        self.report = BuildReport(scenario=self.name)

        self._validate_config()

    @classmethod
    def from_yaml(
        cls,
        path: str | os.PathLike[str],
        std_types: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
        **overrides: Any,
    ) -> "NetworkBuilder":
        """
        Create a builder from a YAML input file.
        ``overrides`` are merged one level deep into the parsed config, e.g.
        ``NetworkBuilder.from_yaml(p, options={"run_powerflow": False})``.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(config.get(key), Mapping):
                merged = dict(config[key])
                merged.update(value)
                config[key] = merged
            else:
                config[key] = value
        builder = cls(config, std_types=std_types)
        builder.source_path = path
        return builder

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def _validate_config(self) -> None:
        if not self.mv:
            raise NetworkConfigError("config is missing the 'mv' block.")
        for key in ("sections", "voltage_levels", "bus_counts"):
            if key not in self.mv:
                raise NetworkConfigError(f"config['mv'] is missing '{key}'.")

        # every voltage level / section combination with a bus count list
        missing = [
            f"{voltage}_{section}"
            for voltage in self.mv["voltage_levels"]
            for section in self.mv["sections"]
            if f"{voltage}_{section}" not in self.mv["bus_counts"]
        ]
        if missing:
            raise NetworkConfigError("config['mv']['bus_counts'] is missing entries: " + ", ".join(missing))

        # feeder sections with one supplying bus per feeder subsection
        feeder_buses = self.mv.get("feeder_buses") or []
        for voltage in self.mv["voltage_levels"]:
            counts = self.mv["bus_counts"].get(f"{voltage}_feeder") or []
            if len(counts) > len(feeder_buses):
                raise NetworkConfigError(
                    f"{voltage}_feeder declares {len(counts)} feeders but only "
                    f"{len(feeder_buses)} feeder_buses are given."
                )

        if self.lv.get("sec_substations") and not self.lv.get("buses_lv_per_network"):
            raise NetworkConfigError(
                "config['lv']['buses_lv_per_network'] must be > 0 when secondary "
                "substations are defined."
            )

        # check that std type names referenced are resolvable
        unknown: list[str] = []
        for element, names in self._referenced_std_types().items():
            unknown += [
                f"{element}:{n}" for n in names if n not in self.std_types[element]
            ]
        if unknown:
            raise NetworkConfigError( "std types not found in the catalogue: " + ", ".join(sorted(unknown)))

        for placement in self._lp_placements():
            if placement not in {"lv", "mv_reduced", "feeder_eq", "trunk_feeder"}:
                raise NetworkConfigError(f"options.lp_placement: unknown value '{placement}'.")

    def _referenced_std_types(self) -> dict[str, set[str]]:
        declared = self.config.get("std_types") or {}
        names = {
            "line": set(declared.get("line") or []),
            "trafo": set(declared.get("trafo") or []),
        }
        names["line"] |= set(self.mv.get("lines") or {})
        names["line"] |= set(self.lv.get("lines") or {})
        names["trafo"] |= set(self.mv.get("trafos") or {})
        names["trafo"] |= set(self.lv.get("trafos") or {})
        return names

    def _lp_placements(self) -> list[str]:
        placement = self.options.get("lp_placement", ["lv"])
        if isinstance(placement, str):
            placement = [placement]
        return list(placement)

    # ------------------------------------------------------------------ #
    # Call Pandapower Network building API
    # ------------------------------------------------------------------ #
    def build(self) -> pp.pandapowerNet:
        """Build the network and return the pandapower net."""
        start = time.time()

        self.net = pp.create_empty_network(name=self.name)
        self._register_std_types()

        self._add_mv_buses()
        self._add_lv_buses()

        self._add_mv_circuit_breakers()
        self._add_lv_circuit_breakers()
        self._add_mv_fuses()
        self._add_lv_fuses()

        self._add_mv_lines()
        self._add_lv_lines()

        self._add_mv_trafos()
        self._add_lv_trafos()

        self._add_load_points()

        self.report.build_seconds = time.time() - start
        self.report.counts = {
            element: int(len(getattr(self.net, element))) for element in self.ELEMENTS
        }

        if self.options.get("run_diagnostic", True):
            self.diagnose(raise_on_issue=self.options.get("strict_diagnostic", False))
        if self.options.get("run_powerflow", False):
            self.run_powerflow()

        return self.net

    def diagnose(self, raise_on_issue: bool = False) -> dict[str, Any]:
        """Run pandapower's diagnostic; optionally raise if issues are found."""
        net = self._require_net()
        # the diagnostic prints a long report through the pandapower loggers
        quiet = bool(self.options.get("quiet_diagnostic", True))
        previous_disable = logging.root.manager.disable
        if quiet:
            logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = pp.diagnostic(
                    net,
                    report_style="compact",
                    warnings_only=True,
                    return_result_dict=True,
                )
        finally:
            logging.disable(previous_disable)
        result = {k: v for k, v in (result or {}).items() if v}
        self.report.diagnostic = result
        if result and raise_on_issue:
            raise NetworkDiagnosticError("diagnostic found issues: " + ", ".join(sorted(result)))
        return result

    def run_powerflow(self, **kwargs: Any) -> pp.pandapowerNet:
        """Run a balanced power flow on the built network."""
        net = self._require_net()
        pp.runpp(net, **kwargs)
        self.report.powerflow_converged = bool(net.converged)
        return net

    def export(self, folder: str | os.PathLike[str], write_excel: bool = True) -> dict[str, str]:
        """Save the network as JSON (+ optional power flow results workbook)."""
        net = self._require_net()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}

        json_path = folder / f"{self._slug()}.json"
        pp.to_json(net, str(json_path))
        written["network_json"] = str(json_path)

        report_path = folder / f"{self._slug()}_report.json"
        report_path.write_text(json.dumps(self.report.as_dict(), indent=2))
        written["report"] = str(report_path)

        if write_excel:
            if not net.converged:
                self.run_powerflow()
            import pandas as pd

            xlsx_path = folder / f"{self._slug()}_results.xlsx"
            sheets = {
                "Bus Results": net.res_bus,
                "Line Results": net.res_line,
                "Transformer Results": net.res_trafo,
                "Load Results": net.res_load,
                "Switches": net.switch,
            }
            with pd.ExcelWriter(xlsx_path) as writer:
                for sheet, frame in sheets.items():
                    frame.to_excel(writer, sheet_name=sheet)
            written["results_xlsx"] = str(xlsx_path)
        return written

    def summary(self) -> str:
        """One-line summary of the built network."""
        counts = self.report.counts or {}
        parts = [f"{k}={v}" for k, v in counts.items()]
        return f"{self.name}: " + ", ".join(parts)

    # ------------------------------------------------------------------ #
    # std types
    # ------------------------------------------------------------------ #
    def _register_std_types(self) -> None:
        net = self._require_net()
        for element, names in self._referenced_std_types().items():
            for name in sorted(names):
                pp.create_std_type(net, dict(self.std_types[element][name]), name, element=element)

    # ------------------------------------------------------------------ #
    # MV buses
    # ------------------------------------------------------------------ #
    def _add_mv_buses(self) -> None:
        """Create the MV buses section by section and connect the external grid."""
        net = self._require_net()
        feeder_buses = list(self.mv.get("feeder_buses") or [])

        start = self.options.get("mv_bus_index_start", 1)
        if start == "first_feeder_bus":  # legacy addbuses_MV_feeder_eq behaviour
            if not feeder_buses:
                raise NetworkConfigError("options.mv_bus_index_start='first_feeder_bus' needs feeder_buses.")
            bus_index = int(feeder_buses[0])
        else:
            bus_index = int(start)

        self.mv_buses = {}
        for voltage_level, voltage in self.mv["voltage_levels"].items():
            self.mv_buses[voltage_level] = {}
            for section in self.mv["sections"]:
                self.mv_buses[voltage_level][section] = {}
                counts = self.mv["bus_counts"][f"{voltage_level}_{section}"] or []
                for sub_i, num_buses in enumerate(counts):
                    subsection = f"{section}{sub_i + 1}"
                    indices: list[int] = []
                    for i in range(int(num_buses)):
                        if section == "feeder":
                            head = feeder_buses[sub_i]
                            name = f"{voltage_level}_{section}_{head}{i + 1}"
                            index = int(f"{head}{i + 1}")
                        else:
                            name = f"{voltage_level}_{section}_{subsection}_{i + 1}"
                            index = bus_index
                            bus_index += 1
                        pp.create_bus(net, vn_kv=float(voltage), name=name, index=index, type="b")
                        indices.append(index)
                    self.mv_buses[voltage_level][section][subsection] = indices

        slack = self.mv.get("slack_bus")
        if slack is not None:
            pp.create_ext_grid(
                net,
                bus=int(slack),
                vm_pu=float(self.mv.get("v_ref", 1.0)),
                name="external_grid",
            )

    # ------------------------------------------------------------------ #
    # LV buses
    # ------------------------------------------------------------------ #
    def _add_lv_buses(self) -> None:
        """Replicate one identical LV network below every secondary substation."""
        net = self._require_net()
        substations = self._sec_substations()
        if not substations:
            return

        n_mv = int(self.lv.get("buses_mv_per_network", 0))
        n_lv = int(self.lv.get("buses_lv_per_network", 0))
        v_mv = float(self._lv_voltage("MV"))
        v_lv = float(self._lv_voltage("LV"))

        used: set[int] = set()
        self.lv_buses = {}
        for substation in substations:
            record = {"mv": [], "lv": []}
            for j in range(n_mv):
                index = self._unique_index(substation, j, used, n_lv)
                pp.create_bus(net, index=index, vn_kv=v_mv, name=f"{index}", type="b")
                used.add(index)
                record["mv"].append(index)
            for j in range(n_lv):
                index = self._unique_index(substation, j + n_mv, used, n_lv)
                pp.create_bus(net, index=index, vn_kv=v_lv, name=f"{index}", type="b")
                used.add(index)
                record["lv"].append(index)
            self.lv_buses.setdefault(substation, record)

    # ------------------------------------------------------------------ #
    # switches
    # ------------------------------------------------------------------ #
    def _add_mv_circuit_breakers(self) -> None:
        """MV circuit breakers; the secondary substation supplying 'LV' networks."""
        net = self._require_net()
        n_lv = int(self.lv.get("buses_lv_per_network", 0))
        for section, connections in (self.mv.get("circuit_breakers") or {}).items():
            if str(section).upper() != "LV":
                for from_bus, to_bus in connections:
                    pp.create_switch(
                        net,
                        bus=int(from_bus),
                        element=int(to_bus),
                        et="b",
                        type="CB",
                        closed=True,
                    )
                continue
            for from_local, to_local in connections:
                used: set[int] = set()
                for substation in self._sec_substations():
                    from_bus = (
                        substation
                        if from_local is None
                        else int(f"{substation}{from_local}")
                    )
                    to_bus = self._unique_index(substation, to_local, used, n_lv)
                    used.add(to_bus)
                    pp.create_switch(
                        net,
                        bus=from_bus,
                        element=to_bus,
                        et="b",
                        type="CB",
                        closed=True,
                    )

    def _add_lv_circuit_breakers(self) -> None:
        """LV circuit breakers, replicated once per LV network.
        Numbered according to the MV bus of the secondary substation supplying the LV network
        """
        connections = self.lv.get("circuit_breakers") or []
        if not connections:
            return
        net = self._require_net()
        n_lv = int(self.lv.get("buses_lv_per_network", 0))
        seen: set[int] = set()
        for substation in self._sec_substations():
            prefix = f"{substation}" if substation not in seen else f"{substation}{n_lv + 1}"
            seen.add(substation)
            for from_local, to_local in connections:
                pp.create_switch(
                    net,
                    bus=int(f"{prefix}{from_local}"),
                    element=int(f"{prefix}{to_local}"),
                    et="b",
                    type="CB",
                    closed=True,
                )

    def _add_mv_fuses(self) -> None:
        net = self._require_net()
        for _section, connections in (self.mv.get("fuses") or {}).items():
            for from_bus, to_bus in connections:
                pp.create_switch(
                    net,
                    bus=int(from_bus),
                    element=int(to_bus),
                    et="b",
                    type="LS",
                    closed=True,
                )

    def _add_lv_fuses(self) -> None:
        self._add_replicated_switches(connections=self.lv.get("fuses") or [], switch_type="LS")

    def _add_replicated_switches(self, connections: Sequence[Sequence[int]], switch_type: str) -> None:
        """Create the same switch inside every replicated LV network."""
        if not connections:
            return
        net = self._require_net()
        n_lv = int(self.lv.get("buses_lv_per_network", 0))
        used: set[int] = set()
        for substation in self._sec_substations():
            for from_local, to_local in connections:
                from_bus = self._unique_index(substation, from_local, used, n_lv)
                used.add(from_bus)
                to_bus = self._unique_index(substation, to_local, used, n_lv)
                used.add(to_bus)
                pp.create_switch(
                    net,
                    bus=from_bus,
                    element=to_bus,
                    et="b",
                    type=switch_type,
                    closed=True,
                )

    # ------------------------------------------------------------------ #
    # lines
    # ------------------------------------------------------------------ #
    def _add_mv_lines(self) -> None:
        """MV lines, grouped std type -> length_km -> section."""
        net = self._require_net()
        for std_type, lengths in (self.mv.get("lines") or {}).items():
            for length, sections in (lengths or {}).items():
                for section, connections in (sections or {}).items():
                    for from_bus, to_bus in connections:
                        pp.create_line(
                            net,
                            from_bus=int(from_bus),
                            to_bus=int(to_bus),
                            length_km=float(length),
                            std_type=std_type,
                            name=f"{section}_{from_bus}_{to_bus}",
                        )

    def _add_lv_lines(self) -> None:
        """LV lines, grouped std type -> length_km, replicated per LV network."""
        net = self._require_net()
        lines = self.lv.get("lines") or {}
        if not lines:
            return
        n_lv = int(self.lv.get("buses_lv_per_network", 0))
        used_from: set[int] = set()
        used_to: set[int] = set()
        for substation in self._sec_substations():
            for std_type, lengths in lines.items():
                for length, connections in (lengths or {}).items():
                    for from_local, to_local in connections:
                        suffix = n_lv
                        from_bus = int(f"{substation}{from_local}")
                        to_bus = int(f"{substation}{to_local}")
                        while from_bus in used_from and to_bus in used_to:
                            suffix += 1
                            from_bus = int(f"{substation}{suffix}{from_local}")
                            to_bus = int(f"{substation}{suffix}{to_local}")
                        used_from.add(from_bus)
                        used_to.add(to_bus)
                        pp.create_line(
                            net,
                            from_bus=from_bus,
                            to_bus=to_bus,
                            length_km=float(length),
                            std_type=std_type,
                            name=f"LV{substation}_{from_local}_{to_local}",
                        )

    # ------------------------------------------------------------------ #
    # transformers
    # ------------------------------------------------------------------ #
    def _add_mv_trafos(self) -> None:
        net = self._require_net()
        for std_type, connections in (self.mv.get("trafos") or {}).items():
            for hv_bus, lv_bus in connections:
                pp.create_transformer(
                    net,
                    hv_bus=int(hv_bus),
                    lv_bus=int(lv_bus),
                    std_type=std_type,
                    name=f"MV_{hv_bus}_{lv_bus}",
                )

    def _add_lv_trafos(self) -> None:
        net = self._require_net()
        trafos = self.lv.get("trafos") or {}
        if not trafos:
            return
        n_lv = int(self.lv.get("buses_lv_per_network", 0))
        used: set[int] = set()
        for substation in self._sec_substations():
            for std_type, connections in trafos.items():
                for hv_local, lv_local in connections:
                    hv_bus = self._unique_index(substation, hv_local, used, n_lv)
                    used.add(hv_bus)
                    lv_bus = self._unique_index(substation, lv_local, used, n_lv)
                    used.add(lv_bus)
                    pp.create_transformer(
                        net,
                        hv_bus=hv_bus,
                        lv_bus=lv_bus,
                        std_type=std_type,
                        name=f"LV{substation}_{hv_local}_{lv_local}",
                    )

    # ------------------------------------------------------------------ #
    # load points
    # ------------------------------------------------------------------ #
    def _add_load_points(self) -> None:
        loads = self.load_points.get("loads") or []
        if not loads:
            return
        for placement in self._lp_placements():
            getattr(self, f"_add_lps_{placement}")()

    def _lp_power(self, customers: int) -> tuple[float, float]:
        """Active / reactive power of a load point from ADMD and power factor."""
        admd = float(self.load_points.get("lp_per_customer", 0.0))
        pf = float(self.load_points.get("power_factor", 1.0))
        p_mw = int(customers) * admd
        q_mvar = p_mw * math.tan(math.acos(pf))
        return p_mw, q_mvar

    def _add_lps_lv(self) -> None:
        """Load points inside every replicated LV network."""
        net = self._require_net()
        n_lv = int(self.lv.get("buses_lv_per_network", 0))
        used: set[int] = set()
        for substation in self._sec_substations():
            for i, load in enumerate(self.load_points["loads"]):
                bus = self._unique_index(substation, load["bus"], used, n_lv)
                used.add(bus)
                p_mw, q_mvar = self._lp_power(load["customers"])
                pp.create_load(
                    net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=f"LP_{substation}_{i + 1}"
                )

    def _add_lps_mv_reduced(self) -> None:
        """Lumped load points at MV buses that represent equivalent LV networks."""
        net = self._require_net()
        for substation in self.lv.get("sec_substations_reduced") or []:
            for i, load in enumerate(self.load_points["loads"]):
                p_mw, q_mvar = self._lp_power(load["customers"])
                pp.create_load(
                    net,
                    bus=int(substation),
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    name=f"LP_eq{substation}_{i + 1}",
                )

    def _add_lps_feeder_eq(self) -> None:
        """Lumped load points at feeder buses, one set per equivalent LV network."""
        net = self._require_net()
        substations = self.lv.get("sec_substations_reduced") or []
        networks = self.lv.get("feeder_buses_lvnetworks") or []
        if len(substations) != len(networks):
            raise NetworkConfigError("lv.sec_substations_reduced and lv.feeder_buses_lvnetworks must have the same length for lp_placement='feeder_eq'."
            )
        for substation, num_networks in zip(substations, networks):
            for i, load in enumerate(self.load_points["loads"]):
                p_mw, q_mvar = self._lp_power(load["customers"])
                for n in range(int(num_networks)):
                    pp.create_load(
                        net,
                        bus=int(substation),
                        p_mw=p_mw,
                        q_mvar=q_mvar,
                        name=f"LP_Sub{substation}_Net{n + 1}_Load{i + 1}",
                    )

    def _add_lps_trunk_feeder(self) -> None:
        """Lumped load points at main trunk feeder buses with equivalent networks."""
        net = self._require_net()
        for substation in (
            self.lv.get("sec_substations_reduced_main_trunk_feeder") or []
        ):
            for i, load in enumerate(self.load_points["loads"]):
                p_mw, q_mvar = self._lp_power(load["customers"])
                pp.create_load(
                    net,
                    bus=int(substation),
                    p_mw=p_mw,
                    q_mvar=q_mvar,
                    name=f"LP_trunk{substation}_{i + 1}",
                )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _require_net(self) -> pp.pandapowerNet:
        if self.net is None:
            raise RuntimeError("call build() before using the network.")
        return self.net

    def _sec_substations(self) -> list[int]:
        return [int(b) for b in (self.lv.get("sec_substations") or [])]

    def _lv_voltage(self, key: str) -> float:
        """Look up the LV-side voltage levels, tolerating the legacy key names."""
        levels = self.lv.get("voltage_levels") or {}
        for name, value in levels.items():
            if name == key or name.startswith(f"{key}_") or name.endswith(f"_{key}"):
                return float(value)
        raise NetworkConfigError(
            f"config['lv']['voltage_levels'] has no '{key}' entry (got {list(levels)})."
        )

    @staticmethod
    def _unique_index(
        substation: int, local: int | None, used: Iterable[int], suffix_start: int
    ) -> int:
        """Index of local bus ``local`` in the LV network below ``substation``.
        LV buses are numbered according to the ID of the MV bus of the secondary substation supplying that network
        """
        used = set(used)
        if local is None:
            return int(substation)
        suffix = int(suffix_start)
        index = int(f"{substation}{local}")
        while index in used:
            suffix += 1
            index = int(f"{substation}{suffix}{local}")
        return index

    def _slug(self) -> str:
        if self.source_path is not None:
            return self.source_path.stem
        return "".join(c if c.isalnum() else "_" for c in self.name).strip("_").lower()


# --------------------------------------------------------------------------- #
# convenience wrapper
# --------------------------------------------------------------------------- #
def build_network(config_path: str | os.PathLike[str], **overrides: Any) -> pp.pandapowerNet:
    """Build and return the pandapower network described by ``config_path``."""
    return NetworkBuilder.from_yaml(config_path, **overrides).build()


__all__ = [
    "NetworkBuilder",
    "BuildReport",
    "NetworkConfigError",
    "NetworkDiagnosticError",
    "build_network",
]
