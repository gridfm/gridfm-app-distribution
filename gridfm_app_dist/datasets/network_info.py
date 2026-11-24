"""
Per-network MVA base (``data.baseMVA``) and its check against ``network_info.txt``.

Every network is normalised with its own MVA base, the ``sn_mva`` of its
pandapower net, which the data generator writes to
``<data_path>/<network>/network_info.txt`` as ``base_mva``. The admittances
(Yff/Yft) and shunts (GS/BS) in the processed data are in per unit on that base,
so the normaliser converts them with ``base_mva(network) / fitted_baseMVA``.
A single base for all networks (the old ``baseMVA: 100``) scaled them wrongly by
``sn_mva / 100`` on every network whose base is not 100.

Config (one value per network, in the order of ``data.networks``)::

    data:
      networks: [scenario_kerber_2, scenario_33radial, JRC_LV_urban]
      baseMVA:  [1.0,               100.0,             0.01]
      check_base_mva: true     # default; false only to evaluate legacy checkpoints

A mapping ``baseMVA: {scenario_kerber_2: 1.0, ...}`` is accepted as well. A single
number is accepted only when ``data.networks`` has exactly one network.

``check_base_mva`` compares the configured bases with ``network_info.txt`` and
raises one error that lists every mismatch or missing file. The datamodule runs
it when it is created, i.e. before any normaliser is built, fitted or restored.
The same check runs as a pre-flight from a job script::

    python -m gridfm_app_dist.datasets.network_info --config config.yaml --data_path data

(exit status 1 on any problem).
"""

import argparse
import math
import os
import sys
import warnings

INFO_FILE = "network_info.txt"
BASE_KEYS = ("base_mva", "sn_mva")
REL_TOL = 1e-6


class BaseMVAError(ValueError):
    """The configured MVA bases are malformed or disagree with network_info.txt."""


# ---------------------------------------------------------------- network_info
def network_info_path(data_dir, network):
    """``<data_dir>/<network>/network_info.txt`` (or in its ``raw/`` folder), or None."""
    for path in (
        os.path.join(data_dir, network, INFO_FILE),
        os.path.join(data_dir, network, "raw", INFO_FILE),
    ):
        if os.path.isfile(path):
            return path
    return None


def read_network_info(path):
    """``key: value`` lines of network_info.txt; numbers as floats.

    The first occurrence of a key wins, so the top-level entries are not
    overwritten by indented sub-entries further down.
    """
    info = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if not key or key in info:
                continue
            try:
                info[key] = float(value)
            except ValueError:
                info[key] = value
    return info


def info_base_mva(info):
    """The MVA base recorded in a parsed network_info.txt, or None."""
    for key in BASE_KEYS:
        value = info.get(key)
        if isinstance(value, float):
            return value
    return None


# ---------------------------------------------------------------- config
def _as_mapping(raw):
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, argparse.Namespace):  # NestedNamespace turns a YAML mapping into one
        return vars(raw)
    return None


def configured_base_mva(args):
    """``data.baseMVA`` as one float per network, in the order of ``data.networks``.

    Raises BaseMVAError if it is missing, has the wrong length or keys, is a
    single number for several networks, or holds a non-positive value.
    """
    data = args.data
    networks = list(getattr(data, "networks", None) or [])
    raw = getattr(data, "baseMVA", None)
    example = "e.g. baseMVA: [1.0, 100.0] for networks: [scenario_kerber_2, scenario_33radial]"
    if raw is None:
        raise BaseMVAError(
            "data.baseMVA is required: one MVA base per network in data.networks "
            f"(the network's base_mva in network_info.txt), {example}.",
        )

    mapping = _as_mapping(raw)
    if mapping is not None:
        missing = [n for n in networks if n not in mapping]
        extra = [k for k in mapping if k not in networks]
        if missing or extra:
            raise BaseMVAError(
                "data.baseMVA (mapping) must have exactly one entry per network in "
                f"data.networks; missing: {missing}, not in data.networks: {extra}.",
            )
        values = [mapping[n] for n in networks]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
        if networks and len(values) != len(networks):
            raise BaseMVAError(
                f"data.baseMVA has {len(values)} value(s) but data.networks has "
                f"{len(networks)}: give one MVA base per network, in the order of "
                f"data.networks ({example}).",
            )
    else:
        if len(networks) > 1:
            raise BaseMVAError(
                f"data.baseMVA is a single value ({raw!r}) but data.networks lists "
                f"{len(networks)} networks. Each network is normalised with its own MVA "
                "base, so give a list with one value per network, in the order of "
                f"data.networks ({example}).",
            )
        values = [raw]

    if not networks:
        networks = [f"network {i}" for i in range(len(values))]
    bases = []
    for network, value in zip(networks, values):
        try:
            base = float(value)
        except (TypeError, ValueError):
            raise BaseMVAError(f"data.baseMVA for {network} is not a number: {value!r}.")
        if not math.isfinite(base) or base <= 0:
            raise BaseMVAError(f"data.baseMVA for {network} must be > 0, got {value!r}.")
        bases.append(base)
    return bases


def single_network_base_mva(args):
    """The configured base if the config has exactly one network, else None."""
    bases = configured_base_mva(args)
    return bases[0] if len(bases) == 1 else None


def base_mva_matches(a, b):
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=1e-12)


# ---------------------------------------------------------------- check
def compare_with_network_info(args, data_dir):
    """Rows (network, configured, network_info, path, problem or None)."""
    networks = list(args.data.networks)
    rows = []
    for network, base in zip(networks, configured_base_mva(args)):
        path = network_info_path(data_dir, network)
        if path is None:
            rows.append(
                (
                    network,
                    base,
                    None,
                    None,
                    f"no {INFO_FILE} in {os.path.join(data_dir, network)} or its raw/ folder",
                ),
            )
            continue
        info_base = info_base_mva(read_network_info(path))
        if info_base is None:
            rows.append(
                (network, base, None, path, f"{path} has no {' / '.join(BASE_KEYS)} entry"),
            )
        elif not base_mva_matches(base, info_base):
            rows.append(
                (
                    network,
                    base,
                    info_base,
                    path,
                    f"data.baseMVA = {base:g} but {path} says base_mva = {info_base:g}",
                ),
            )
        else:
            rows.append((network, base, info_base, path, None))
    return rows


def check_base_mva(args, data_dir, enabled=None):
    """Validate ``data.baseMVA`` and compare it with every network's network_info.txt.

    Returns the per-network bases (order of ``data.networks``). Raises
    BaseMVAError listing every problem. ``enabled`` defaults to
    ``data.check_base_mva`` (true); when disabled only the config format is
    checked and a warning is issued.
    """
    bases = configured_base_mva(args)
    if enabled is None:
        enabled = bool(getattr(args.data, "check_base_mva", True))
    if not enabled:
        warnings.warn(
            "data.check_base_mva is false: the MVA bases "
            f"{dict(zip(args.data.networks, bases))} are NOT checked against "
            f"{INFO_FILE}. Use this only to evaluate checkpoints trained with the "
            "legacy common base.",
        )
        return bases
    problems = [f"{row[0]}: {row[4]}" for row in compare_with_network_info(args, data_dir) if row[4]]
    if problems:
        raise BaseMVAError(
            "MVA base check failed (data.baseMVA vs network_info.txt):\n  - "
            + "\n  - ".join(problems)
            + "\nSet data.baseMVA to each network's base_mva, one value per network in "
            "the order of data.networks.",
        )
    return bases


# ---------------------------------------------------------------- pre-flight CLI
def info_bases(networks, data_dir):
    """network_info.txt base per network (None where missing)."""
    out = []
    for network in networks:
        path = network_info_path(data_dir, network)
        out.append(info_base_mva(read_network_info(path)) if path else None)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Check data.baseMVA of a config against each network's network_info.txt "
            "(pre-flight for job scripts; exit status 1 on any problem). It always "
            "checks, whatever data.check_base_mva says."
        ),
    )
    parser.add_argument("--config", required=True, help="training / evaluation YAML")
    parser.add_argument("--data_path", default="data", help="same --data_path as the run")
    a = parser.parse_args(argv)

    import yaml
    from gridfm_app_dist.io.param_handler import NestedNamespace

    with open(a.config) as f:
        args = NestedNamespace(**yaml.safe_load(f))
    networks = list(args.data.networks)
    found = info_bases(networks, a.data_path)
    if all(b is not None for b in found):
        print(
            "network_info.txt bases (in the order of data.networks):\n"
            f"  baseMVA: [{', '.join(f'{b:g}' for b in found)}]",
        )
    try:
        rows = compare_with_network_info(args, a.data_path)
    except BaseMVAError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    width = max(len(r[0]) for r in rows) if rows else 7
    print(f"{'network':<{width}}  {'config':>10}  {'network_info':>12}  status")
    for network, base, info_base, _, problem in rows:
        shown = "-" if info_base is None else f"{info_base:g}"
        print(f"{network:<{width}}  {base:>10g}  {shown:>12}  {'MISMATCH' if problem else 'ok'}")
    problems = [f"{r[0]}: {r[4]}" for r in rows if r[4]]
    if problems:
        print("ERROR: MVA base check failed:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    print(f"MVA base check passed ({a.config}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
