"""
Config-driven, sequential multi-scenario runner for contingency data generation.

Describe one or more jobs in a YAML config and run them in one go::

    contingency_data_gen --config contingency_data_gen/configs/data_gen_1.yaml
    contingency_data_gen --config data_gen_1.yaml --only case33radial      # shipped config by name
    contingency_data_gen --config my_jobs.yaml --dry-run
    contingency_data_gen --list-configs

Each scenario is processed separately and written as gridfm-datakit-format
partitioned parquet to ``<output_dir>/raw/`` (bus_data / gen_data /
branch_data), the layout that ``gridfm_app_dist --data_path`` reads. Relative
``output_dir`` values resolve against the current working directory, so with
the shipped configs (``output_dir: data/<name>``) running from the repository
root puts the data where ``gridfm_app_dist train --data_path data`` expects it.

Path resolution for ``network_file_path`` / ``builtin_network`` file values:
absolute paths are used as is; relative paths are tried against the config
file's own directory first, then against the package ``configs/`` folder and
its ``networks/`` (Excel, MATPOWER) and ``netbuilder/`` (YAML) sub-folders.
A legacy ``./config/`` prefix is stripped before searching, so the original
datakit configs keep working unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml

from contingency_data_gen import CONFIG_DIR

# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ("name", "output_dir")

# Keys that ScenarioGenerator needs; everything else is metadata.
KNOWN_KEYS = {
    "name",
    "network_file_path",
    "flag",
    "output_dir",
    "max_n_level",
    "max_size",
    "n_scenario_per_partition",
    "builtin_network",
    "run_summary",
    "run_validation",
    "continue_on_error",
}

BUILTIN_DEFAULTS = {
    "flag": 1,
    "max_n_level": 1,
    "max_size": 500,
    "n_scenario_per_partition": 200,
    "run_summary": True,
    "run_validation": True,
    "continue_on_error": True,
    "network_file_path": None,
    "builtin_network": None,
}

# Flags whose builtin_network value is a FILE (not a pandapower factory name).
FILE_FLAGS = {9, 10}


def _package_config_dirs() -> List[str]:
    root = str(CONFIG_DIR)
    return [root, os.path.join(root, "networks"), os.path.join(root, "netbuilder")]


def list_shipped_configs() -> List[str]:
    root = str(CONFIG_DIR)
    return sorted(f for f in os.listdir(root) if f.endswith((".yaml", ".yml")))


def resolve_config_path(path: str) -> str:
    """Accept a real path or the bare name of a shipped config."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    cand = os.path.join(str(CONFIG_DIR), path)
    if os.path.isfile(cand):
        return cand
    if not path.endswith((".yaml", ".yml")) and os.path.isfile(cand + ".yaml"):
        return cand + ".yaml"
    raise FileNotFoundError(
        f"Config '{path}' not found (searched the path given and the shipped configs: "
        f"{', '.join(list_shipped_configs())})",
    )


def resolve_input_path(value: Optional[str], config_dir: str) -> Optional[str]:
    """Resolve a network file reference from a config; see module docstring."""
    if not value:
        return value
    if os.path.isabs(value):
        return value
    rel = value
    for prefix in ("./config/", "config/", "./configs/", "configs/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix) :]
            break
    for base in [config_dir, *_package_config_dirs()]:
        for candidate in (os.path.join(base, value), os.path.join(base, rel)):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return value  # let the loader raise a precise error later


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "scenarios" not in cfg or not cfg["scenarios"]:
        raise ValueError(f"Config '{path}' has no 'scenarios' list.")
    return cfg


def merge_defaults(cfg: Dict[str, Any], config_dir: str) -> List[Dict[str, Any]]:
    """Merge global defaults into each scenario; validate required keys; resolve paths."""
    merged_defaults = deepcopy(BUILTIN_DEFAULTS)
    merged_defaults.update(cfg.get("defaults", {}) or {})

    jobs: List[Dict[str, Any]] = []
    seen_names, seen_dirs = set(), set()
    for i, raw in enumerate(cfg["scenarios"]):
        job = deepcopy(merged_defaults)
        job.update(raw or {})

        for k in REQUIRED_KEYS:
            if not job.get(k):
                raise ValueError(f"scenarios[{i}] is missing required key '{k}'.")

        name = job["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate scenario name '{name}'.")
        seen_names.add(name)

        out = os.path.abspath(job["output_dir"])
        if out in seen_dirs:
            raise ValueError(
                f"Scenario '{name}' reuses output_dir '{job['output_dir']}' "
                f"already used by another job; each job needs its own dir.",
            )
        seen_dirs.add(out)

        flag = job["flag"]
        # Excel jobs (flag 1/2/3) need a workbook; flag 9/10 and builtin jobs need builtin_network.
        if flag not in (None, *FILE_FLAGS) and not job.get("network_file_path"):
            raise ValueError(
                f"Scenario '{name}' has flag={flag} but no network_file_path.",
            )
        if (flag is None or flag in FILE_FLAGS) and not job.get("builtin_network"):
            raise ValueError(
                f"Scenario '{name}' has flag={flag!r} but no builtin_network.",
            )

        job["network_file_path"] = resolve_input_path(
            job.get("network_file_path"),
            config_dir,
        )
        if flag in FILE_FLAGS:
            job["builtin_network"] = resolve_input_path(
                job.get("builtin_network"),
                config_dir,
            )

        unknown = set(raw or {}) - KNOWN_KEYS
        if unknown:
            print(f"[warn] scenario '{name}' has unrecognised keys: {sorted(unknown)}")
        jobs.append(job)
    return jobs


def build_builtin_net(factory_name: str):
    """Instantiate a pandapower.networks factory by name (for flag=null jobs)."""
    import pandapower.networks as pn

    if not hasattr(pn, factory_name):
        raise ValueError(f"pandapower.networks has no factory '{factory_name}'.")
    net = getattr(pn, factory_name)()
    print(f"Built in-memory network from pandapower.networks.{factory_name}()")
    return net


def _check_input_exists(job: Dict[str, Any]) -> None:
    flag = job["flag"]
    key = "builtin_network" if flag in FILE_FLAGS else "network_file_path"
    if flag is None:
        return
    path = job.get(key)
    if path and not os.path.isfile(path):
        raise FileNotFoundError(
            f"Scenario '{job['name']}': {key} '{path}' does not exist. Shipped network files "
            f"live in {os.path.join(str(CONFIG_DIR), 'networks')} and {os.path.join(str(CONFIG_DIR), 'netbuilder')}.",
        )


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single scenario job end to end; return a result record."""
    # Imported here so `--list-configs` / `--dry-run` work without pandapower installed.
    from contingency_data_gen.generation.scenario_data_gen import ScenarioGenerator

    name = job["name"]
    print(f"Running job for scenario: {name}")
    _check_input_exists(job)

    # Propagate the datakit partition size to workers via the environment.
    os.environ["N_SCENARIO_PER_PARTITION"] = str(int(job["n_scenario_per_partition"]))

    flag = job["flag"]
    if flag == 9:  # YAML network definition -> NetworkBuilder
        from contingency_data_gen.netlib import NetworkBuilder

        net = NetworkBuilder.from_yaml(job["builtin_network"]).build()
    elif flag is None:  # pandapower.networks factory
        net = build_builtin_net(job["builtin_network"])
    elif flag == 10:  # MATPOWER .m file
        from pandapower.converter.matpower import from_mpc

        net = from_mpc(job["builtin_network"], casename_mpc_file="mpc")
    else:  # Excel workbook, loaded inside ScenarioGenerator
        net = None

    t0 = time.time()
    generator = ScenarioGenerator(
        job.get("network_file_path"),
        flag,
        net,
        job["output_dir"],
        name,
    )
    generator.run_complete_contingency_analysis(
        max_n_level=int(job["max_n_level"]),
        max_size=int(job["max_size"]),
    )
    result: Dict[str, Any] = {"name": name, "output_dir": job["output_dir"], "ok": True}
    if job.get("run_summary", True):
        result["summary"] = generator.get_results_summary()
    if job.get("run_validation", True):
        try:
            result["validation"] = generator.validate_generated_data()
        except Exception as e:  # validation is diagnostic; don't fail the whole job
            print(f"[{name}] Validation raised: {e}")
            result["validation_error"] = str(e)
    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="contingency_data_gen",
        description="Generate N-k contingency scenario data for gridfm_app_dist from a YAML config.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to a YAML config, or the name of a shipped config (see --list-configs).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these scenario names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without running.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List the configs shipped with the package.",
    )
    args = parser.parse_args(argv)

    if args.list_configs:
        print(f"Shipped configs in {CONFIG_DIR}:")
        for name in list_shipped_configs():
            print(f"  {name}")
        return 0
    if not args.config:
        parser.error("--config is required (or use --list-configs).")

    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    jobs = merge_defaults(cfg, os.path.dirname(config_path))

    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if j["name"] in wanted]
        missing = wanted - {j["name"] for j in jobs}
        if missing:
            print(f"[warn] --only names not found in config: {sorted(missing)}")
    if not jobs:
        print("No scenarios to run.")
        return 1

    print(f"Config: {config_path}")
    for j in jobs:
        src = (
            j.get("builtin_network")
            if (j["flag"] is None or j["flag"] in FILE_FLAGS)
            else j.get("network_file_path")
        )
        print(
            f"  - {j['name']:<28} flag={j['flag']!s:<5} N-{j['max_n_level']}  <- {src}  -> {j['output_dir']}/raw/",
        )
    if args.dry_run:
        for j in jobs:
            try:
                _check_input_exists(j)
            except FileNotFoundError as e:
                print(f"[dry-run][warn] {e}")
        print("\n[dry-run] Config is valid. No jobs executed.")
        return 0

    results, failures = [], []
    overall_start = time.time()
    for job in jobs:
        try:
            results.append(run_job(job))
        except Exception as e:
            print(f"\n[ERROR] Scenario '{job['name']}' failed: {e}")
            traceback.print_exc()
            failures.append(job["name"])
            if not job.get("continue_on_error", True):
                print("continue_on_error is false; aborting remaining scenarios.")
                break

    print("\n" + "=" * 70)
    print("ALL SCENARIOS FINISHED")
    print(f"  succeeded : {len(results)}")
    print(f"  failed    : {len(failures)} {failures if failures else ''}")
    print(f"  total time: {round(time.time() - overall_start, 2)}s")
    print("=" * 70)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
