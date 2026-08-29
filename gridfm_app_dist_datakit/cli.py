from __future__ import annotations
import argparse
import os
import sys
import time
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional
import yaml

from gridfm_app_dist_datakit.datakit_gen.scenario_data_gen import ScenarioGenerator
from gridfm_app_dist_datakit.datakit_gen.pandanetwork_builder import NetworkBuilder

# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ("name", "output_dir")

# Keys that ScenarioGenerator needs; everything else is metadata.
KNOWN_KEYS = {
    "name", "network_file_path", "flag", "output_dir",
    "max_n_level", "csv_size", "n_scenario_per_partition",
    "builtin_network", "run_summary", "run_validation", "continue_on_error",
}

BUILTIN_DEFAULTS = {
    "flag": 1,
    "max_n_level": 1,
    "csv_size": 500,
    "n_scenario_per_partition": 200,
    "run_summary": True,
    "run_validation": True,
    "continue_on_error": True,
    "network_file_path": None,
    "builtin_network": None,
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if "scenarios" not in cfg or not cfg["scenarios"]:
        raise ValueError(f"Config '{path}' has no 'scenarios' list.")
    return cfg


def merge_defaults(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge global defaults into each scenario; validate required keys."""
    merged_defaults = deepcopy(BUILTIN_DEFAULTS)
    merged_defaults.update(cfg.get("defaults", {}) or {})

    jobs: List[Dict[str, Any]] = []
    seen_names, seen_dirs = set(), set()
    data_path = cfg.data_path
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

        job["output_dir"] = os.path.join(data_path, job["output_dir"] )
        out = os.path.abspath(job["output_dir"])
        if out in seen_dirs:
            raise ValueError(f"Scenario '{name}' reuses output_dir '{job['output_dir']}' already used by another job; each job needs its own dir.")
        seen_dirs.add(out)

        # Excel jobs (flag 1/2/3) require a network file; builtin jobs need a name.
        if job["flag"] is not None and not job.get("network_file_path") and not job["flag"] == 9:
            raise ValueError(f"Scenario '{name}' has flag={job['flag']} but no network_file_path.")
        if job["flag"] is None and not job.get("builtin_network") and not job["flag"] == 9:
            raise ValueError(f"Scenario '{name}' has flag=null ")
        unknown = set(raw or {}) - KNOWN_KEYS
        if unknown:
            print(f"[warn] scenario '{name}' has unrecognised keys: {sorted(unknown)}")
        jobs.append(job)
    return jobs

# Extracting the built-in network if availed in this file for jobs when flag is None
def build_builtin_net(factory_name: str):
    """Instantiate a pandapower.networks factory by name (for flag=null jobs)."""
    import pandapower.networks as pn
    if not hasattr(pn, factory_name):
        raise ValueError(f"pandapower.networks has no factory '{factory_name}'.")
    net = getattr(pn, factory_name)()
    print(f"Built in-memory network from pandapower.networks.{factory_name}()")
    return net


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single scenario job end to end; return a result record."""
    name = job["name"]
    print(f"Running job for scenario: {name}")

    # Propagate the datakit partition size to workers via the environment.
    os.environ["N_SCENARIO_PER_PARTITION"] = str(int(job["n_scenario_per_partition"]))

    # Extract network using custom Network builder (Flag 9 and builtin network in yaml file)
    if job["flag"] == 9:
        builder = NetworkBuilder.from_yaml(job["builtin_network"])
        net = builder.build() #Build the pandapower network directly from builtin_network if the flag is None
    elif job["flag"] is None:
        net = build_builtin_net(job["builtin_network"])
    else:
        net = None

    t0 = time.time()
    # Create the object for this scenario
    generator = ScenarioGenerator(job.get("network_file_path"),job["flag"],net,job["output_dir"],name)
    # Run the complete contingency analysis on this scenario and save the data in <output_dir>/raw/
    generator.run_complete_contingency_analysis(max_n_level=int(job["max_n_level"]),csv_size=int(job["csv_size"]))
    result: Dict[str, Any] = {"name": name, "output_dir": job["output_dir"], "ok": True}
    # Get summary of results
    if job.get("run_summary", True):
        summary = generator.get_results_summary()
        result["summary"] = summary
    # Run validation of the generated data
    if job.get("run_validation", True):
        try:
            validation = generator.validate_generated_data()
            result["validation"] = validation
        except Exception as e:  # validation is diagnostic; don't fail the whole job
            print(f"[{name}] Validation raised: {e}")
            result["validation_error"] = str(e)
    result["elapsed_s"] = round(time.time() - t0, 2)
    return result

def main_cli(args):

    cfg = load_config(args.config)
    jobs = merge_defaults(cfg)

    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if j["name"] in wanted]
        missing = wanted - {j["name"] for j in jobs}
        if missing:
            print(f"[warn] --only names not found in config: {sorted(missing)}")
    if not jobs:
        print("No scenarios to run.")
        return 1
    for j in jobs:
        src = j.get("builtin_network") if j["flag"] is None else j.get("network_file_path")
    if args.dry_run:
        print("\n[dry-run] Config is valid. No jobs executed.")
        return 0

    results, failures = [], []
    overall_start = time.time()
    for job in jobs:
        try:
            results.append(run_job(job)) # Run each scenario
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

