"""Generate VLD test data for the gridfm-app-distribution integration test.

Adapted from the original gridfm-graphkit integration harness. Changes:
  * Single VLD data generator (the old PF/OPF split does not apply; the app has no OPF mode).
  * Pulls the app's own default VLD config (scripts/config/default.yaml) shape.
"""

import urllib.request
import yaml
import subprocess


def execute_and_live_output(cmd) -> None:
    subprocess.run(cmd, text=True, shell=True, check=True)


def _base_config() -> dict:
    """Download the default config from gridfm-app-distribution and apply common test parameters."""
    config_url = (
        "https://raw.githubusercontent.com/gridfm/gridfm-app-distribution/"
        "refs/heads/main/scripts/config/default.yaml"
    )

    print(f"Downloading config from {config_url}...")
    with urllib.request.urlopen(config_url) as response:
        config_content = response.read().decode("utf-8")
    config = yaml.safe_load(config_content)

    # The VLD app config nests the network name under data.networks (a list) not network.name and data.scenarios (a list) rather than load.scenarios.
    # Set them defensively so this works whether the pulled config is the app's VLD default or an older schema.
    if "data" in config:
        config["data"]["networks"] = ["scenario_33radial"]
        config["data"]["scenarios"] = [10000]
    else:  # legacy gridfm-datakit schema
        config.setdefault("network", {})["name"] = "scenario_33radial"
        config.setdefault("load", {})["scenarios"] = 10000
        config.setdefault("topology_perturbation", {})["n_topology_variants"] = 2
    return config


def generate_vld_test_data(
    config_path: str = "integrationtests/default_vld.yaml",
) -> None:
    """Generate VLD test data for scenario_33radial.
    The app has a single task (VoltageLossDetection) and no OPF mode, so there
    is only one generator (the old PF/OPF split is gone).
    """
    config = _base_config()
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"VLD config written to {config_path}")
    execute_and_live_output(f"gridfm_app_dist_datakit generate --configs {config_path}")


if __name__ == "__main__":
    generate_vld_test_data()
