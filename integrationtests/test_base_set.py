"""VLD integration test for gridfm-app-distribution.

Adapted from the original gridfm-graphkit PF/OPF integration harness.
The calibration machinery (compute_bounds / write_baseline / read_baseline / environment fingerprinting / --calibrate,--ci,--pad) is TASK-AGNOSTIC and is kept verbatim;
The following changes are made to the measurements:

  * Single task: VoltageLossDetection (VLD), replacing PF + OPF.
  * CLI is `python -m gridfm_app_dist ... the old `--mp_context` / `--deterministic` flags do NOT exist here.
  * Metrics are computed FROM THE PREDICTIONS PARQUET the app writes (`<grid>_predictions.parquet` with columns scenario, bus, vm_pred, status_prob, status_pred, status_true)
  * Config is the app's own scripts/config/default.yaml (task_name: VoltageLossDetection);
  we shrink training.epochs and model.hidden_size for a fast run -- both keys verified to exist in that config.

Vm error is NOT asserted here: the `predict` output parquet carries status_true but no vm_true (see vld_task.predict_step),
so only status classification metrics are computable from the parquet.
If you want Vm error too, use the`evaluate` path (test_step logs OFF Vm MAE / ON Vm RMSE via MLflow) instead.
"""

import pytest
import subprocess
import os
import glob
import json
import platform
import warnings
import importlib.metadata as importlib_metadata
import pandas as pd
import yaml
import shutil
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score


# ----------------------------------------------------------------------------
# Environment fingerprint (only the package list is app-specific)
# ----------------------------------------------------------------------------
def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def environment_fingerprint() -> dict:
    fp = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "gpus": [],
        "cuda": None,
        "cudnn": None,
    }
    for pkg in ("torch", "numpy", "scipy", "gridfm_app_dist", "gridfm_datakit"):
        try:
            fp[f"{pkg}_version"] = importlib_metadata.version(pkg.replace("_", "-"))
        except importlib_metadata.PackageNotFoundError:
            fp[f"{pkg}_version"] = None
    try:
        import torch

        fp["cuda"] = torch.version.cuda
        if torch.backends.cudnn.is_available():
            fp["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            fp["gpus"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    return fp


def warn_on_fingerprint_mismatch(saved: dict, current: dict) -> None:
    if not saved:
        warnings.warn(
            "Calibration baseline has no environment fingerprint; cannot verify it was calibrated on this machine.",
            stacklevel=2,
        )
        return
    diffs = []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key) != current.get(key):
            diffs.append(
                f"{key}: baseline={saved.get(key)!r}  current={current.get(key)!r}",
            )
    if diffs:
        warnings.warn(
            "Calibration baseline was measured in a DIFFERENT environment; metric bounds may not hold here. "
            "Re-run `pytest integrationtests --calibrate` on the unchanged code to recalibrate on this machine before making changes. "
            "Differing fields:\n" + "\n".join(diffs),
            stacklevel=2,
        )


# ----------------------------------------------------------------------------
# Calibration baseline I/O
# ----------------------------------------------------------------------------
DEFAULT_BASELINE_PATH = os.path.join(
    os.path.dirname(__file__), "calibration_baseline.json",
)


def baseline_path() -> str:
    return os.environ.get("GRIDFM_CALIBRATION_BASELINE", DEFAULT_BASELINE_PATH)


def compute_bounds(all_runs, metric_keys, confidence_interval, pad) -> dict:
    n = len(all_runs)
    alpha_half = (1 + confidence_interval) / 2
    t_crit = stats.t.ppf(alpha_half, df=max(n - 1, 1))
    bounds = {}
    for key in metric_keys:
        values = [run[key] for run in all_runs if key in run]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        me = t_crit * std / np.sqrt(len(arr))
        half_width = max(me, pad * abs(mean))
        bounds[key] = (mean - half_width, mean + half_width)
    return bounds


def write_baseline(test_key: str, bounds: dict) -> None:
    path = baseline_path()
    data = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    data.setdefault("bounds", {})
    data.setdefault("fingerprints", {})
    data.pop("fingerprint", None)
    data["bounds"][test_key] = {k: list(v) for k, v in bounds.items()}
    data["fingerprints"][test_key] = environment_fingerprint()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"\nCalibrated bounds for '{test_key}' written to {path}")


def read_baseline(test_key: str) -> dict:
    path = baseline_path()
    assert os.path.exists(path), (
        f"No calibration baseline at {path}. Bounds are machine-specific -- "
        f"calibrate on THIS machine first BEFORE making changes, e.g.:\n"
        f"    pytest integrationtests --calibrate -s"
    )
    with open(path, "r") as f:
        data = json.load(f)
    bounds = data.get("bounds", {})
    assert test_key in bounds, (
        f"Baseline {path} has no entry for '{test_key}'. Re-run calibration on "
        f"this machine BEFORE making changes: pytest integrationtests --calibrate -s"
    )
    saved_fp = data.get("fingerprints", {}).get(test_key, data.get("fingerprint", {}))
    warn_on_fingerprint_mismatch(saved_fp, environment_fingerprint())
    return {k: tuple(v) for k, v in bounds[test_key].items()}


def execute_and_live_output(cmd) -> None:
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(cmd, text=True, shell=True, check=True, env=env)


# ----------------------------------------------------------------------------
# VLD metric computation FROM THE PREDICTIONS PARQUET (the key rewrite)
# ----------------------------------------------------------------------------
GRID_NAME = "scenario_33radial"

# Metrics computed from status_pred / status_true columns of the predictions
# parquet. These are the self-contained VLD signal. (Vm error is not here
# because the predict parquet has no vm_true column -- see module docstring.)
VLD_METRIC_KEYS = [
    "status_accuracy",
    "status_f1",
    "status_precision",
    "status_recall",
]


def collect_metrics_from_parquet(output_dir: str) -> dict:
    """Compute VLD status-classification metrics from the predictions parquet.

    Replaces the old `collect_metrics_from_log` (which read a
    case14_ieee_metrics.csv this app never writes). Reads
    `<output_dir>/<GRID_NAME>_predictions.parquet` produced by the app's
    predict/save_output path.
    """
    parquet_path = os.path.join(output_dir, f"{GRID_NAME}_predictions.parquet")
    assert os.path.exists(parquet_path), (
        f"Predictions parquet not found: {parquet_path}. Did the predict step "
        f"run with --output_path {output_dir}?"
    )
    df = pd.read_parquet(parquet_path)
    for col in ("status_pred", "status_true"):
        assert col in df.columns, f"'{col}' missing from {parquet_path}"

    y_true = df["status_true"].to_numpy().astype(int)
    y_pred = df["status_pred"].to_numpy().astype(int)

    return {
        "status_accuracy": float((y_true == y_pred).mean()),
        # zero_division=0 keeps a degenerate all-one-class fold from raising.
        "status_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "status_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "status_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def print_calibration_stats(all_runs, metric_keys, bounds, confidence_interval=0.995):
    n = len(all_runs)
    ci_pct = f"{confidence_interval * 100:g}"
    col_w = max(len(k) for k in metric_keys) + 2
    header = (
        f"  {'Metric':<{col_w}}  {'Mean':>10}  {'Std(ddof=1)':>12}  "
        f"{f'Bound {ci_pct}% lo':>12}  {f'Bound {ci_pct}% hi':>12}"
    )
    print(f"\n===== Calibration Results (n={n}, CI={confidence_interval}) =====")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in metric_keys:
        values = [run[key] for run in all_runs if key in run]
        if not values or key not in bounds:
            print(f"  {key:<{col_w}}  {'no data':>10}")
            continue
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        lo, hi = bounds[key]
        print(
            f"  {key:<{col_w}}  {mean:>10.4f}  {std:>12.4f}  {lo:>12.4f}  {hi:>12.4f}",
        )
    print("=" * (len(header)) + "\n")


# ----------------------------------------------------------------------------
# Config preparation: shrink epochs + hidden_size for a fast run.
# Keys verified against scripts/config/default.yaml:
#   training.epochs (int), model.hidden_size (int), task.task_name.
# ----------------------------------------------------------------------------
VLD_CONFIG_PATH = os.environ.get("GRIDFM_VLD_CONFIG", "scripts/config/default.yaml")
VLD_DATA_PATH = os.environ.get("GRIDFM_VLD_DATA", "tests/data")
LOG_PATH = os.environ.get("LOG_PATH", "integrationtests/logs")


def prepare_training_config():
    with open(VLD_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Fail loudly if the expected keys are absent -- editing a missing key would silently train a full-size model and invalidate the fast test.
    assert "training" in config and "epochs" in config["training"], (
        f"{VLD_CONFIG_PATH}: expected key training.epochs not found"
    )
    assert "model" in config and "hidden_size" in config["model"], (
        f"{VLD_CONFIG_PATH}: expected key model.hidden_size not found"
    )
    assert config.get("task", {}).get("task_name") == "VoltageLossDetection", (
        f"{VLD_CONFIG_PATH}: task.task_name must be VoltageLossDetection"
    )

    config["training"]["epochs"] = 20
    config["model"]["hidden_size"] = 12

    with open(VLD_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"VLD training config updated: epochs=20, hidden_size=12 ({VLD_CONFIG_PATH})")
    return VLD_CONFIG_PATH


@pytest.fixture
def cleanup_test_artifacts():
    """Back up the edited config and remove generated artifacts afterwards."""
    backup_config = VLD_CONFIG_PATH + ".bak"
    if os.path.exists(VLD_CONFIG_PATH):
        shutil.copy2(VLD_CONFIG_PATH, backup_config)

    yield

    if os.path.exists(backup_config):
        shutil.move(backup_config, VLD_CONFIG_PATH)

    for d in ["logs", "predict_out"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def _find_latest(pattern: str) -> str:
    hits = glob.glob(pattern)
    assert hits, f"No file matched {pattern}"
    return max(hits, key=os.path.getmtime)


def test_train_vld(cleanup_test_artifacts, calibrate_runs, ci_level, calibrate_pad):
    """Integration test for VoltageLossDetection.
    Steps:
      1. (Data expected at data_out/ -- generate via generate_test_data.py)
      2. Train a short VLD model with the app CLI.
      3. Run predict to produce the predictions parquet.
      4. Compute status-classification metrics from that parquet and assert they fall within calibrated bounds.
    Pass --calibrate (default 5 runs; --calibrate N for another count) to calibrate bounds instead of asserting.
    """
    n_runs = max(calibrate_runs, 1)
    data_dir = VLD_DATA_PATH
    log_dir = LOG_PATH

    assert os.path.isdir(data_dir) and os.listdir(data_dir), (
        f"VLD data dir '{data_dir}' missing/empty. Generate it first: python integrationtests/generate_test_data.py\n"
    )

    training_config_path = prepare_training_config()
    all_runs = []

    for run_i in range(n_runs):
        print(f"\n--- VLD run {run_i + 1}/{n_runs} ---")
        # 1) TRAIN. Note: no --mp_context / --deterministic (not app flags).
        execute_and_live_output(
            f"python -m gridfm_app_dist train "
            f"--config {training_config_path} "
            f"--data_path {data_dir} "
            f"--exp_name exp_vld "
            f"--run_name run{run_i + 1} "
            f"--log_dir {log_dir}",
        )
        # 2) PREDICT -> writes <GRID_NAME>_predictions.parquet to --output_path.
        model_ckpt = _find_latest("logs/**/*.ckpt")
        stats_pt = _find_latest("logs/**/normalizer_stats.pt")
        predict_out = "predict_out"
        os.makedirs(predict_out, exist_ok=True)
        execute_and_live_output(
            f"python -m gridfm_app_dist predict "
            f"--config {training_config_path} "
            f"--data_path {data_dir}/ "
            f"--model_path {model_ckpt} "
            f"--normalizer_stats {stats_pt} "
            f"--output_path {predict_out} "
            f"--exp_name exp_vld --run_name pred{run_i + 1} --log_dir logs",
        )
        metrics = collect_metrics_from_parquet(predict_out)
        all_runs.append(metrics)

    if calibrate_runs > 0:
        bounds = compute_bounds(all_runs, VLD_METRIC_KEYS, ci_level, calibrate_pad)
        print_calibration_stats(
            all_runs, VLD_METRIC_KEYS, bounds, confidence_interval=ci_level,
        )
        write_baseline("vld", bounds)
        return

    checks = read_baseline("vld")

    MAX_RETRIES = 5
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"\n--- VLD retry {attempt}/{MAX_RETRIES} after interval failure ---")
            execute_and_live_output(
                f"python -m gridfm_app_dist train "
                f"--config {training_config_path} --data_path {data_dir}/ "
                f"--exp_name exp_vld --run_name retry{attempt} --log_dir logs",
            )
            model_ckpt = _find_latest("logs/**/*.ckpt")
            stats_pt = _find_latest("logs/**/normalizer_stats.pt")
            execute_and_live_output(
                f"python -m gridfm_app_dist predict "
                f"--config {training_config_path} --data_path {data_dir}/ "
                f"--model_path {model_ckpt} --normalizer_stats {stats_pt} "
                f"--output_path predict_out "
                f"--exp_name exp_vld --run_name predretry{attempt} --log_dir logs",
            )
            metrics = collect_metrics_from_parquet("predict_out")
        else:
            metrics = all_runs[0]

        try:
            for metric_name, (lo, hi) in checks.items():
                assert metric_name in metrics, f"Metric '{metric_name}' not computed"
                value = metrics[metric_name]
                assert lo <= value <= hi, (
                    f"Metric '{metric_name}' value {value} outside calibrated "
                    f"bounds [{lo}, {hi}]"
                )
                print(f"{metric_name}: {value} within [{lo}, {hi}] (attempt {attempt})")
            last_error = None
            break
        except AssertionError as e:
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            last_error = e

    if last_error is not None:
        raise last_error
