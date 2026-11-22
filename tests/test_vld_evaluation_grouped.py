"""
The grouped evaluator must give the same results as the in-memory evaluator
(``vld_evaluation.evaluate_vld_predictions``) on a file small enough for both.
"""

import math

import numpy as np
import pandas as pd
import pytest

from gridfm_app_dist.io.vld_evaluation import evaluate_vld_predictions
from gridfm_app_dist.io.vld_evaluation_grouped import evaluate_vld_file_grouped

# ROC-AUC, AP and ECE come from a 65,536-bin histogram; everything else is exact.
APPROX = {"roc_auc": 2e-4, "ap_deenergised": 5e-4, "ap_energised": 5e-4, "ece": 2e-3}
NOT_COMPUTED = (
    "roc_auc_ci95",
    "ap_deenergised_ci95",
)  # no bootstrap for these by design


def _predictions(n_scen=600, n_bus=23, seed=0, scattered=False, with_pred=True):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_scen) * 3 + 7  # unsorted, with gaps
    bus = np.tile(np.arange(n_bus), n_scen)
    cut = np.repeat(rng.integers(1, n_bus + 4, n_scen), n_bus)
    y = (bus < cut).astype(np.float32)
    logit = np.where(y == 1, 3.0, -3.0) + rng.normal(0, 2.5, y.size)
    logit[rng.random(y.size) < 0.05] *= 8  # saturated probabilities (exact 0 / 1)
    with np.errstate(over="ignore"):
        prob = (1 / (1 + np.exp(-logit.astype(np.float32)))).astype(np.float32)
    thr = np.float32(0.0924)
    df = pd.DataFrame(
        {"scenario": np.repeat(ids, n_bus), "bus": bus, "status_prob": prob},
    )
    if with_pred:
        pred = (prob.astype(np.float64) >= thr).astype(np.float32)
        flip = rng.random(y.size) < 0.002
        pred[flip] = 1 - pred[flip]
        df["status_pred"] = pred
    df["threshold"] = thr
    df["status_true"] = y
    df["vm_pred"] = (y * 0.98 + rng.normal(0, 0.2, y.size)).astype(np.float32)
    df["vm_true"] = (y * (1 + rng.normal(0, 0.02, y.size))).astype(np.float32)
    if scattered:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def _assert_same(ref, got):
    for k, v in ref.items():
        if k.startswith(NOT_COMPUTED):
            continue
        assert k in got, k
        if isinstance(v, float) and math.isnan(v):
            assert math.isnan(got[k]), k
        elif k in APPROX:
            assert abs(got[k] - v) <= APPROX[k], (k, v, got[k])
        else:
            assert got[k] == pytest.approx(v, rel=1e-9, abs=1e-12), (k, v, got[k])


@pytest.mark.parametrize(
    "fmt, scattered, with_pred",
    [("parquet", False, True), ("parquet", True, True), ("csv", False, False)],
)
def test_grouped_matches_in_memory(tmp_path, fmt, scattered, with_pred):
    df = _predictions(scattered=scattered, with_pred=with_pred)
    path = tmp_path / f"preds.{fmt}"
    if fmt == "parquet":
        df.to_parquet(path, index=False, row_group_size=2000)  # many row groups
        df_in = pd.read_parquet(path)
    else:
        df.to_csv(path, index=False)
        df_in = pd.read_csv(path)
    ref = evaluate_vld_predictions(df_in, str(tmp_path / "ref"), n_boot=50, seed=0)
    got = evaluate_vld_file_grouped(
        str(path),
        str(tmp_path / "grouped"),
        group_size=130,
        batch_rows=1777,
        n_boot=50,
        seed=0,
    )
    _assert_same(ref, got)
    assert got["n_groups"] == math.ceil(600 / 130)
    for name in (
        "confusion_matrix",
        "roc_pr",
        "per_bus",
        "group_variation",
        "voltage_status",
    ):
        assert (tmp_path / "grouped" / f"{name}.png").exists(), name


def test_trial_run_then_resume(tmp_path):
    df = _predictions(seed=3)
    path = tmp_path / "preds.parquet"
    df.to_parquet(path, index=False, row_group_size=3000)
    out = str(tmp_path / "grouped")
    part = evaluate_vld_file_grouped(
        str(path),
        out,
        group_size=200,
        max_groups=1,
        n_boot=20,
    )
    assert part["groups_evaluated"] == 1 and part["n_scenarios"] == 200
    full = evaluate_vld_file_grouped(str(path), out, group_size=200, n_boot=20)
    fresh = evaluate_vld_file_grouped(
        str(path),
        str(tmp_path / "fresh"),
        group_size=200,
        n_boot=20,
        resume=False,
    )
    assert full == fresh
    with pytest.raises(RuntimeError):  # different group size in the same outdir
        evaluate_vld_file_grouped(str(path), out, group_size=150, n_boot=20)
