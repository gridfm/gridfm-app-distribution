"""
Evaluation metrics and figures for Voltage-Loss-Detection (VLD) predictions.

Input is the dataframe written by ``VoltageLossDetectionTask`` (test CSV or prediction parquet) with columns
    scenario, bus, status_prob, status_pred, status_true, threshold, vm_pred[, vm_true]

where
``status_prob`` is P(energised),
``status_true`` is 1 = energised and 0 = de-energised, and
``threshold`` is the decision threshold the model used, 1 / (1 + w_off), derived by ``VLDTopologyLoss``.

What is reported
----------------
Three operating points are compared side by side:

  model      the model's own threshold (``threshold`` column);
  cost       the threshold minimising a weighted error-rate cost, chosen on a CALIBRATION set (a validation dataframe, or a scenario-wise split of the input when none is given) and applied to the test set;
  naive      0.5, as a reference only.

For each: balanced accuracy, MCC, recall and precision of both classes.
Threshold-free: ROC-AUC and average precision with the DE-ENERGISED class as positive (AP on the minority class is the number that moves when the detector improves).
Calibration: Brier score and expected calibration error.
Scenario level: exact-match rate and per-scenario missed outages / false alarms.
Uncertainty: cluster-bootstrap 95 % intervals over scenarios (buses within a scenario are not independent).

Cost weights are named by what they mean, not by FN/FP:
    c_missed_outage   bus predicted energised, actually de-energised
    c_false_alarm     bus predicted de-energised, actually energised
cost(theta) = c_missed_outage * missed_outage_rate + c_false_alarm * false_alarm_rate
with rates normalised per class so the class imbalance does not drive the cost.

Usage from code (what the task and the CLI do)::
    from gridfm_app_dist.io.vld_evaluation import evaluate_vld_predictions
    metrics = evaluate_vld_predictions(df, out_dir, model_threshold=theta)

Usage from the shell::
    python -m gridfm_app_dist.io.vld_evaluation preds.parquet --outdir figs --calib val_preds.parquet] [--cost-weights 1:1,3:1,5:1]
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Plot style: applied through rc_context inside each figure so importing this
# module never changes the style of other figures in the process.
# ----------------------------------------------------------------------------
_RC = {
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}
C_ON = "#d9731a"     # energised
C_OFF = "#1f5fa8"    # de-energised
C_MODEL = "#1a1a1a"
C_COST = "#7b3fa0"
C_NAIVE = "#9a9a9a"
_GRID = np.linspace(0.005, 0.995, 199)


def _plt():
    import matplotlib
    if os.environ.get("DISPLAY", "") == "" and matplotlib.get_backend().lower() not in ("agg","pdf","svg"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------------------
# Core metrics
# ----------------------------------------------------------------------------
def _confusion(y: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TN, FP, FN, TP) with ENERGISED (1) as positive."""
    y = y.astype(bool)
    p = pred.astype(bool)
    tp = int(np.sum(y & p))
    tn = int(np.sum(~y & ~p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    return tn, fp, fn, tp


def operating_point_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Threshold-dependent metrics for one set of hard predictions."""
    from sklearn.metrics import matthews_corrcoef
    tn, fp, fn, tp = _confusion(y, pred)
    n_on = tp + fn
    n_off = tn + fp
    on_recall = tp / n_on if n_on else float("nan")      # 1 - false-alarm rate
    off_recall = tn / n_off if n_off else float("nan")   # outage detection rate
    on_precision = tp / (tp + fp) if (tp + fp) else float("nan")
    off_precision = tn / (tn + fn) if (tn + fn) else float("nan")
    off_f1 = (2 * off_precision * off_recall / (off_precision + off_recall) if (off_precision + off_recall) > 0 else float("nan"))
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": float(np.nanmean([on_recall, off_recall])),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) == 2 else float("nan"),
        "off_recall": off_recall,          # detected outages / true outages
        "off_precision": off_precision,
        "off_f1": off_f1,
        "on_recall": on_recall,            # 1 - false de-energisation rate
        "on_precision": on_precision,
        "missed_outage_rate": 1.0 - off_recall if n_off else float("nan"),
        "false_alarm_rate": 1.0 - on_recall if n_on else float("nan"),
        "TP_on": tp, "TN_off": tn, "FP_pred_on_true_off": fp, "FN_pred_off_true_on": fn,
    }


def threshold_free_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """ROC-AUC / AP (de-energised as positive), Brier, ECE."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {"brier": float(np.mean((prob - y) ** 2))}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["ap_energised"] = float(average_precision_score(y, prob))
        out["ap_deenergised"] = float(average_precision_score(1 - y, 1.0 - prob))
    out["ece"] = expected_calibration_error(y, prob)
    return out


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE with quantile bins (robust to the bimodal probabilities of a confident model)."""
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(ece)


def cost_curve(
    y: np.ndarray,
    prob: np.ndarray,
    c_missed_outage: float,
    c_false_alarm: float,
    grid: np.ndarray = _GRID,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rate-normalised weighted cost for every threshold on the grid."""
    on = y.astype(bool)
    n_on = max(on.sum(), 1)
    n_off = max((~on).sum(), 1)
    # vectorised over the grid
    pred_on = prob[None, :] >= grid[:, None]            # [G, N]
    missed_outage = np.sum(pred_on & ~on[None, :], axis=1) / n_off
    false_alarm = np.sum(~pred_on & on[None, :], axis=1) / n_on
    return grid, c_missed_outage * missed_outage + c_false_alarm * false_alarm


def cost_optimal_threshold(y, prob, c_missed_outage, c_false_alarm) -> float:
    grid, cost = cost_curve(y, prob, c_missed_outage, c_false_alarm)
    return float(grid[int(np.argmin(cost))])


def metric_sweep(y, prob, grid: np.ndarray = _GRID) -> Dict[str, np.ndarray]:
    """Balanced accuracy and MCC over the threshold grid (vectorised)."""
    on = y.astype(bool)
    n_on = max(on.sum(), 1)
    n_off = max((~on).sum(), 1)
    pred_on = prob[None, :] >= grid[:, None]
    tp = np.sum(pred_on & on[None, :], axis=1).astype(float)
    tn = np.sum(~pred_on & ~on[None, :], axis=1).astype(float)
    fp = n_off - tn
    fn = n_on - tp
    bacc = 0.5 * (tp / n_on + tn / n_off)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.where(denom > 0, denom, 1), 0.0)
    return {"balanced_accuracy": bacc, "mcc": mcc}


# ----------------------------------------------------------------------------
# Scenario-level metrics and cluster bootstrap
# ----------------------------------------------------------------------------
def scenario_metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, object]:
    y = df["status_true"].to_numpy().astype(int)
    p = df[pred_col].to_numpy().astype(int)
    g = df["scenario"].to_numpy()
    err = p != y
    missed = (p == 1) & (y == 0)
    alarm = (p == 0) & (y == 1)
    tab = pd.DataFrame({"scenario": g, "err": err, "missed": missed, "alarm": alarm}).groupby("scenario", sort=False).sum()
    return {
        "n_scenarios": int(len(tab)),
        "scenario_exact_match_rate": float((tab["err"] == 0).mean()),
        "mean_missed_outages_per_scenario": float(tab["missed"].mean()),
        "mean_false_alarms_per_scenario": float(tab["alarm"].mean()),
        "_per_scenario": tab,  # for plotting; stripped before JSON
    }


def cluster_bootstrap(
    df: pd.DataFrame,
    threshold: float,
    n_boot: int = 1000,
    seed: int = 0,
    keys: Sequence[str] = ("balanced_accuracy", "mcc", "off_recall", "on_recall", "ap_deenergised"),
) -> Dict[str, Tuple[float, float]]:
    """95 % intervals by resampling SCENARIOS with replacement."""
    rng = np.random.default_rng(seed)
    scen = df["scenario"].to_numpy()
    uniq, inv = np.unique(scen, return_inverse=True)
    y_all = df["status_true"].to_numpy().astype(int)
    p_all = df["status_prob"].to_numpy().astype(float)
    # row indices grouped per scenario
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    groups = [order[bounds[i]: bounds[i + 1]] for i in range(len(uniq))]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        y, pr = y_all[idx], p_all[idx]
        if len(np.unique(y)) < 2:
            continue
        m = operating_point_metrics(y, (pr >= threshold).astype(int))
        m.update(threshold_free_metrics(y, pr))
        for k in keys:
            samples[k].append(m.get(k, np.nan))
    return {
        k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for k, v in samples.items()
        if len(v)
    }


def split_by_scenario(df: pd.DataFrame, calib_frac: float = 0.3, seed: int = 0):
    """Scenario-wise split used only when no separate calibration set is given."""
    rng = np.random.default_rng(seed)
    scen = np.unique(df["scenario"].to_numpy())
    rng.shuffle(scen)
    n_cal = max(1, int(round(calib_frac * len(scen))))
    cal = set(scen[:n_cal].tolist())
    is_cal = df["scenario"].isin(cal).to_numpy()
    return df[is_cal], df[~is_cal]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
@dataclass
class OperatingPoint:
    name: str
    threshold: float
    color: str
    ls: str


def plot_confusions(y, prob, points: Sequence[OperatingPoint], path):
    plt = _plt()
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(points), figsize=(2.4 * len(points), 2.5), squeeze=False)
        for ax, op in zip(axes[0], points):
            tn, fp, fn, tp = _confusion(y, (prob >= op.threshold).astype(int))
            cm = np.array([[tn, fp], [fn, tp]])  # rows: true off, true on; cols: pred off, pred on
            cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}\n{cmn[i, j]:.1%}", ha="center", va="center",
                            fontsize=7, color="white" if cmn[i, j] > 0.55 else "black")
            ax.set_xticks([0, 1], ["off", "on"]); ax.set_yticks([0, 1], ["off", "on"])
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"{op.name}  $\\theta$={op.threshold:.3f}")
            ax.grid(False)
        fig.savefig(path); plt.close(fig)


def plot_roc_pr(y, prob, points: Sequence[OperatingPoint], path):
    from sklearn.metrics import (average_precision_score, precision_recall_curve,roc_auc_score, roc_curve)
    plt = _plt()
    y_off, p_off = 1 - y, 1.0 - prob  # de-energised as positive
    with plt.rc_context(_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
        fpr, tpr, thr = roc_curve(y_off, p_off)
        a1.plot(fpr, tpr, color=C_OFF, lw=1.6, label=f"AUC = {roc_auc_score(y_off, p_off):.3f}")
        a1.plot([0, 1], [0, 1], "--", color=C_NAIVE, lw=0.8)
        prec, rec, thr_pr = precision_recall_curve(y_off, p_off)
        a2.plot(rec, prec, color=C_OFF, lw=1.6,
                label=f"AP = {average_precision_score(y_off, p_off):.3f}")
        a2.axhline(y_off.mean(), ls="--", color=C_NAIVE, lw=0.8, label=f"prevalence {y_off.mean():.2f}")
        for op in points:
            pred_off = (prob < op.threshold)
            tn, fp, fn, tp = _confusion(y, (~pred_off).astype(int))
            # in the off-positive frame: TPR_off = tn/(tn+fp), FPR_off = fn/(fn+tp)
            tpr_o = tn / max(tn + fp, 1); fpr_o = fn / max(fn + tp, 1)
            prec_o = tn / max(tn + fn, 1)
            a1.plot(fpr_o, tpr_o, "o", color=op.color, ms=5, mec="white", mew=0.6, label=op.name)
            a2.plot(tpr_o, prec_o, "o", color=op.color, ms=5, mec="white", mew=0.6, label=op.name)
        a1.set_xlabel("false positive rate (energised called off)"); a1.set_ylabel("true positive rate (outage detected)")
        a1.set_title("ROC, de-energised as positive"); a1.legend(loc="lower right")
        a2.set_xlabel("recall (outage detected)"); a2.set_ylabel("precision")
        a2.set_title("Precision-recall, de-energised as positive"); a2.legend(loc="lower left")
        fig.savefig(path); plt.close(fig)


def plot_cost_and_sweep(y, prob, points: Sequence[OperatingPoint], weights, path):
    plt = _plt()
    sweep = metric_sweep(y, prob)
    with plt.rc_context(_RC):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.2), sharex=True)
        shades = ["#7b3fa0", "#a46fc4", "#c9a6dd", "#e2cfee"]
        for (cmo, cfa), col in zip(weights, shades):
            grid, cost = cost_curve(y, prob, cmo, cfa)
            a1.plot(grid, cost, color=col, lw=1.4, label=f"$c_{{miss}}$:$c_{{alarm}}$ = {cmo:g}:{cfa:g}")
        a1.set_ylabel("weighted error-rate cost"); a1.legend(loc="upper center", ncol=2)
        a2.plot(grid, sweep["balanced_accuracy"], color=C_OFF, lw=1.4, label="balanced accuracy")
        a2.plot(grid, sweep["mcc"], color=C_ON, lw=1.4, label="MCC")
        a2.set_ylabel("score"); a2.set_xlabel(r"decision threshold $\theta$ on P(energised)")
        a2.legend(loc="upper right", ncol=1)
        for ax in (a1, a2):
            for op in points:
                ax.axvline(op.threshold, color=op.color, ls=op.ls, lw=1.0)
        a1.legend(loc="upper right", ncol=1)
        ylo = a2.get_ylim()[0]
        for op in points:
            a2.text(op.threshold, ylo, f" {op.name}", color=op.color, fontsize=6.5,va="bottom", ha="left", rotation=90)
        fig.savefig(path); plt.close(fig)


def plot_reliability(y, prob, path, n_bins=10):
    plt = _plt()
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            xs.append(prob[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax2 = ax.twinx()
        ax2.bar(xs, ns, width=0.04, color="#e6e6e6", zorder=0)
        ax2.set_ylabel("buses per bin", color=C_NAIVE); ax2.tick_params(axis="y", colors=C_NAIVE)
        ax2.grid(False); ax2.spines["right"].set_visible(True)
        ax.plot([0, 1], [0, 1], "--", color=C_NAIVE, lw=0.8, label="perfect")
        ax.plot(xs, ys, "o-", color=C_OFF, lw=1.4, ms=4, label="model", zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted P(energised)"); ax.set_ylabel("observed energised fraction")
        ax.set_title(f"Reliability  (ECE = {expected_calibration_error(y, prob):.3f}, "
                     f"Brier = {np.mean((prob - y) ** 2):.3f})")
        ax.legend(loc="upper left"); ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        fig.savefig(path); plt.close(fig)


def plot_scenario_errors(per_scenario: pd.DataFrame, name: str, path):
    plt = _plt()
    with plt.rc_context(_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6), sharey=True)
        mx = int(max(per_scenario["missed"].max(), per_scenario["alarm"].max(), 1))
        bins = np.arange(-0.5, mx + 1.5, 1)
        a1.hist(per_scenario["missed"], bins=bins, color=C_OFF, edgecolor="white")
        a1.set_xlabel("missed outages per scenario"); a1.set_ylabel("scenarios")
        a2.hist(per_scenario["alarm"], bins=bins, color=C_ON, edgecolor="white")
        a2.set_xlabel("false alarms per scenario")
        exact = float((per_scenario["err"] == 0).mean())
        fig.suptitle(f"Per-scenario errors at {name} threshold  (exact match {exact:.1%})", fontsize=9)
        fig.savefig(path); plt.close(fig)


def plot_per_bus(df: pd.DataFrame, pred_col: str, path):
    plt = _plt()
    y = df["status_true"].to_numpy().astype(int); p = df[pred_col].to_numpy().astype(int)
    b = df["bus"].to_numpy()
    buses = np.unique(b)
    off_rec, on_rec, off_n = [], [], []
    for bus in buses:
        m = b == bus
        yo, po = y[m] == 0, p[m] == 0
        off_rec.append(np.sum(yo & po) / yo.sum() if yo.sum() else np.nan)
        on_rec.append(np.sum(~yo & ~po) / (~yo).sum() if (~yo).sum() else np.nan)
        off_n.append(yo.mean())
    x = np.arange(len(buses)); w = 0.4
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        ax.bar(x - w / 2, off_rec, w, color=C_OFF, label="off-recall (outage detected)")
        ax.bar(x + w / 2, on_rec, w, color=C_ON, label="on-recall (no false alarm)")
        ax.plot(x, off_n, "k.", ms=3, label="outage frequency of bus")
        ax.set_xticks(x); ax.set_xticklabels(buses, fontsize=6, rotation=90)
        ax.set_ylim(0, 1.02); ax.set_xlabel("bus (feeder order)"); ax.set_ylabel("recall")
        ax.set_title("Per-bus detection performance"); ax.legend(loc="lower left", ncol=3)
        fig.savefig(path); plt.close(fig)


def plot_voltage_status(df: pd.DataFrame, threshold: float, path):
    """Voltage-status consistency: predicted Vm against P(energised)."""
    plt = _plt()
    y = df["status_true"].to_numpy().astype(int)
    prob = df["status_prob"].to_numpy().astype(float)
    vm = df["vm_pred"].to_numpy().astype(float)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(y), size=min(len(y), 20000), replace=False)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        for cls, col, lab in ((0, C_OFF, "true de-energised"), (1, C_ON, "true energised")):
            m = keep[y[keep] == cls]
            ax.scatter(prob[m], vm[m], s=3, alpha=0.35, color=col, label=lab, linewidths=0)
        ax.axvline(threshold, color=C_MODEL, ls="--", lw=0.9, label=rf"$\theta$ = {threshold:.3f}")
        ax.axhline(0.5, color=C_NAIVE, ls=":", lw=0.8)
        ax.set_xlabel("P(energised)"); ax.set_ylabel(r"predicted $\hat V$ (p.u.)")
        ax.set_title("Voltage-status consistency")
        ax.text(0.02, 0.96, "status off\nvoltage on", transform=ax.transAxes, fontsize=6, va="top", color=C_NAIVE)
        ax.text(0.98, 0.04, "status on\nvoltage off", transform=ax.transAxes, fontsize=6, va="bottom", ha="right", color=C_NAIVE)
        ax.legend(loc="center right", markerscale=3)
        fig.savefig(path); plt.close(fig)
    # quadrant counts as metrics
    pred_on = prob >= threshold
    v_on = vm >= 0.5
    return {
        "inconsistent_status_on_voltage_off": float(np.mean(pred_on & ~v_on)),
        "inconsistent_status_off_voltage_on": float(np.mean(~pred_on & v_on)),
    }


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def evaluate_vld_predictions(
    df: pd.DataFrame,
    out_dir: str,
    model_threshold: Optional[float] = None,
    calib_df: Optional[pd.DataFrame] = None,
    cost_weights: Iterable[Tuple[float, float]] = ((1.0, 1.0), (3.0, 1.0), (5.0, 1.0)),
    primary_cost_weights: Tuple[float, float] = (3.0, 1.0),
    calib_frac: float = 0.3,
    n_boot: int = 1000,
    seed: int = 0,
    prefix: str = "",
    fig_format: str = "png",
) -> Dict[str, float]:
    """
    Evaluate VLD predictions, write figures + metrics.json to ``out_dir`` and return a flat dict of metrics (suitable for ``logger.log_metrics``).
    ``model_threshold`` defaults to the ``threshold`` column of ``df``. The cost-optimal threshold is chosen on ``calib_df`` when given;
    otherwise the input is split by scenario (``calib_frac`` of scenarios for calibration) and all reported metrics refer to the remaining scenarios,
    so the cost threshold is never evaluated on the data it was selected on.
    """
    os.makedirs(out_dir, exist_ok=True)
    cost_weights = [tuple(map(float, w)) for w in cost_weights]
    df = df.copy()
    df["status_true"] = df["status_true"].astype(int)
    df["status_prob"] = df["status_prob"].astype(float)

    if model_threshold is None:
        model_threshold = float(df["threshold"].iloc[0]) if "threshold" in df else 0.5

    note = {}
    if calib_df is None:
        calib_df, df = split_by_scenario(df, calib_frac=calib_frac, seed=seed)
        note["calibration"] = (
            f"no separate calibration set given; {calib_df['scenario'].nunique()} scenarios "
            f"({calib_frac:.0%}) held out from the input for threshold selection"
        )
    else:
        note["calibration"] = f"separate calibration set with {calib_df['scenario'].nunique()} scenarios"

    y = df["status_true"].to_numpy()
    prob = df["status_prob"].to_numpy()
    y_cal = calib_df["status_true"].to_numpy().astype(int)
    p_cal = calib_df["status_prob"].to_numpy().astype(float)

    cmo, cfa = primary_cost_weights
    cost_thr = cost_optimal_threshold(y_cal, p_cal, cmo, cfa)
    points = [
        OperatingPoint("model", float(model_threshold), C_MODEL, "--"),
        OperatingPoint(f"cost {cmo:g}:{cfa:g}", cost_thr, C_COST, "-."),
        OperatingPoint("naive 0.5", 0.5, C_NAIVE, ":"),
    ]
    for op in points:
        df[f"pred_{op.name.split()[0]}"] = (prob >= op.threshold).astype(int)

    metrics: Dict[str, float] = {
        "n_buses": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "prevalence_deenergised": float(1 - y.mean()),
        "threshold_model": float(model_threshold),
        "threshold_cost": cost_thr,
    }
    metrics.update({f"{k}": v for k, v in threshold_free_metrics(y, prob).items()})
    # cost-optimal thresholds for all weight ratios (chosen on calibration)
    for w in cost_weights:
        metrics[f"threshold_cost_{w[0]:g}_{w[1]:g}"] = cost_optimal_threshold(y_cal, p_cal, *w)

    for op in points:
        tag = op.name.split()[0]
        m = operating_point_metrics(y, df[f"pred_{tag}"].to_numpy())
        sm = scenario_metrics(df, f"pred_{tag}")
        per_scen = sm.pop("_per_scenario")
        for k, v in {**m, **sm}.items():
            metrics[f"{tag}/{k}"] = v
        if tag in ("model", "cost"):
            ci = cluster_bootstrap(df, op.threshold, n_boot=n_boot, seed=seed)
            for k, (lo, hi) in ci.items():
                metrics[f"{tag}/{k}_ci95_lo"] = lo
                metrics[f"{tag}/{k}_ci95_hi"] = hi
            plot_scenario_errors(per_scen, op.name, os.path.join(out_dir, f"{prefix}scenario_errors_{tag}.{fig_format}"))

    plot_confusions(y, prob, points, os.path.join(out_dir, f"{prefix}confusion_matrices.{fig_format}"))
    plot_roc_pr(y, prob, points, os.path.join(out_dir, f"{prefix}roc_pr.{fig_format}"))
    plot_cost_and_sweep(y, prob, points, cost_weights, os.path.join(out_dir, f"{prefix}cost_threshold.{fig_format}"))
    plot_reliability(y, prob, os.path.join(out_dir, f"{prefix}reliability.{fig_format}"))
    plot_per_bus(df, "pred_model", os.path.join(out_dir, f"{prefix}per_bus.{fig_format}"))
    if "vm_pred" in df:
        metrics.update(plot_voltage_status(df, float(model_threshold), os.path.join(out_dir, f"{prefix}voltage_status.{fig_format}")))
        if "vm_true" in df:
            vm_p = df["vm_pred"].to_numpy().astype(float); vm_t = df["vm_true"].to_numpy().astype(float)
            off = y == 0
            metrics["vm_mae_true_off"] = float(np.mean(np.abs(vm_p[off]))) if off.any() else float("nan")
            metrics["vm_rmse_true_on"] = float(np.sqrt(np.mean((vm_p[~off] - vm_t[~off]) ** 2))) if (~off).any() else float("nan")

    metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in metrics.items()}
    with open(os.path.join(out_dir, f"{prefix}metrics.json"), "w") as fh:
        json.dump({"metrics": metrics, "notes": note}, fh, indent=2)
    _write_summary(metrics, points, note, os.path.join(out_dir, f"{prefix}summary.md"))
    return metrics


def _write_summary(metrics, points, note, path):
    rows = ["balanced_accuracy", "mcc", "off_recall", "off_precision", "on_recall", "on_precision",
            "scenario_exact_match_rate", "mean_missed_outages_per_scenario", "mean_false_alarms_per_scenario"]
    lines = ["# VLD evaluation summary", "", f"_{note.get('calibration', '')}_", "",
             f"Buses: {metrics['n_buses']:,}  Scenarios: {metrics['n_scenarios']:,}  "
             f"De-energised prevalence: {metrics['prevalence_deenergised']:.3f}", "",
             "Threshold-free (de-energised as positive): "
             f"ROC-AUC {metrics.get('roc_auc', float('nan')):.4f}, AP {metrics.get('ap_deenergised', float('nan')):.4f}, "
             f"Brier {metrics['brier']:.4f}, ECE {metrics['ece']:.4f}", "",
             "| metric | " + " | ".join(f"{op.name} (θ={op.threshold:.3f})" for op in points) + " |",
             "|---|" + "---|" * len(points)]
    for r in rows:
        cells = []
        for op in points:
            tag = op.name.split()[0]
            v = metrics.get(f"{tag}/{r}", float("nan"))
            lo, hi = metrics.get(f"{tag}/{r}_ci95_lo"), metrics.get(f"{tag}/{r}_ci95_hi")
            cells.append(f"{v:.4f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else ""))
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _parse_weights(s: str):
    return [tuple(float(x) for x in w.split(":")) for w in s.split(",")]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="parquet or csv written by the VLD task")
    ap.add_argument("--outdir", default="vld_eval")
    ap.add_argument("--calib", default=None, help="validation predictions used to pick the cost threshold")
    ap.add_argument("--threshold", type=float, default=None, help="override the model threshold column")
    ap.add_argument("--cost-weights", default="1:1,3:1,5:1", help="c_missed_outage:c_false_alarm ratios")
    ap.add_argument("--primary-cost", default="3:1")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    def _read(p):
        return pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_parquet(p)

    df = _read(args.predictions)
    cal = _read(args.calib) if args.calib else None
    m = evaluate_vld_predictions(
        df, args.outdir, model_threshold=args.threshold, calib_df=cal,
        cost_weights=_parse_weights(args.cost_weights),
        primary_cost_weights=_parse_weights(args.primary_cost)[0],
        n_boot=args.n_boot, seed=args.seed,
    )
    with open(os.path.join(args.outdir, "summary.md"), encoding="utf-8") as fh:
        print(fh.read())
    return m


if __name__ == "__main__":
    main()
