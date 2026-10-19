"""
Evaluation metrics and figures for Voltage-Loss-Detection (VLD) predictions.

Input is the dataframe written by ``VoltageLossDetectionTask`` (test CSV or
prediction parquet) with columns

    scenario, bus, status_prob, status_pred, status_true, threshold, vm_pred[, vm_true]

where ``status_prob`` is P(energised), ``status_true`` is 1 = energised /
0 = de-energised, ``status_pred`` is the model's hard prediction and
``threshold`` is the decision threshold the model used, 1 / (1 + w_off),
derived by ``VLDTopologyLoss``.

Everything here is about the MODEL'S OWN predictions at its own threshold.
There is no alternative threshold and no comparison operating point.

The metric names deliberately mirror the ones ``VLDTopologyLoss`` logs during
training/validation so the two can be read side by side:

    loss key                                evaluator key
    ------------------------------------    -----------------------------
    VLD Target Accuracy                     accuracy
    VLD Balanced Accuracy                   balanced_accuracy
    VLD Off Recall / Precision / F1         off_recall / off_precision / off_f1
    VLD On Recall                           on_recall
    VLD Mean Prob (reachable & label 1)     mean_prob_label_on   (label-1 buses; the
                                            evaluator has no reachability column)
    VLD Off Rate (labels)                   off_rate
    VLD Decision Threshold                  threshold
    OFF Vm MAE / ON Vm RMSE (task)          vm_mae_true_off / vm_rmse_true_on

"off" = de-energised (label 0), the class VLD exists to detect; "on" =
energised (label 1). off_recall is the outage detection rate; on_recall is one
minus the false-de-energisation rate.

Added on top of the loss metrics: threshold-free ROC-AUC and average precision
with the de-energised class as positive, calibration (Brier, ECE), scenario-level
error counts, a consistency check that ``status_pred`` really equals
``status_prob >= threshold``, the voltage-status consistency of the two heads,
and cluster-bootstrap 95 % intervals over scenarios (buses within a scenario are
not independent, so a row bootstrap would be too narrow).

Usage from code (what the task and the CLI do)::

    from gridfm_app_dist.io.vld_evaluation import evaluate_vld_predictions
    metrics = evaluate_vld_predictions(df, out_dir)

Usage from the shell::

    python -m gridfm_app_dist.io.vld_evaluation preds.parquet --outdir figs
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple

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
C_ON = "#d9731a"  # energised
C_OFF = "#1f5fa8"  # de-energised
C_MODEL = "#1a1a1a"  # the model's operating point
C_MUTED = "#9a9a9a"


def _plt():
    import matplotlib

    if os.environ.get("DISPLAY", "") == "" and matplotlib.get_backend().lower() not in (
        "agg",
        "pdf",
        "svg",
    ):
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


def prediction_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
) -> Dict[str, float]:
    """
    Metrics of the hard predictions ``pred`` plus the calibration of ``prob``.
    Definitions match VLDTopologyLoss (off = label 0 = de-energised).
    """
    from sklearn.metrics import matthews_corrcoef

    tn, fp, fn, tp = _confusion(y, pred)
    n_on, n_off = tp + fn, tn + fp
    nan = float("nan")
    on_recall = tp / n_on if n_on else nan  # 1 - false de-energisation rate
    off_recall = tn / n_off if n_off else nan  # outage detection rate
    on_precision = tp / (tp + fp) if (tp + fp) else nan
    off_precision = tn / (tn + fn) if (tn + fn) else nan
    off_f1 = (
        2 * off_precision * off_recall / (off_precision + off_recall)
        if (off_precision + off_recall) > 0
        else nan
    )
    on = y.astype(bool)
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": float(np.nanmean([on_recall, off_recall])),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) == 2 else nan,
        "off_recall": off_recall,
        "off_precision": off_precision,
        "off_f1": off_f1,
        "on_recall": on_recall,
        "on_precision": on_precision,
        "missed_outage_rate": 1.0 - off_recall if n_off else nan,
        "false_alarm_rate": 1.0 - on_recall if n_on else nan,
        "mean_prob_label_on": float(prob[on].mean()) if on.any() else nan,
        "mean_prob_label_off": float(prob[~on].mean()) if (~on).any() else nan,
        "off_rate": float(1.0 - y.mean()),
        "TP_on": tp,
        "TN_off": tn,
        "FP_pred_on_true_off": fp,  # missed outages
        "FN_pred_off_true_on": fn,  # false alarms
    }


def threshold_free_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """ROC-AUC / AP (de-energised as positive), Brier, ECE."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {"brier": float(np.mean((prob - y) ** 2))}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["ap_deenergised"] = float(average_precision_score(1 - y, 1.0 - prob))
        out["ap_energised"] = float(average_precision_score(y, prob))
    out["ece"] = expected_calibration_error(y, prob)
    return out


def expected_calibration_error(
    y: np.ndarray,
    prob: np.ndarray,
    n_bins: int = 10,
) -> float:
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


def scenario_metrics(
    df: pd.DataFrame,
    pred: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Scenario-level error counts (an operator consumes a scenario, not a bus)."""
    y = df["status_true"].to_numpy().astype(int)
    p = pred.astype(int)
    tab = (
        pd.DataFrame(
            {
                "scenario": df["scenario"].to_numpy(),
                "err": p != y,
                "missed": (p == 1) & (y == 0),
                "alarm": (p == 0) & (y == 1),
            },
        )
        .groupby("scenario", sort=False)
        .sum()
    )
    return {
        "scenario_exact_match_rate": float((tab["err"] == 0).mean()),
        "scenario_no_missed_outage_rate": float((tab["missed"] == 0).mean()),
        "mean_missed_outages_per_scenario": float(tab["missed"].mean()),
        "mean_false_alarms_per_scenario": float(tab["alarm"].mean()),
        "max_missed_outages_in_a_scenario": int(tab["missed"].max()),
        "max_false_alarms_in_a_scenario": int(tab["alarm"].max()),
    }, tab


def cluster_bootstrap(
    df: pd.DataFrame,
    pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    keys: Sequence[str] = (
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "off_recall",
        "off_precision",
        "off_f1",
        "on_recall",
        "ap_deenergised",
        "roc_auc",
    ),
) -> Dict[str, Tuple[float, float]]:
    """95 % intervals by resampling SCENARIOS with replacement."""
    rng = np.random.default_rng(seed)
    scen = df["scenario"].to_numpy()
    uniq, inv = np.unique(scen, return_inverse=True)
    y_all = df["status_true"].to_numpy().astype(int)
    p_all = df["status_prob"].to_numpy().astype(float)
    pred_all = pred.astype(int)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    groups = [order[bounds[i] : bounds[i + 1]] for i in range(len(uniq))]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        y = y_all[idx]
        if len(np.unique(y)) < 2:
            continue
        m = prediction_metrics(y, pred_all[idx], p_all[idx])
        m.update(threshold_free_metrics(y, p_all[idx]))
        for k in keys:
            samples[k].append(m.get(k, np.nan))
    return {
        k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for k, v in samples.items()
        if len(v)
    }


# ----------------------------------------------------------------------------
# Figures (all at the model's own operating point)
# ----------------------------------------------------------------------------
def plot_confusion(y, pred, threshold, path):
    plt = _plt()
    tn, fp, fn, tp = _confusion(y, pred)
    cm = np.array(
        [[tn, fp], [fn, tp]],
    )  # rows: true off, true on; cols: pred off, pred on
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n{cmn[i, j]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if cmn[i, j] > 0.55 else "black",
                )
        ax.set_xticks([0, 1], ["de-energised", "energised"])
        ax.set_yticks([0, 1], ["de-energised", "energised"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Bus status (row-normalised), $\\theta$ = {threshold:.3f}")
        ax.grid(False)
        fig.savefig(path)
        plt.close(fig)


def plot_roc_pr(y, prob, pred, path):
    """ROC and PR with the de-energised class as positive; the model's point marked."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    plt = _plt()
    y_off, p_off = 1 - y, 1.0 - prob
    tn, fp, fn, tp = _confusion(y, pred)
    tpr_o = tn / max(tn + fp, 1)  # outage detection rate
    fpr_o = fn / max(fn + tp, 1)  # energised buses called off
    prec_o = tn / max(tn + fn, 1)
    with plt.rc_context(_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))
        fpr, tpr, _ = roc_curve(y_off, p_off)
        a1.plot(
            fpr,
            tpr,
            color=C_OFF,
            lw=1.6,
            label=f"AUC = {roc_auc_score(y_off, p_off):.3f}",
        )
        a1.plot([0, 1], [0, 1], "--", color=C_MUTED, lw=0.8)
        a1.plot(
            fpr_o,
            tpr_o,
            "o",
            color=C_MODEL,
            ms=5.5,
            mec="white",
            mew=0.7,
            label="model operating point",
        )
        a1.set_xlabel("false positive rate (energised called off)")
        a1.set_ylabel("true positive rate (outage detected)")
        a1.set_title("ROC, de-energised as positive")
        a1.legend(loc="lower right")
        prec, rec, _ = precision_recall_curve(y_off, p_off)
        a2.plot(
            rec,
            prec,
            color=C_OFF,
            lw=1.6,
            label=f"AP = {average_precision_score(y_off, p_off):.3f}",
        )
        a2.axhline(
            y_off.mean(),
            ls="--",
            color=C_MUTED,
            lw=0.8,
            label=f"prevalence {y_off.mean():.2f}",
        )
        a2.plot(
            tpr_o,
            prec_o,
            "o",
            color=C_MODEL,
            ms=5.5,
            mec="white",
            mew=0.7,
            label="model operating point",
        )
        a2.set_xlabel("recall (outage detected)")
        a2.set_ylabel("precision")
        a2.set_title("Precision-recall, de-energised as positive")
        a2.legend(loc="lower left")
        fig.savefig(path)
        plt.close(fig)


def plot_probability_separation(y, prob, threshold, path):
    """Histogram of P(energised) per true class with the model threshold."""
    plt = _plt()
    bins = np.linspace(0, 1, 41)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        ax.hist(
            prob[y == 0],
            bins=bins,
            color=C_OFF,
            alpha=0.75,
            label="true de-energised",
        )
        ax.hist(prob[y == 1], bins=bins, color=C_ON, alpha=0.75, label="true energised")
        ax.axvline(
            threshold,
            color=C_MODEL,
            ls="--",
            lw=1.0,
            label=rf"$\theta$ = {threshold:.3f}",
        )
        ax.set_yscale("log")
        ax.set_xlabel("P(energised)")
        ax.set_ylabel("buses (log)")
        ax.set_title("Predicted probability by true class")
        ax.legend(loc="lower center", framealpha=0.9)
        fig.savefig(path)
        plt.close(fig)


def plot_reliability(y, prob, path, n_bins=10):
    plt = _plt()
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            xs.append(prob[m].mean())
            ys.append(y[m].mean())
            ns.append(m.sum())
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax2 = ax.twinx()
        ax2.bar(xs, ns, width=0.04, color="#e6e6e6", zorder=0)
        ax2.set_ylabel("buses per bin", color=C_MUTED)
        ax2.tick_params(axis="y", colors=C_MUTED)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax.plot([0, 1], [0, 1], "--", color=C_MUTED, lw=0.8, label="perfect")
        ax.plot(xs, ys, "o-", color=C_OFF, lw=1.4, ms=4, label="model", zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted P(energised)")
        ax.set_ylabel("observed energised fraction")
        ax.set_title(
            f"Reliability  (ECE = {expected_calibration_error(y, prob):.3f}, "
            f"Brier = {np.mean((prob - y) ** 2):.3f})",
        )
        ax.legend(loc="upper left")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        fig.savefig(path)
        plt.close(fig)


def plot_scenario_errors(per_scenario: pd.DataFrame, path):
    plt = _plt()
    with plt.rc_context(_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6), sharey=True)
        mx = int(max(per_scenario["missed"].max(), per_scenario["alarm"].max(), 1))
        bins = np.arange(-0.5, mx + 1.5, 1)
        a1.hist(per_scenario["missed"], bins=bins, color=C_OFF, edgecolor="white")
        a1.set_xlabel("missed outages per scenario")
        a1.set_ylabel("scenarios")
        a2.hist(per_scenario["alarm"], bins=bins, color=C_ON, edgecolor="white")
        a2.set_xlabel("false alarms per scenario")
        exact = float((per_scenario["err"] == 0).mean())
        fig.suptitle(f"Per-scenario errors  (exact match {exact:.1%})", fontsize=9)
        fig.savefig(path)
        plt.close(fig)


def plot_per_bus(df: pd.DataFrame, pred: np.ndarray, path):
    plt = _plt()
    y = df["status_true"].to_numpy().astype(int)
    p = pred.astype(int)
    b = df["bus"].to_numpy()
    buses = np.unique(b)
    off_rec, on_rec, off_n = [], [], []
    for bus in buses:
        m = b == bus
        yo, po = y[m] == 0, p[m] == 0
        off_rec.append(np.sum(yo & po) / yo.sum() if yo.sum() else np.nan)
        on_rec.append(np.sum(~yo & ~po) / (~yo).sum() if (~yo).sum() else np.nan)
        off_n.append(yo.mean())
    x = np.arange(len(buses))
    w = 0.4
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        ax.bar(x - w / 2, off_rec, w, color=C_OFF, label="off-recall (outage detected)")
        ax.bar(x + w / 2, on_rec, w, color=C_ON, label="on-recall (no false alarm)")
        ax.plot(x, off_n, "k.", ms=3, label="outage frequency of bus")
        ax.set_xticks(x)
        ax.set_xticklabels(buses, fontsize=6, rotation=90)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("bus (feeder order)")
        ax.set_ylabel("recall")
        ax.set_title("Per-bus detection performance")
        ax.legend(loc="lower left", ncol=3)
        fig.savefig(path)
        plt.close(fig)


def plot_voltage_status(df: pd.DataFrame, threshold: float, path) -> Dict[str, float]:
    """Voltage-status consistency: predicted Vm against P(energised)."""
    plt = _plt()
    y = df["status_true"].to_numpy().astype(int)
    prob = df["status_prob"].to_numpy().astype(float)
    vm = df["vm_pred"].to_numpy().astype(float)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(y), size=min(len(y), 20000), replace=False)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        for cls, col, lab in (
            (0, C_OFF, "true de-energised"),
            (1, C_ON, "true energised"),
        ):
            m = keep[y[keep] == cls]
            ax.scatter(
                prob[m],
                vm[m],
                s=3,
                alpha=0.35,
                color=col,
                label=lab,
                linewidths=0,
            )
        ax.axvline(
            threshold,
            color=C_MODEL,
            ls="--",
            lw=0.9,
            label=rf"$\theta$ = {threshold:.3f}",
        )
        ax.axhline(0.5, color=C_MUTED, ls=":", lw=0.8)
        ax.set_xlabel("P(energised)")
        ax.set_ylabel(r"predicted $\hat V$ (p.u.)")
        ax.set_title("Voltage-status consistency")
        ax.text(
            0.02,
            0.96,
            "status off\nvoltage on",
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            color=C_MUTED,
        )
        ax.text(
            0.98,
            0.04,
            "status on\nvoltage off",
            transform=ax.transAxes,
            fontsize=6,
            va="bottom",
            ha="right",
            color=C_MUTED,
        )
        ax.legend(loc="center right", markerscale=3)
        fig.savefig(path)
        plt.close(fig)
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
    n_boot: int = 1000,
    seed: int = 0,
    prefix: str = "",
    fig_format: str = "png",
) -> Dict[str, float]:
    """
    Evaluate the model's VLD predictions, write figures + metrics.json +
    summary.md to ``out_dir`` and return a flat metric dict (suitable for
    ``logger.log_metrics``).

    The hard predictions are taken from the ``status_pred`` column when present
    (what the model actually exported); otherwise they are recomputed as
    ``status_prob >= model_threshold``. ``model_threshold`` defaults to the
    ``threshold`` column. ``seed`` only affects the bootstrap intervals.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df["status_true"] = df["status_true"].astype(int)
    df["status_prob"] = df["status_prob"].astype(float)
    y = df["status_true"].to_numpy()
    prob = df["status_prob"].to_numpy()

    if model_threshold is None:
        model_threshold = float(df["threshold"].iloc[0]) if "threshold" in df else 0.5
    model_threshold = float(model_threshold)

    recomputed = (prob >= model_threshold).astype(int)
    if "status_pred" in df:
        pred = df["status_pred"].to_numpy().astype(int)
        pred_mismatch = float(np.mean(pred != recomputed))
    else:
        pred = recomputed
        pred_mismatch = 0.0

    metrics: Dict[str, float] = {
        "n_buses": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "threshold": model_threshold,
        # Fraction of exported status_pred that disagree with prob >= threshold.
        # Should be 0; anything else means the export and the loss used different
        # thresholds (e.g. a hard-coded value somewhere in the inference path).
        "status_pred_threshold_mismatch": pred_mismatch,
    }
    metrics.update(prediction_metrics(y, pred, prob))
    metrics.update(threshold_free_metrics(y, prob))
    sm, per_scen = scenario_metrics(df, pred)
    metrics.update(sm)
    for k, (lo, hi) in cluster_bootstrap(df, pred, n_boot=n_boot, seed=seed).items():
        metrics[f"{k}_ci95_lo"] = lo
        metrics[f"{k}_ci95_hi"] = hi

    f = lambda name: os.path.join(out_dir, f"{prefix}{name}.{fig_format}")  # noqa: E731
    plot_confusion(y, pred, model_threshold, f("confusion_matrix"))
    plot_roc_pr(y, prob, pred, f("roc_pr"))
    plot_probability_separation(y, prob, model_threshold, f("probability_separation"))
    plot_reliability(y, prob, f("reliability"))
    plot_scenario_errors(per_scen, f("scenario_errors"))
    plot_per_bus(df, pred, f("per_bus"))
    if "vm_pred" in df:
        metrics.update(plot_voltage_status(df, model_threshold, f("voltage_status")))
        vm_p = df["vm_pred"].to_numpy().astype(float)
        off = y == 0
        metrics["vm_mae_true_off"] = (
            float(np.mean(np.abs(vm_p[off]))) if off.any() else float("nan")
        )
        if "vm_true" in df:
            vm_t = df["vm_true"].to_numpy().astype(float)
            metrics["vm_rmse_true_on"] = (
                float(np.sqrt(np.mean((vm_p[~off] - vm_t[~off]) ** 2)))
                if (~off).any()
                else float("nan")
            )

    metrics = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in metrics.items()
    }
    with open(
        os.path.join(out_dir, f"{prefix}metrics.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(metrics, fh, indent=2)
    _write_summary(metrics, os.path.join(out_dir, f"{prefix}summary.md"))
    return metrics


def _fmt(metrics, key, digits=4):
    v = metrics.get(key, float("nan"))
    s = f"{v:.{digits}f}" if isinstance(v, float) else str(v)
    lo, hi = metrics.get(f"{key}_ci95_lo"), metrics.get(f"{key}_ci95_hi")
    if lo is not None and hi is not None:
        s += f"  [{lo:.3f}, {hi:.3f}]"
    return s


def _write_summary(metrics, path):
    m = metrics
    rows = [
        ("accuracy", "VLD Target Accuracy"),
        ("balanced_accuracy", "VLD Balanced Accuracy"),
        ("mcc", "-"),
        ("off_recall", "VLD Off Recall"),
        ("off_precision", "VLD Off Precision"),
        ("off_f1", "VLD Off F1"),
        ("on_recall", "VLD On Recall"),
        ("on_precision", "-"),
        ("mean_prob_label_on", "VLD Mean Prob (reachable & label 1)"),
        ("mean_prob_label_off", "-"),
        ("off_rate", "VLD Off Rate (labels)"),
        ("roc_auc", "-"),
        ("ap_deenergised", "-"),
        ("brier", "-"),
        ("ece", "-"),
        ("vm_mae_true_off", "OFF Vm MAE (task)"),
        ("vm_rmse_true_on", "ON Vm RMSE (task)"),
        ("inconsistent_status_on_voltage_off", "-"),
        ("inconsistent_status_off_voltage_on", "-"),
        ("scenario_exact_match_rate", "-"),
        ("scenario_no_missed_outage_rate", "-"),
        ("mean_missed_outages_per_scenario", "-"),
        ("mean_false_alarms_per_scenario", "-"),
    ]
    lines = [
        "# VLD evaluation summary (model predictions at its own threshold)",
        "",
        f"Buses: {m['n_buses']:,}   Scenarios: {m['n_scenarios']:,}   "
        f"De-energised prevalence: {m['off_rate']:.3f}   Threshold θ = {m['threshold']:.4f}",
        "",
    ]
    if m.get("status_pred_threshold_mismatch", 0) > 0:
        lines += [
            f"**Warning:** {m['status_pred_threshold_mismatch']:.2%} of exported `status_pred` "
            "values differ from `status_prob >= threshold`; the export and the loss used "
            "different thresholds.",
            "",
        ]
    lines += [
        "Intervals are cluster-bootstrap 95 % over scenarios. 'off' = de-energised (label 0).",
        "",
        "| metric | value | corresponding loss / task metric |",
        "|---|---|---|",
    ]
    for key, loss_name in rows:
        if key in m:
            lines.append(f"| {key} | {_fmt(m, key)} | {loss_name} |")
    lines += [
        "",
        f"Confusion (rows true off/on, cols pred off/on): "
        f"TN_off={m['TN_off']:,}  FP(missed outage)={m['FP_pred_on_true_off']:,}  "
        f"FN(false alarm)={m['FN_pred_off_true_on']:,}  TP_on={m['TP_on']:,}",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("predictions", help="parquet or csv written by the VLD task")
    ap.add_argument("--outdir", default="vld_eval")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the threshold column",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    p = args.predictions
    df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_parquet(p)
    m = evaluate_vld_predictions(
        df,
        args.outdir,
        model_threshold=args.threshold,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    with open(os.path.join(args.outdir, "summary.md"), encoding="utf-8") as fh:
        print(fh.read())
    return m


if __name__ == "__main__":
    main()
