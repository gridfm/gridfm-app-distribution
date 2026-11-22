"""
Grouped (out-of-core) evaluation of VLD predictions for prediction files that are too
large to load at once.

``vld_evaluation.evaluate_vld_predictions`` needs the whole prediction table in memory
(and its cluster bootstrap re-evaluates the full table 1,000 times). For a file with
millions of scenarios (e.g. JRC_MV_rural at contingency level 3: 2.1 M scenarios x 117
buses = 250 M rows) that is not possible. This module evaluates the same file in groups
of scenarios and combines the groups afterwards:

1. **Scan** (columns ``scenario``, ``bus``, ``threshold`` only): counts rows and
   scenarios, collects the bus ids and the threshold, and checks whether the rows of each
   scenario are stored together (they are in files written by ``gridfm_app_dist predict``,
   one model batch after another).
2. **Evaluate group by group**: the scenarios are split into groups of ``group_size``
   (default 100,000) and each group is read and reduced to additive statistics:
   confusion counts, per-bus counts, per-scenario error counts, probability histograms,
   voltage error sums and a small random sample for the scatter plot. Only one group is in
   memory at a time. Each group's statistics are saved to ``<outdir>/groups/`` so an
   interrupted run resumes where it stopped.
3. **Aggregate**: the group statistics are summed and the final metrics and figures are
   computed from the totals.

How the groups are combined
---------------------------
Summing the group statistics is the same as averaging the groups weighted by their size,
so the final metrics and figures are those of the whole file: counts, rates, Brier score,
voltage errors, per-bus and per-scenario results are identical to what
``evaluate_vld_predictions`` would report if the file fitted in memory. ROC-AUC, average
precision, ECE and the ROC / PR / reliability curves come from a histogram of the
predicted probabilities with 65,536 bins on the logit scale; they agree with the exact
values to about 1e-4. The 95 % intervals are the same cluster bootstrap over scenarios as
in ``vld_evaluation`` (same seed, same resampling), run on per-scenario counts; that is
exact for the count-based metrics. ROC-AUC and AP get no bootstrap interval; their
variation between groups is reported instead.

Group-level results are also kept: ``group_metrics.csv`` has every metric for every
group, ``summary.md`` shows the mean, standard deviation and range over the groups next
to the overall value, and ``group_variation`` plots the main metrics group by group.

Usage from the shell::

    python -m gridfm_app_dist.io.vld_evaluation_grouped JRC_MV_rural_predictions.parquet \\
        --outdir JRC_MV_rural_eval --group-size 100000

    # quick trial on the first two groups only
    python -m gridfm_app_dist.io.vld_evaluation_grouped preds.parquet --outdir trial --max-groups 2

Usage from code::

    from gridfm_app_dist.io.vld_evaluation_grouped import evaluate_vld_file_grouped
    metrics = evaluate_vld_file_grouped("preds.parquet", "eval_dir", group_size=100_000)

Memory: about 2 GB for a group of 100,000 scenarios of a 117-bus network (use a smaller
``--group-size`` if needed); the scan needs well under 1 GB. The input can be the parquet
written by the task or a CSV with the same columns.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
import warnings
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from gridfm_app_dist.io.vld_evaluation import (
    _RC,
    C_MODEL,
    C_MUTED,
    C_OFF,
    C_ON,
    _limit_arrow_threads,
    _plt,
)

REQUIRED = ("scenario", "bus", "status_prob", "status_true")
OPTIONAL = ("status_pred", "vm_pred", "vm_true")

# Fine probability histogram (logit scale) used for ROC-AUC, AP, ECE and the curves.
N_FINE = 65536
LOGIT_LO, LOGIT_HI = -40.0, 40.0
SEP_EDGES = np.linspace(0,1,41)  # same bins as vld_evaluation.plot_probability_separation
N_ECE_BINS = 10
SCATTER_POINTS = 20000  # same total as vld_evaluation.plot_voltage_status
BOOT_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "off_recall",
    "off_precision",
    "off_f1",
    "on_recall",
)
SUMMARY_ROWS = [
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


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _peak_rss_gb() -> float:
    try:
        import resource

        return (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        )  # KB on Linux
    except Exception:  # Windows: no resource module
        return float("nan")


# ----------------------------------------------------------------------------
# Reading in chunks (parquet via pyarrow, or CSV via pandas)
# ----------------------------------------------------------------------------
def _is_csv(path: str) -> bool:
    return path.lower().endswith(".csv")


def _column_names(path: str) -> List[str]:
    if _is_csv(path):
        import pandas as pd

        return list(pd.read_csv(path, nrows=0).columns)
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    try:
        return list(pf.schema_arrow.names)
    finally:
        pf.close()


def _iter_chunks(
    path: str,
    columns: Sequence[str],
    batch_rows: int,
    use_threads: bool = False,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield the file in row chunks as {column: numpy array}, in file order."""
    columns = list(columns)
    if _is_csv(path):
        import pandas as pd

        for chunk in pd.read_csv(path, usecols=columns, chunksize=batch_rows):
            yield {c: chunk[c].to_numpy() for c in columns}
        return
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    try:
        for batch in pf.iter_batches(
            batch_size=batch_rows,
            columns=columns,
            use_threads=use_threads,
        ):
            names = batch.schema.names
            yield {
                c: batch.column(names.index(c)).to_numpy(zero_copy_only=False)
                for c in columns
            }
    finally:
        pf.close()


def _compact(chunk: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Smaller dtypes for the buffered columns (labels and hard predictions as int8).

    ``status_prob`` and the voltages keep their stored dtype so the values are exactly
    those ``vld_evaluation`` would see.
    """
    out = {}
    for c, v in chunk.items():
        if c in ("status_true", "status_pred"):
            out[c] = v.astype(np.int8)  # same truncation as .astype(int) for 0/1 labels
        elif c == "scenario":
            out[c] = v.astype(np.int64, copy=False)
        else:
            out[c] = v
    return out


# ----------------------------------------------------------------------------
# Pass 1: scan
# ----------------------------------------------------------------------------
def scan_file(
    path: str,
    batch_rows: int = 1_000_000,
    use_threads: bool = False,
) -> dict:
    """Count rows and scenarios, collect bus ids and threshold, check scenario contiguity."""
    names = _column_names(path)
    missing = [c for c in REQUIRED if c not in names]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}")
    cols = ["scenario", "bus"] + (["threshold"] if "threshold" in names else [])
    n_rows, n_runs, last = 0, 0, None
    pending: List[np.ndarray] = []
    pending_n = 0
    scen_ids = np.empty(0, np.int64)
    buses = np.empty(0, np.int64)
    thr_first, thr_min, thr_max = None, math.inf, -math.inf
    for ch in _iter_chunks(path, cols, batch_rows, use_threads):
        s = ch["scenario"].astype(np.int64, copy=False)
        if s.size == 0:
            continue
        runs = 1 + int(np.count_nonzero(s[1:] != s[:-1]))
        if last is not None and s[0] == last:
            runs -= 1  # the run continues from the previous chunk
        n_runs += runs
        last = s[-1]
        n_rows += s.size
        u = np.unique(s)
        pending.append(u)
        pending_n += u.size
        if pending_n > 5_000_000:
            scen_ids = np.union1d(scen_ids, np.concatenate(pending))
            pending, pending_n = [], 0
        buses = np.union1d(buses, np.unique(ch["bus"].astype(np.int64, copy=False)))
        if "threshold" in ch:
            t = ch["threshold"]
            if thr_first is None:
                thr_first = float(t[0])
            thr_min = min(thr_min, float(np.min(t)))
            thr_max = max(thr_max, float(np.max(t)))
    if pending:
        scen_ids = np.union1d(scen_ids, np.concatenate(pending))
    if n_rows == 0:
        raise ValueError(f"{path} has no rows")
    return {
        "columns": names,
        "n_rows": n_rows,
        "n_scenarios": int(scen_ids.size),
        "scenario_ids": scen_ids,  # sorted
        "contiguous": n_runs == scen_ids.size,
        "n_runs": n_runs,
        "bus_ids": buses,  # sorted
        "threshold_first": thr_first,
        "threshold_min": None if thr_first is None else thr_min,
        "threshold_max": None if thr_first is None else thr_max,
    }


# ----------------------------------------------------------------------------
# Pass 2: groups
# ----------------------------------------------------------------------------
def _concat(pieces: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate the buffered chunks of a group and empty the buffer (frees the pieces
    before the group is processed, so a group is held only once)."""
    out = {c: np.concatenate([p[c] for p in pieces]) for c in pieces[0]}
    pieces.clear()
    return out


def _iter_groups_contiguous(
    path,
    cols,
    group_size,
    n_groups,
    batch_rows,
    use_threads,
    skip,
    stop_after,
) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
    """One streaming pass. Group k = scenarios k*G .. (k+1)*G-1 in order of appearance.

    Valid because every scenario's rows are contiguous, so a scenario never spans two
    groups. Groups in ``skip`` (already on disk) are read past without buffering.
    """
    runs_before, last = 0, None
    current, pieces = None, []
    for ch in _iter_chunks(path, cols, batch_rows, use_threads):
        s = ch["scenario"]
        if s.size == 0:
            continue
        new = np.empty(s.size, bool)
        new[0] = last is None or s[0] != last
        new[1:] = s[1:] != s[:-1]
        last = s[-1]
        run_idx = runs_before + np.cumsum(new) - 1
        runs_before += int(new.sum())
        grp = run_idx // group_size  # non-decreasing within the chunk
        cut = np.flatnonzero(np.diff(grp)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, s.size]):
            g = int(grp[a])
            if g != current:
                if current is not None and current not in skip and pieces:
                    yield current, _concat(pieces)
                pieces = []
                current = g
                if stop_after is not None and g >= stop_after:
                    return
            if g not in skip:
                pieces.append(_compact({c: v[a:b] for c, v in ch.items()}))
    if current is not None and current not in skip and pieces:
        yield current, _concat(pieces)


def _iter_groups_by_id(
    path,
    cols,
    hi_ids,
    batch_rows,
    use_threads,
    skip,
    stop_after,
) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
    """Fallback when a scenario's rows are scattered: one pass over the file per group.

    Group k = the k-th block of ``group_size`` scenario ids in sorted order
    (``hi_ids[k]`` is the last id of group k).
    """
    n_groups = len(hi_ids)
    for k in range(n_groups if stop_after is None else min(stop_after, n_groups)):
        if k in skip:
            continue
        pieces = []
        for ch in _iter_chunks(path, cols, batch_rows, use_threads):
            g = np.searchsorted(
                hi_ids,
                ch["scenario"].astype(np.int64, copy=False),
                "left",
            )
            m = g == k
            if m.any():
                pieces.append(_compact({c: v[m] for c, v in ch.items()}))
        yield k, _concat(pieces)


# ----------------------------------------------------------------------------
# Statistics of one group (all additive, except the scatter sample)
# ----------------------------------------------------------------------------
def _fine_bin(prob: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(prob) - np.log1p(-prob)
    lg = np.nan_to_num(lg, nan=0.0, posinf=LOGIT_HI, neginf=LOGIT_LO)
    idx = np.floor((lg - LOGIT_LO) * (N_FINE / (LOGIT_HI - LOGIT_LO))).astype(np.int64)
    np.clip(idx, 0, N_FINE - 1, out=idx)
    return idx


def group_statistics(
    d: Dict[str, np.ndarray],
    threshold: float,
    bus_ids: np.ndarray,
    n_sample: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Reduce one group's rows to the statistics the final metrics and figures need."""
    y = d["status_true"].astype(np.int8)
    prob = d["status_prob"].astype(np.float64)
    n = y.size
    recomputed = prob >= threshold
    if "status_pred" in d:
        pred = d["status_pred"].astype(np.int8)
        mismatch = int(np.count_nonzero(pred != recomputed))
    else:
        pred = recomputed.astype(np.int8)
        mismatch = 0
    on, pon = y.astype(bool), pred.astype(bool)  # as vld_evaluation._confusion
    st: Dict[str, np.ndarray] = {
        "n_rows": np.int64(n),
        "tp": np.int64(np.count_nonzero(on & pon)),
        "tn": np.int64(np.count_nonzero(~on & ~pon)),
        "fp": np.int64(np.count_nonzero(~on & pon)),  # missed outages
        "fn": np.int64(np.count_nonzero(on & ~pon)),  # false alarms
        "mismatch": np.int64(mismatch),
        "sum_y": np.int64(y.sum(dtype=np.int64)),
        "sum_prob_on": np.float64(prob[on].sum()),
        "sum_prob_off": np.float64(prob[~on].sum()),
        "brier_sum": np.float64(np.sum((prob - y) ** 2)),
    }
    # probability histograms
    fb = _fine_bin(prob)
    st["fine_on"] = np.bincount(fb[on], minlength=N_FINE).astype(np.int64)
    st["fine_off"] = np.bincount(fb[~on], minlength=N_FINE).astype(np.int64)
    st["fine_sum_prob"] = np.bincount(fb, weights=prob, minlength=N_FINE)
    del fb
    st["sep_off"] = np.histogram(prob[y == 0], bins=SEP_EDGES)[0].astype(np.int64)
    st["sep_on"] = np.histogram(prob[y == 1], bins=SEP_EDGES)[0].astype(np.int64)
    # per bus (as vld_evaluation.plot_per_bus: y == 0 / pred == 0)
    bus = d["bus"].astype(np.int64, copy=False)
    bi = np.searchsorted(bus_ids, bus)
    if bi.max() >= bus_ids.size or not np.array_equal(bus_ids[bi], bus):
        raise ValueError(
            "bus id not seen in the scan; the file changed during the run?",
        )
    yo, po = y == 0, pred == 0
    nb = bus_ids.size
    st["bus_n"] = np.bincount(bi, minlength=nb).astype(np.int64)
    st["bus_off_n"] = np.bincount(bi[yo], minlength=nb).astype(np.int64)
    st["bus_off_hit"] = np.bincount(bi[yo & po], minlength=nb).astype(np.int64)
    st["bus_on_hit"] = np.bincount(bi[~yo & ~po], minlength=nb).astype(np.int64)
    # per scenario (as vld_evaluation.scenario_metrics: missed = pred 1 & true 0, alarm = pred 0 & true 1)
    scen_u, inv = np.unique(d["scenario"], return_inverse=True)
    ns = scen_u.size
    st["scen_id"] = scen_u.astype(np.int64)
    st["scen_tp"] = np.bincount(inv[on & pon], minlength=ns).astype(np.int32)
    st["scen_tn"] = np.bincount(inv[~on & ~pon], minlength=ns).astype(np.int32)
    st["scen_missed"] = np.bincount(inv[(pred == 1) & (y == 0)], minlength=ns).astype(
        np.int32,
    )
    st["scen_alarm"] = np.bincount(inv[(pred == 0) & (y == 1)], minlength=ns).astype(
        np.int32,
    )
    st["scen_err"] = np.bincount(inv[pred != y], minlength=ns).astype(np.int32)
    del inv
    # voltage (as vld_evaluation: consistency uses prob >= threshold and vm >= 0.5)
    if "vm_pred" in d:
        vm = d["vm_pred"].astype(np.float64)
        v_on = vm >= 0.5
        off = y == 0
        st["inc_on_voff"] = np.int64(np.count_nonzero(recomputed & ~v_on))
        st["inc_off_von"] = np.int64(np.count_nonzero(~recomputed & v_on))
        st["vm_n_off"] = np.int64(np.count_nonzero(off))
        st["vm_abs_off_sum"] = np.float64(np.abs(vm[off]).sum())
        if "vm_true" in d:
            vt = d["vm_true"].astype(np.float64)
            st["vm_n_on"] = np.int64(np.count_nonzero(~off))
            st["vm_sq_on_sum"] = np.float64(np.sum((vm[~off] - vt[~off]) ** 2))
        k = min(n, n_sample)
        pick = (
            np.sort(rng.choice(n, size=k, replace=False))
            if k
            else np.empty(0, np.int64)
        )
        st["smp_prob"] = prob[pick].astype(np.float32)
        st["smp_vm"] = vm[pick].astype(np.float32)
        st["smp_y"] = y[pick]
    return st


_CONCAT = (
    "scen_id",
    "scen_tp",
    "scen_tn",
    "scen_missed",
    "scen_alarm",
    "scen_err",
    "smp_prob",
    "smp_vm",
    "smp_y",
)


def combine(stats: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Sum the additive statistics of several groups; concatenate the per-scenario ones."""
    keys = set(stats[0])
    for s in stats[1:]:
        if set(s) != keys:
            raise ValueError("groups have different statistics (mixed column sets?)")
    out = {}
    for k in stats[0]:
        if k in _CONCAT:
            out[k] = np.concatenate([s[k] for s in stats])
        else:
            out[k] = np.sum([s[k] for s in stats], axis=0)
    return out


# ----------------------------------------------------------------------------
# Metrics from statistics
# ----------------------------------------------------------------------------
def _count_metrics(tp, tn, fp, fn) -> Dict[str, float]:
    """Same definitions as vld_evaluation.prediction_metrics (energised = positive)."""
    nan = float("nan")
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    n_on, n_off = tp + fn, tn + fp
    on_recall = tp / n_on if n_on else nan
    off_recall = tn / n_off if n_off else nan
    on_precision = tp / (tp + fp) if (tp + fp) else nan
    off_precision = tn / (tn + fn) if (tn + fn) else nan
    off_f1 = (
        2 * off_precision * off_recall / (off_precision + off_recall)
        if (off_precision + off_recall) > 0
        else nan
    )
    if n_on and n_off:
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0  # sklearn returns 0 here
    else:
        mcc = nan  # vld_evaluation: only one class present
    rec = [r for r in (on_recall, off_recall) if not math.isnan(r)]
    return {
        "accuracy": (tp + tn) / max(n_on + n_off, 1),
        "balanced_accuracy": float(np.mean(rec)) if rec else nan,
        "mcc": mcc,
        "off_recall": off_recall,
        "off_precision": off_precision,
        "off_f1": off_f1,
        "on_recall": on_recall,
        "on_precision": on_precision,
        "missed_outage_rate": 1.0 - off_recall if n_off else nan,
        "false_alarm_rate": 1.0 - on_recall if n_on else nan,
    }


def _roc_auc(h_on: np.ndarray, h_off: np.ndarray) -> float:
    """P(prob_on > prob_off) + 0.5 P(tie); positive = energised (as roc_auc_score(y, prob))."""
    n_on, n_off = float(h_on.sum()), float(h_off.sum())
    below = np.cumsum(h_off) - h_off
    return float(np.sum(h_on * (below + 0.5 * h_off)) / (n_on * n_off))


def _average_precision(h_pos: np.ndarray, h_neg: np.ndarray) -> float:
    """sklearn's step-wise AP; bins ordered from the highest score of the positive class."""
    keep = (h_pos + h_neg) > 0
    tps, fps = np.cumsum(h_pos[keep]), np.cumsum(h_neg[keep])
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _ece_bins(h_all: np.ndarray, n_bins: int = N_ECE_BINS) -> np.ndarray:
    """Quantile bins as in vld_evaluation.expected_calibration_error, at fine-bin resolution.

    Returns the ECE bin of every fine bin (or an empty array when only one edge remains).
    """
    total = int(h_all.sum())
    cum = np.cumsum(h_all)
    # np.quantile's order statistic for q = b / n_bins, mapped to the fine bin holding it
    pos = np.floor(np.linspace(0, 1, n_bins + 1) * (total - 1)).astype(np.int64)
    edges = np.unique(np.searchsorted(cum, pos, side="right"))
    if edges.size < 2:
        return np.empty(0, np.int64)
    fine = np.arange(h_all.size)
    return np.clip(np.searchsorted(edges, fine, side="right") - 1, 0, edges.size - 2)


def _reliability(st) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    h_all = st["fine_on"] + st["fine_off"]
    b = _ece_bins(h_all)
    if b.size == 0:
        return np.empty(0), np.empty(0), np.empty(0), 0.0
    nb = int(b.max()) + 1
    n = np.bincount(b, weights=h_all, minlength=nb)
    sp = np.bincount(b, weights=st["fine_sum_prob"], minlength=nb)
    so = np.bincount(b, weights=st["fine_on"], minlength=nb)
    m = n > 0
    xs, ys, ns = sp[m] / n[m], so[m] / n[m], n[m]
    ece = float(np.sum(ns / h_all.sum() * np.abs(xs - ys)))
    return xs, ys, ns, ece


def metrics_from_statistics(
    st: Dict[str, np.ndarray],
    threshold: float,
) -> Dict[str, float]:
    """All metrics of ``vld_evaluation.evaluate_vld_predictions`` except the intervals."""
    nan = float("nan")
    n = int(st["n_rows"])
    tp, tn, fp, fn = (int(st[k]) for k in ("tp", "tn", "fp", "fn"))
    n_on, n_off = tp + fn, tn + fp
    m: Dict[str, float] = {
        "n_buses": n,
        "n_scenarios": int(st["scen_id"].size),
        "threshold": float(threshold),
        "status_pred_threshold_mismatch": int(st["mismatch"]) / n,
    }
    m.update(_count_metrics(tp, tn, fp, fn))
    m["mean_prob_label_on"] = float(st["sum_prob_on"]) / n_on if n_on else nan
    m["mean_prob_label_off"] = float(st["sum_prob_off"]) / n_off if n_off else nan
    m["off_rate"] = 1.0 - int(st["sum_y"]) / n
    m.update(
        {
            "TP_on": tp,
            "TN_off": tn,
            "FP_pred_on_true_off": fp,
            "FN_pred_off_true_on": fn,
        },
    )
    m["brier"] = float(st["brier_sum"]) / n
    if n_on and n_off:
        m["roc_auc"] = _roc_auc(st["fine_on"], st["fine_off"])
        m["ap_deenergised"] = _average_precision(st["fine_off"], st["fine_on"])
        m["ap_energised"] = _average_precision(
            st["fine_on"][::-1],
            st["fine_off"][::-1],
        )
    m["ece"] = _reliability(st)[3]
    missed, alarm, err = st["scen_missed"], st["scen_alarm"], st["scen_err"]
    m.update(
        {
            "scenario_exact_match_rate": float(np.mean(err == 0)),
            "scenario_no_missed_outage_rate": float(np.mean(missed == 0)),
            "mean_missed_outages_per_scenario": float(np.mean(missed)),
            "mean_false_alarms_per_scenario": float(np.mean(alarm)),
            "max_missed_outages_in_a_scenario": int(missed.max()),
            "max_false_alarms_in_a_scenario": int(alarm.max()),
        },
    )
    if "inc_on_voff" in st:
        m["inconsistent_status_on_voltage_off"] = int(st["inc_on_voff"]) / n
        m["inconsistent_status_off_voltage_on"] = int(st["inc_off_von"]) / n
        n_voff = int(st["vm_n_off"])
        m["vm_mae_true_off"] = float(st["vm_abs_off_sum"]) / n_voff if n_voff else nan
        if "vm_n_on" in st:
            n_von = int(st["vm_n_on"])
            m["vm_rmse_true_on"] = (
                math.sqrt(float(st["vm_sq_on_sum"]) / n_von) if n_von else nan
            )
    return m


def bootstrap_intervals(
    st: Dict[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Tuple[float, float]]:
    """Cluster bootstrap over scenarios, as vld_evaluation.cluster_bootstrap.

    Scenarios are ordered by id (as np.unique orders them there) and resampled with the
    same generator calls, so for the count-based metrics the intervals are the same.
    """
    order = np.argsort(st["scen_id"], kind="stable")
    c = np.stack(
        [st["scen_tp"], st["scen_tn"], st["scen_missed"], st["scen_alarm"]],
        axis=1,
    )[order].astype(np.float64)
    n = c.shape[0]
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in BOOT_KEYS}
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        tp, tn, fp, fn = np.bincount(pick, minlength=n).astype(np.float64) @ c
        if tp + fn == 0 or tn + fp == 0:
            continue
        m = _count_metrics(tp, tn, fp, fn)
        for k in BOOT_KEYS:
            samples[k].append(m[k])
    return {
        k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for k, v in samples.items()
        if len(v)
    }


# ----------------------------------------------------------------------------
# Figures (from the combined statistics; same style as vld_evaluation)
# ----------------------------------------------------------------------------
def _roc_pr_points(h_on, h_off):
    """ROC and PR points with the de-energised class as positive (ascending P(energised))."""
    keep = (h_on + h_off) > 0
    tps, fps = np.cumsum(h_off[keep]), np.cumsum(h_on[keep])
    tpr, fpr = tps / tps[-1], fps / fps[-1]
    prec = tps / (tps + fps)
    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[1.0, prec], np.r_[0.0, tpr]


def plot_confusion_counts(tn, fp, fn, tp, threshold, path, subtitle=""):
    plt = _plt()
    cm = np.array([[tn, fp], [fn, tp]])
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
        ax.set_title(
            f"Bus status (row-normalised), $\\theta$ = {threshold:.3f}{subtitle}",
        )
        ax.grid(False)
        fig.savefig(path)
        plt.close(fig)


def plot_roc_pr_grouped(st, groups, metrics, path):
    plt = _plt()
    tn, fp, fn, tp = (int(st[k]) for k in ("tn", "fp", "fn", "tp"))
    tpr_o, fpr_o, prec_o = (
        tn / max(tn + fp, 1),
        fn / max(fn + tp, 1),
        tn / max(tn + fn, 1),
    )
    prevalence = 1.0 - int(st["sum_y"]) / int(st["n_rows"])
    with plt.rc_context(_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))
        for i, g in enumerate(groups):
            if g["fine_on"].sum() == 0 or g["fine_off"].sum() == 0:
                continue
            gx, gy, gp, gr = _roc_pr_points(g["fine_on"], g["fine_off"])
            lab = "single groups" if i == 0 else None
            a1.plot(gx, gy, color=C_MUTED, lw=0.5, alpha=0.6, label=lab)
            a2.plot(gr, gp, color=C_MUTED, lw=0.5, alpha=0.6, label=lab)
        fx, fy, fp_, fr = _roc_pr_points(st["fine_on"], st["fine_off"])
        a1.plot(
            fx,
            fy,
            color=C_OFF,
            lw=1.6,
            label=f"all groups, AUC = {metrics['roc_auc']:.3f}",
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
        a2.plot(
            fr,
            fp_,
            color=C_OFF,
            lw=1.6,
            label=f"all groups, AP = {metrics['ap_deenergised']:.3f}",
        )
        a2.axhline(
            prevalence,
            ls="--",
            color=C_MUTED,
            lw=0.8,
            label=f"prevalence {prevalence:.2f}",
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


def plot_probability_separation_counts(st, threshold, path):
    plt = _plt()
    centres = SEP_EDGES[:-1]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        ax.hist(
            centres,
            bins=SEP_EDGES,
            weights=st["sep_off"],
            color=C_OFF,
            alpha=0.75,
            label="true de-energised",
        )
        ax.hist(
            centres,
            bins=SEP_EDGES,
            weights=st["sep_on"],
            color=C_ON,
            alpha=0.75,
            label="true energised",
        )
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


def plot_reliability_counts(st, path):
    plt = _plt()
    xs, ys, ns, ece = _reliability(st)
    brier = float(st["brier_sum"]) / int(st["n_rows"])
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
        ax.set_title(f"Reliability  (ECE = {ece:.3f}, Brier = {brier:.3f})")
        ax.legend(loc="upper left")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        fig.savefig(path)
        plt.close(fig)


def plot_scenario_errors_counts(st, path):
    plt = _plt()
    missed, alarm = st["scen_missed"], st["scen_alarm"]
    mx = int(max(missed.max(), alarm.max(), 1))
    bins = np.arange(-0.5, mx + 1.5, 1)
    vals = np.arange(mx + 1)
    with plt.rc_context(_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6), sharey=True)
        a1.hist(
            vals,
            bins=bins,
            weights=np.bincount(missed, minlength=mx + 1),
            color=C_OFF,
            edgecolor="white",
        )
        a1.set_xlabel("missed outages per scenario")
        a1.set_ylabel("scenarios")
        a2.hist(
            vals,
            bins=bins,
            weights=np.bincount(alarm, minlength=mx + 1),
            color=C_ON,
            edgecolor="white",
        )
        a2.set_xlabel("false alarms per scenario")
        exact = float(np.mean(st["scen_err"] == 0))
        fig.suptitle(f"Per-scenario errors  (exact match {exact:.1%})", fontsize=9)
        fig.savefig(path)
        plt.close(fig)


def _bus_rates(st):
    with np.errstate(divide="ignore", invalid="ignore"):
        on_n = st["bus_n"] - st["bus_off_n"]
        off_rec = np.where(
            st["bus_off_n"] > 0,
            st["bus_off_hit"] / st["bus_off_n"],
            np.nan,
        )
        on_rec = np.where(on_n > 0, st["bus_on_hit"] / on_n, np.nan)
        off_freq = np.where(st["bus_n"] > 0, st["bus_off_n"] / st["bus_n"], np.nan)
    return off_rec, on_rec, off_freq


def plot_per_bus_grouped(st, groups, bus_ids, path):
    """Bars: all groups; whiskers: lowest and highest value over the single groups."""
    plt = _plt()
    off_rec, on_rec, off_freq = _bus_rates(st)
    g_off = np.array([_bus_rates(g)[0] for g in groups])
    g_on = np.array([_bus_rates(g)[1] for g in groups])
    x = np.arange(bus_ids.size)
    w = 0.4

    def err(center, g):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bus columns
            lo, hi = np.nanmin(g, axis=0), np.nanmax(g, axis=0)
        return np.vstack([np.clip(center - lo, 0, None), np.clip(hi - center, 0, None)])

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        kw = (
            dict(error_kw=dict(lw=0.5, capsize=0, ecolor="#555555"))
            if len(groups) > 1
            else {}
        )
        ax.bar(
            x - w / 2,
            off_rec,
            w,
            color=C_OFF,
            label="off-recall (outage detected)",
            yerr=err(off_rec, g_off) if len(groups) > 1 else None,
            **kw,
        )
        ax.bar(
            x + w / 2,
            on_rec,
            w,
            color=C_ON,
            label="on-recall (no false alarm)",
            yerr=err(on_rec, g_on) if len(groups) > 1 else None,
            **kw,
        )
        ax.plot(x, off_freq, "k.", ms=3, label="outage frequency of bus")
        ax.set_xticks(x)
        ax.set_xticklabels(bus_ids, fontsize=6, rotation=90)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("bus (feeder order)")
        ax.set_ylabel("recall")
        title = "Per-bus detection performance"
        if len(groups) > 1:
            title += " (whiskers: range over groups)"
        ax.set_title(title)
        ax.legend(loc="lower left", ncol=3)
        fig.savefig(path)
        plt.close(fig)


def plot_voltage_status_sample(st, threshold, path):
    plt = _plt()
    prob, vm, y = st["smp_prob"], st["smp_vm"], st["smp_y"]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        for cls, col, lab in (
            (0, C_OFF, "true de-energised"),
            (1, C_ON, "true energised"),
        ):
            m = y == cls
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
        ax.set_title(f"Voltage-status consistency ({y.size:,} random buses)")
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


GROUP_PLOT_KEYS = [
    ("balanced_accuracy", "balanced accuracy"),
    ("off_recall", "off recall"),
    ("on_recall", "on recall"),
    ("off_precision", "off precision"),
    ("roc_auc", "ROC-AUC"),
    ("ap_deenergised", "AP (de-energised)"),
    ("scenario_exact_match_rate", "scenario exact match"),
    ("vm_rmse_true_on", "ON Vm RMSE"),
]


def plot_group_variation(group_rows, overall, path):
    plt = _plt()
    keys = [(k, t) for k, t in GROUP_PLOT_KEYS if k in overall]
    ncol = 4
    nrow = math.ceil(len(keys) / ncol)
    gi = np.arange(len(group_rows))
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.9 * nrow), squeeze=False)
        for ax, (k, t) in zip(axes.flat, keys):
            v = np.array([r.get(k, np.nan) for r in group_rows], float)
            ax.plot(gi, v, "o", color=C_OFF, ms=3, label="group")
            ax.axhline(overall[k], color=C_MODEL, lw=0.9, label="all groups")
            ax.axhline(
                np.nanmean(v),
                color=C_ON,
                lw=0.9,
                ls="--",
                label="mean of groups",
            )
            ax.set_title(t)
            ax.set_xlabel("group")
            ax.ticklabel_format(axis="y", useOffset=False)
        for ax in list(axes.flat)[len(keys) :]:
            ax.set_visible(False)
        axes.flat[0].legend(loc="best", fontsize=6)
        fig.suptitle(
            f"Metrics per group of scenarios ({len(group_rows)} groups)",
            fontsize=9,
        )
        fig.savefig(path)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def _write_summary(m, group_rows, info, path):
    def fmt(v, d=4):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        return f"{v:.{d}f}" if isinstance(v, float) else f"{v:,}"

    lines = [
        "# VLD evaluation summary, grouped (model predictions at its own threshold)",
        "",
        f"File: `{info['file']}`",
        "",
        f"Buses: {m['n_buses']:,}   Scenarios: {m['n_scenarios']:,}   "
        f"De-energised prevalence: {m['off_rate']:.3f}   Threshold θ = {m['threshold']:.4f}",
        "",
        f"Evaluated in {info['groups_evaluated']} of {info['n_groups']} groups of up to "
        f"{info['group_size']:,} scenarios ({info['grouping']}).",
        "",
    ]
    if info["groups_evaluated"] < info["n_groups"]:
        lines += [
            f"**Partial run:** only the first {info['groups_evaluated']} groups were evaluated "
            "(`--max-groups`); the values below cover those groups only.",
            "",
        ]
    if m.get("status_pred_threshold_mismatch", 0) > 0:
        lines += [
            f"**Warning:** {m['status_pred_threshold_mismatch']:.2%} of exported `status_pred` "
            "values differ from `status_prob >= threshold`; the export and the loss used "
            "different thresholds.",
            "",
        ]
    if (
        info.get("threshold_min") is not None
        and info["threshold_min"] != info["threshold_max"]
    ):
        lines += [
            f"**Warning:** the `threshold` column is not constant "
            f"({info['threshold_min']:.6f} to {info['threshold_max']:.6f}); θ above is the first row's.",
            "",
        ]
    lines += [
        "'All groups' = statistics of all groups summed, i.e. the whole file evaluated at once. "
        "Intervals are cluster-bootstrap 95 % over scenarios (count-based metrics only). "
        "ROC-AUC, AP and ECE use a 65,536-bin probability histogram (accurate to about 1e-4). "
        "'off' = de-energised (label 0).",
        "",
        "| metric | all groups [95 % CI] | mean ± sd over groups | min – max over groups "
        "| corresponding loss / task metric |",
        "|---|---|---|---|---|",
    ]
    for key, loss_name in SUMMARY_ROWS:
        if key not in m:
            continue
        s = fmt(m[key])
        if f"{key}_ci95_lo" in m:
            s += f"  [{m[f'{key}_ci95_lo']:.3f}, {m[f'{key}_ci95_hi']:.3f}]"
        v = np.array([r.get(key, np.nan) for r in group_rows], float)
        if np.isfinite(v).any():
            sd = (
                float(np.nanstd(v, ddof=1))
                if np.isfinite(v).sum() > 1
                else float("nan")
            )
            g1 = f"{np.nanmean(v):.4f} ± {sd:.4f}"
            g2 = f"{np.nanmin(v):.4f} – {np.nanmax(v):.4f}"
        else:
            g1 = g2 = "-"
        lines.append(f"| {key} | {s} | {g1} | {g2} | {loss_name} |")
    lines += [
        "",
        f"Confusion (rows true off/on, cols pred off/on): "
        f"TN_off={m['TN_off']:,}  FP(missed outage)={m['FP_pred_on_true_off']:,}  "
        f"FN(false alarm)={m['FN_pred_off_true_on']:,}  TP_on={m['TP_on']:,}",
        "",
        "Per-group values: `group_metrics.csv`; figure `group_variation`.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def _file_signature(path):
    s = os.stat(path)
    return {"file": os.path.abspath(path), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def evaluate_vld_file_grouped(
    path: str,
    out_dir: str,
    group_size: int = 100_000,
    model_threshold: Optional[float] = None,
    n_boot: int = 1000,
    seed: int = 0,
    batch_rows: int = 1_000_000,
    max_groups: Optional[int] = None,
    resume: bool = True,
    fig_format: str = "png",
    prefix: str = "",
    use_threads: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a VLD prediction file (parquet or CSV) group by group and write the combined
    figures, ``metrics.json``, ``summary.md`` and ``group_metrics.csv`` to ``out_dir``.

    Hard predictions come from ``status_pred`` when present, otherwise
    ``status_prob >= threshold``; ``model_threshold`` defaults to the first value of the
    ``threshold`` column (as in ``vld_evaluation``). Returns the flat metric dict.
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be >= 1")
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    gdir = os.path.join(out_dir, "groups")
    os.makedirs(gdir, exist_ok=True)
    _limit_arrow_threads()

    _log(f"scanning {path}")
    info = scan_file(path, batch_rows, use_threads)
    n_scen = info["n_scenarios"]
    n_groups = math.ceil(n_scen / group_size)
    if model_threshold is None:
        model_threshold = (
            info["threshold_first"] if info["threshold_first"] is not None else 0.5
        )
    model_threshold = float(model_threshold)
    grouping = (
        "scenarios in file order"
        if info["contiguous"]
        else "scenario ids in sorted order; one pass over the file per group"
    )
    _log(
        f"{info['n_rows']:,} rows, {n_scen:,} scenarios, {info['bus_ids'].size} buses, "
        f"threshold {model_threshold:.6f}; {n_groups} groups of up to {group_size:,} scenarios "
        f"({grouping})",
    )
    if not info["contiguous"]:
        _log(
            "WARNING: the rows of some scenarios are not stored together, so each group needs "
            f"its own pass over the file ({n_groups} passes). This is slower but gives the same result.",
        )

    manifest = {
        **_file_signature(path),
        "group_size": group_size,
        "n_groups": n_groups,
        "n_rows": info["n_rows"],
        "n_scenarios": n_scen,
        "contiguous": bool(info["contiguous"]),
        "threshold": model_threshold,
        "seed": seed,
        "columns": [c for c in REQUIRED + OPTIONAL if c in info["columns"]],
        "n_bus_ids": int(info["bus_ids"].size),
    }
    mpath = os.path.join(gdir, "manifest.json")
    gfile = lambda k: os.path.join(gdir, f"group_{k:05d}.npz")  # noqa: E731
    done = set()
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            old = json.load(fh)
        if resume and old == manifest:
            done = {k for k in range(n_groups) if os.path.exists(gfile(k))}
            if done:
                _log(f"resuming: {len(done)} groups already evaluated in {gdir}")
        elif resume:
            raise RuntimeError(
                f"{gdir} holds groups of a different run (file, group size, threshold or seed "
                "changed). Use a new --outdir or --no-resume to recompute.",
            )
    if not resume or not done:
        for f in glob.glob(os.path.join(gdir, "group_*.npz")):
            os.remove(f)
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    cols = manifest["columns"]
    last_group = n_groups if max_groups is None else min(max_groups, n_groups)
    todo = set(range(last_group)) - done
    skip = set(range(n_groups)) - todo
    if todo:
        if info["contiguous"]:
            it = _iter_groups_contiguous(
                path,
                cols,
                group_size,
                n_groups,
                batch_rows,
                use_threads,
                skip,
                last_group,
            )
        else:
            hi_ids = info["scenario_ids"][
                np.minimum(np.arange(1, n_groups + 1) * group_size, n_scen) - 1
            ]
            it = _iter_groups_by_id(
                path,
                cols,
                hi_ids,
                batch_rows,
                use_threads,
                skip,
                last_group,
            )
        for k, d in it:
            tg = time.time()
            n_rows_k = d["scenario"].size
            n_sample = math.ceil(SCATTER_POINTS * n_rows_k / info["n_rows"])
            st = group_statistics(
                d,
                model_threshold,
                info["bus_ids"],
                n_sample,
                np.random.default_rng([seed, k]),
            )
            del d
            tmp = gfile(k)[:-4] + ".tmp.npz"
            np.savez(tmp, **st)
            os.replace(tmp, gfile(k))
            _log(
                f"group {k + 1}/{n_groups}: {st['scen_id'].size:,} scenarios, {n_rows_k:,} rows, "
                f"{time.time() - tg:.0f} s, peak memory {_peak_rss_gb():.2f} GB",
            )
    groups = []
    for k in range(last_group):
        with np.load(gfile(k)) as z:
            groups.append({key: z[key] for key in z.files})
    st = combine(groups)
    n_unique = np.unique(st["scen_id"]).size
    if n_unique != st["scen_id"].size:
        raise RuntimeError("a scenario was split over two groups; please report this")
    if last_group == n_groups and (
        int(st["n_rows"]) != info["n_rows"] or n_unique != n_scen
    ):
        raise RuntimeError(
            "row or scenario count of the groups does not match the scan",
        )

    _log("combining groups and computing the bootstrap intervals")
    metrics = metrics_from_statistics(st, model_threshold)
    for k, (lo, hi) in bootstrap_intervals(st, n_boot=n_boot, seed=seed).items():
        metrics[f"{k}_ci95_lo"] = lo
        metrics[f"{k}_ci95_hi"] = hi
    group_rows = []
    for k, g in enumerate(groups):
        r = {
            "group": k,
            "first_scenario": int(g["scen_id"].min()),
            "last_scenario": int(g["scen_id"].max()),
        }
        r.update(metrics_from_statistics(g, model_threshold))
        group_rows.append(r)
    for key in [k for k, _ in SUMMARY_ROWS] + ["mcc"]:
        v = np.array([r.get(key, np.nan) for r in group_rows], float)
        if key in metrics and np.isfinite(v).any():
            metrics[f"{key}_group_mean"] = float(np.nanmean(v))
            metrics[f"{key}_group_min"] = float(np.nanmin(v))
            metrics[f"{key}_group_max"] = float(np.nanmax(v))
    metrics.update(
        {
            "n_groups": n_groups,
            "groups_evaluated": last_group,
            "group_size": group_size,
        },
    )

    f = lambda name: os.path.join(out_dir, f"{prefix}{name}.{fig_format}")  # noqa: E731
    plot_confusion_counts(
        int(st["tn"]),
        int(st["fp"]),
        int(st["fn"]),
        int(st["tp"]),
        model_threshold,
        f("confusion_matrix"),
    )
    if "roc_auc" in metrics:
        plot_roc_pr_grouped(st, groups, metrics, f("roc_pr"))
    plot_probability_separation_counts(st, model_threshold, f("probability_separation"))
    plot_reliability_counts(st, f("reliability"))
    plot_scenario_errors_counts(st, f("scenario_errors"))
    plot_per_bus_grouped(st, groups, info["bus_ids"], f("per_bus"))
    if "smp_prob" in st:
        plot_voltage_status_sample(st, model_threshold, f("voltage_status"))
    plot_group_variation(group_rows, metrics, f("group_variation"))

    import pandas as pd

    pd.DataFrame(group_rows).to_csv(
        os.path.join(out_dir, f"{prefix}group_metrics.csv"),
        index=False,
    )
    metrics = _jsonable(metrics)
    with open(
        os.path.join(out_dir, f"{prefix}metrics.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(metrics, fh, indent=2)
    _write_summary(
        metrics,
        group_rows,
        {
            **manifest,
            "groups_evaluated": last_group,
            "grouping": grouping,
            "threshold_min": info["threshold_min"],
            "threshold_max": info["threshold_max"],
        },
        os.path.join(out_dir, f"{prefix}summary.md"),
    )
    _log(f"done in {time.time() - t0:.0f} s; results in {out_dir}")
    return metrics


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("predictions", help="parquet or csv written by the VLD task")
    ap.add_argument("--outdir", default="vld_eval_grouped")
    ap.add_argument(
        "--group-size",
        type=int,
        default=100_000,
        help="scenarios per group (default 100000)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the threshold column",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--batch-rows",
        type=int,
        default=1_000_000,
        help="rows read from the file at a time (default 1000000)",
    )
    ap.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="evaluate only the first N groups (trial run)",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="recompute every group even if saved groups exist in OUTDIR/groups",
    )
    ap.add_argument("--fig-format", default="png")
    ap.add_argument(
        "--threads",
        action="store_true",
        help="let pyarrow decode with several threads (off by default for HPC login nodes)",
    )
    args = ap.parse_args(argv)
    m = evaluate_vld_file_grouped(
        args.predictions,
        args.outdir,
        group_size=args.group_size,
        model_threshold=args.threshold,
        n_boot=args.n_boot,
        seed=args.seed,
        batch_rows=args.batch_rows,
        max_groups=args.max_groups,
        resume=not args.no_resume,
        fig_format=args.fig_format,
        use_threads=args.threads,
    )
    with open(os.path.join(args.outdir, "summary.md"), encoding="utf-8") as fh:
        print(fh.read())
    return m


if __name__ == "__main__":
    main()
