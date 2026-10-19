"""
Helpers for the VLD Colab tutorial (examples/tutorial_vld).

Deliberately thin: the tutorial drives the application through its real CLI
(`contingency_data_gen`, `gridfm_app_dist train/predict`) and the real loss
(`VLDTopologyLoss`). These helpers only (a) run those commands from a notebook,
(b) locate MLflow artifacts, (c) build a five-bus toy problem so the loss and
its gradients can be inspected number by number, and (d) draw a few figures.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import yaml


# ---------------------------------------------------------------------------
# Running the application CLIs from a notebook
# ---------------------------------------------------------------------------
def run(
    cmd: Iterable[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> None:
    """Run a command, streaming its output into the notebook; raise on failure."""
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    full_env = dict(os.environ)
    full_env.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    if env:
        full_env.update(env)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=full_env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}",
        )


def python_module(module: str, *args: str) -> list:
    """`python -m <module> ...` using the notebook's interpreter."""
    return [sys.executable, "-m", module, *args]


def latest_run_artifacts(log_dir: Path, experiment_name: str) -> Path:
    """Artifacts folder of the most recent MLflow run of an experiment (file store)."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(Path(log_dir).resolve().as_uri())
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise FileNotFoundError(
            f"no MLflow experiment named {experiment_name!r} under {log_dir}",
        )
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise FileNotFoundError(f"experiment {experiment_name!r} has no runs")
    run_id = runs[0].info.run_id
    return Path(log_dir) / exp.experiment_id / run_id / "artifacts"


def metric_history(
    log_dir: Path,
    experiment_name: str,
    names: Iterable[str],
) -> Dict[str, pd.DataFrame]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(Path(log_dir).resolve().as_uri())
    exp = client.get_experiment_by_name(experiment_name)
    run = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )[0]
    out = {}
    for name in names:
        hist = sorted(
            client.get_metric_history(run.info.run_id, name),
            key=lambda m: (m.step, m.timestamp),
        )
        out[name] = pd.DataFrame(
            {"step": [m.step for m in hist], "value": [m.value for m in hist]},
        )
    return out


def plot_history(
    log_dir: Path,
    experiment_name: str,
    names,
    title: str,
    ncols: int = 2,
):
    import matplotlib.pyplot as plt

    hist = metric_history(log_dir, experiment_name, names)
    names = list(names)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.0 * nrows),
        squeeze=False,
    )
    for ax, name in zip(axes.ravel(), names):
        df = hist[name]
        if len(df):
            ax.plot(df.step, df.value, "o-", ms=3, lw=1.2)
        else:
            ax.text(0.5, 0.5, "no logged values", ha="center", transform=ax.transAxes)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def read_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: Path, cfg: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def count_scenarios(data_dir: Path, network: str) -> int:
    bus = pd.read_parquet(
        Path(data_dir) / network / "raw" / "bus_data.parquet",
        columns=["scenario"],
    )
    return int(bus["scenario"].nunique())


def adapt_pretrained(
    pretrained_dir: Path,
    network: str,
    n_scenarios: int,
    out_dir: Path,
    accelerator: str = "auto",
):
    """
    Point a saved checkpoint's config and normaliser at the dataset generated in this notebook.
    The normaliser stats are keyed by network NAME, so they are re-keyed; the model weights are untouched. Returns (config_path, stats_path, weights_path).
    """
    pretrained_dir, out_dir = Path(pretrained_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_yaml(pretrained_dir / "config.yaml")
    cfg["data"]["networks"] = [network]
    cfg["data"]["scenarios"] = [n_scenarios]
    cfg["data"]["workers"] = 0
    cfg["training"]["accelerator"] = accelerator
    cfg["training"]["devices"] = 1
    cfg["training"]["strategy"] = "auto"
    config_path = write_yaml(out_dir / "config.yaml", cfg)

    stats = torch.load(
        pretrained_dir / "normalizer_stats.pt",
        map_location="cpu",
        weights_only=False,
    )
    if network not in stats:
        if len(stats) != 1:
            raise ValueError(
                f"normalizer_stats.pt holds {list(stats)}; cannot pick one for {network!r}",
            )
        stats = {network: next(iter(stats.values()))}
    stats_path = out_dir / "normalizer_stats.pt"
    torch.save(stats, stats_path)
    return config_path, stats_path, pretrained_dir / "best_model_state_dict.pt"


# ---------------------------------------------------------------------------
# Five-bus toy problem for inspecting the loss and its gradients
# ---------------------------------------------------------------------------
#  REF(0) -- 1 -- 2        3 -- 4      bus 3 is hit by the contingency;
#                                      the 3-4 section has no path to a source.
def toy_problem(
    conflict_bus: Optional[int] = None,
    pred_vm: float = 1.0,
    pred_logit: float = 2.0,
):
    """
    Build inputs for VLDTopologyLoss on a five-bus example.
    ``conflict_bus`` flips the dataset label of one bus so that it disagrees with
    the BFS reachability label (e.g. 4 -> label 1 on an unreachable bus;
    2 -> label 0 on a reachable bus) to show the label-wins rule.
    """
    from gridfm_app_dist.datasets.globals import (
        BRANCH_BASE_STATUS_E,
        BRANCH_CONT_E,
        BUS_BASE_STATUS_H,
        BUS_CONT_H,
        BUS_STATUS_TARGET,
        PQ_H,
        REF_H,
        VM_H,
    )

    x = torch.zeros(5, 17)
    x[:, PQ_H] = 1.0
    x[0, PQ_H] = 0.0
    x[0, REF_H] = 1.0
    x[:, BUS_BASE_STATUS_H] = 1.0
    x[3, BUS_CONT_H] = 1.0  # bus 3 is taken out by the contingency
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    ea = torch.zeros(6, 13)
    ea[:, BRANCH_BASE_STATUS_E] = 1.0
    ea[:, BRANCH_CONT_E] = 0.0
    target = torch.zeros(5, 7)
    target[:, VM_H] = torch.tensor([1.0, 0.99, 0.98, 0.0, 0.0])
    target[:, BUS_STATUS_TARGET] = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    if conflict_bus is not None:
        target[conflict_bus, BUS_STATUS_TARGET] = (
            1.0 - target[conflict_bus, BUS_STATUS_TARGET]
        )
    pred = torch.zeros(5, 5)
    pred[:, 0] = pred_vm
    pred[:, 4] = pred_logit
    pred.requires_grad_(True)
    return x, edges, ea, target, pred


def loss_toy(
    conflict_bus: Optional[int] = None,
    pred_vm: float = 1.0,
    pred_logit: float = 2.0,
    **loss_overrides,
):
    """
    Run the real VLDTopologyLoss on the toy problem and return
    (terms, prediction, gradient, reachability, w_off, threshold).

    ``loss_overrides`` are passed as loss_args (e.g. target_anchor_weight=1.0).
    """
    from gridfm_app_dist.io.param_handler import NestedNamespace
    from gridfm_app_dist.training.loss import VLDTopologyLoss

    x, edges, ea, target, pred = toy_problem(conflict_bus, pred_vm, pred_logit)
    args = dict(
        topology_weight=1.0,
        target_anchor_weight=0.5,
        off_vm_weight=1.0,
        on_vm_weight=0.5,
        unreachable_l1_weight=1.0,
        unreachable_pos_weight="auto",
        max_pos_weight=10.0,
        pos_weight_momentum=0.0,
        reachability_sources="ref",
        input_state_threshold=0.5,
        topology_confidence_gamma=0.0,
    )
    args.update(loss_overrides)
    loss_fn = VLDTopologyLoss(NestedNamespace(**args), None)
    model = SimpleNamespace(latest_x_dict={"bus": x})
    rel = ("bus", "connects", "bus")
    out = loss_fn(
        {"bus": pred},
        {"bus": target},
        {rel: edges},
        {rel: ea},
        {},
        model=model,
    )
    out["loss"].backward()
    reach, _ = VLDTopologyLoss._build_graph_reachability(5, edges, ea, x, x.device)
    return (
        {k: float(v) for k, v in out.items()},
        pred.detach(),
        pred.grad.detach(),
        reach,
        float(out["VLD Off Class Weight"]),
        float(out["VLD Decision Threshold"]),
    )


def toy_table(conflict_bus=None, **loss_overrides) -> pd.DataFrame:
    """Per-bus view of the toy problem: labels, predictions, gradients."""
    from gridfm_app_dist.datasets.globals import BUS_STATUS_TARGET

    terms, pred, grad, reach, w_off, theta = loss_toy(conflict_bus, **loss_overrides)
    _, _, _, target, _ = toy_problem(conflict_bus)
    df = pd.DataFrame(
        {
            "bus": np.arange(5),
            "y_reach": reach.numpy().astype(int),
            "y_target": target[:, BUS_STATUS_TARGET].numpy().astype(int),
            "P(on)": torch.sigmoid(pred[:, 4]).numpy().round(3),
            "V_hat": pred[:, 0].numpy(),
            "dL/dz (status logit)": grad[:, 4].numpy().round(4),
            "dL/dV_hat": grad[:, 0].numpy().round(4),
        },
    )
    df["conflict"] = df["y_reach"] != df["y_target"]
    df.attrs.update(terms=terms, w_off=w_off, threshold=theta)
    return df


def stationary_point(lam_reach, lam_anc, omega, nu, y_reach, y_target):
    """p* of eq. (20): where the two BCE pulls on one bus cancel."""
    return (lam_reach * omega * y_reach + lam_anc * nu * y_target) / (
        lam_reach * omega + lam_anc * nu + 1e-12
    )


# ---------------------------------------------------------------------------
# One-scenario reconstruction plot
# ---------------------------------------------------------------------------
def plot_scenario(pred_df: pd.DataFrame, scenario: int, threshold: float):
    import matplotlib.pyplot as plt

    one = pred_df[pred_df.scenario == scenario].sort_values("bus")
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
    if "vm_true" in one:
        axes[0].plot(one.bus, one.vm_true, "o-", color="#9a9a9a", label="true Vm")
    axes[0].plot(one.bus, one.vm_pred, "x-", color="#d9731a", label="predicted Vm")
    axes[0].set_ylabel("Vm (p.u.)")
    axes[0].legend(loc="lower left")
    axes[1].step(
        one.bus,
        one.status_true,
        where="mid",
        color="#9a9a9a",
        label="true status",
    )
    axes[1].plot(one.bus, one.status_prob, "x-", color="#1f5fa8", label="P(energised)")
    axes[1].axhline(
        threshold,
        ls="--",
        color="k",
        lw=0.9,
        label=rf"$\theta$ = {threshold:.3f}",
    )
    axes[1].set(xlabel="bus", ylabel="status / probability")
    axes[1].legend(loc="center right")
    wrong = one[
        (one.status_prob >= threshold).astype(int) != one.status_true.astype(int)
    ]
    for b in wrong.bus:
        axes[1].axvspan(b - 0.5, b + 0.5, color="#f28e2b", alpha=0.15)
    fig.suptitle(f"Scenario {scenario}: {len(wrong)} misclassified bus(es) shaded")
    fig.tight_layout()
    return fig
