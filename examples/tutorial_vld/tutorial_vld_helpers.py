"""
Helpers for the VLD Colab tutorial (examples/tutorial_vld).

Deliberately thin: the tutorial drives the application through its real CLI
(`contingency_data_gen`, `gridfm_app_dist train/predict`) and the real loss
(`VLDTopologyLoss`). These helpers only
(a) run those commands from a notebook,
(b) locate MLflow artifacts,
(c) build loss inputs for one generated contingency scenario so the loss and its gradients can be
inspected number by number, and
(d) draw a few figures.
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


# Loss settings used when inspecting the loss on a single sample (scenario or toy).
TUTORIAL_LOSS_ARGS = dict(
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


# ---------------------------------------------------------------------------
# One generated contingency scenario as VLD loss inputs
# ---------------------------------------------------------------------------
def _scenario_frames(bus_df: pd.DataFrame, branch_df: pd.DataFrame, scenario: int):
    b = bus_df[bus_df.scenario == scenario].sort_values("bus").reset_index(drop=True)
    br = branch_df[branch_df.scenario == scenario].reset_index(drop=True)
    return b, br


def scenario_problem(
    bus_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    scenario: int,
    conflict_bus: Optional[int] = None,
    pred_vm: float = 1.0,
    pred_logit: float = 2.0,
):
    """
    Build VLDTopologyLoss inputs (same layout as ``toy_problem``) from one
    scenario of the generated gridfm-datakit parquet (``bus_data`` /
    ``branch_data``). Rows are buses in ascending bus-id order; ``conflict_bus``
    is a row position and flips that bus's dataset label.
    """
    from gridfm_app_dist.datasets.globals import (
        BRANCH_BASE_STATUS_E,
        BRANCH_CONT_E,
        BUS_BASE_STATUS_H,
        BUS_CONT_H,
        BUS_STATUS_TARGET,
        PQ_H,
        PV_H,
        REF_H,
        VM_H,
    )

    b, br = _scenario_frames(bus_df, branch_df, scenario)
    n = len(b)
    pos = {bid: i for i, bid in enumerate(b["bus"].to_numpy())}

    def col(df, name, fill):
        return torch.tensor(
            df[name].fillna(fill).to_numpy(dtype=float),
            dtype=torch.float,
        )

    x = torch.zeros(n, 17)
    x[:, PQ_H] = col(b, "PQ", 0)
    x[:, PV_H] = col(b, "PV", 0)
    x[:, REF_H] = col(b, "REF", 0)
    x[:, BUS_BASE_STATUS_H] = col(b, "BUS_BASE_STATUS", 1)
    x[:, BUS_CONT_H] = col(b, "BUS_CONTINGENCY", 0)

    edges = torch.tensor(
        np.stack(
            [br["from_bus"].map(pos).to_numpy(), br["to_bus"].map(pos).to_numpy()],
        ),
        dtype=torch.long,
    )
    ea = torch.zeros(len(br), 13)
    ea[:, BRANCH_BASE_STATUS_E] = col(br, "BRANCH_BASE_STATUS", 1)
    ea[:, BRANCH_CONT_E] = col(br, "BRANCH_CONTINGENCY", 0)

    target = torch.zeros(n, 7)
    target[:, VM_H] = col(b, "Vm", 0)
    target[:, BUS_STATUS_TARGET] = col(b, "BUS_STATUS", 0)
    if conflict_bus is not None:
        target[conflict_bus, BUS_STATUS_TARGET] = (
            1.0 - target[conflict_bus, BUS_STATUS_TARGET]
        )

    pred = torch.zeros(n, 5)
    pred[:, 0] = pred_vm
    pred[:, 4] = pred_logit
    pred.requires_grad_(True)
    return b, x, edges, ea, target, pred


def scenario_reachability(bus_df, branch_df, scenario):
    """BFS reachability and availability of one scenario (the loss's own BFS)."""
    from gridfm_app_dist.training.loss import VLDTopologyLoss

    b, x, edges, ea, _, _ = scenario_problem(bus_df, branch_df, scenario)
    reach, avail = VLDTopologyLoss._build_graph_reachability(
        len(b),
        edges,
        ea,
        x,
        x.device,
        threshold=TUTORIAL_LOSS_ARGS["input_state_threshold"],
        sources=TUTORIAL_LOSS_ARGS["reachability_sources"],
    )
    return reach.numpy().astype(int), avail.numpy().astype(int)


def pick_scenario(bus_df, branch_df) -> int:
    """
    Pick an instructive scenario: one that has all four bus roles (source,
    energised non-source bus, directly hit bus, healthy-but-islanded bus), with
    the energised and islanded groups as balanced as possible. On a radial
    feeder, maximising the islanded count alone would pick an outage next to
    the source that blacks out everything but the REF bus.
    """
    best, best_key = None, None
    branch_groups = dict(tuple(branch_df.groupby("scenario")))
    for sid, b in bus_df.groupby("scenario"):
        br = branch_groups.get(sid, branch_df.iloc[:0])
        reach, avail = scenario_reachability(b, br, sid)
        b = b.sort_values("bus")
        is_ref = b["REF"].fillna(0).to_numpy() > 0.5
        n_on = int(((reach == 1) & ~is_ref).sum())
        hit = int((avail == 0).sum())
        islanded = int(((avail == 1) & (reach == 0)).sum())
        has_all = int(n_on > 0 and hit > 0 and islanded > 0)
        key = (has_all, min(n_on, islanded), islanded, -int(sid))
        if best_key is None or key > best_key:
            best, best_key = int(sid), key
    return best


def scenario_roles(tbl: pd.DataFrame) -> Dict[str, Optional[int]]:
    """
    Row positions of representative buses in a ``scenario_table``:
    ``ref`` (source), ``on`` (an energised non-source bus), ``hit`` (a bus taken
    out by the contingency) and ``islanded`` (healthy but no path to a source).
    """

    def first(mask):
        idx = tbl.index[mask]
        return int(idx[0]) if len(idx) else None

    is_ref = tbl["REF"] == 1
    on = (tbl.y_reach == 1) & ~is_ref
    return {
        "ref": first(is_ref),
        "on": first(on),
        "hit": first(tbl.available == 0),
        "islanded": first((tbl.available == 1) & (tbl.y_reach == 0)),
    }


def scenario_table(
    bus_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    scenario: int,
    conflict_bus: Optional[int] = None,
    **loss_overrides,
) -> pd.DataFrame:
    """
    Per-bus view of one real scenario under the real VLDTopologyLoss: labels,
    initial predictions (V_hat = 1, z = 2) and gradients. Same columns and
    ``attrs`` as ``toy_table`` plus ``available``, ``REF`` and ``V_true``.
    """
    from gridfm_app_dist.datasets.globals import BUS_STATUS_TARGET, VM_H
    from gridfm_app_dist.io.param_handler import NestedNamespace
    from gridfm_app_dist.training.loss import VLDTopologyLoss

    b, x, edges, ea, target, pred = scenario_problem(
        bus_df,
        branch_df,
        scenario,
        conflict_bus,
    )
    args = dict(TUTORIAL_LOSS_ARGS)
    args.update(loss_overrides)
    loss_fn = VLDTopologyLoss(NestedNamespace(**args), None)
    rel = ("bus", "connects", "bus")
    out = loss_fn(
        {"bus": pred},
        {"bus": target},
        {rel: edges},
        {rel: ea},
        {},
        model=SimpleNamespace(latest_x_dict={"bus": x}),
    )
    out["loss"].backward()
    reach, avail = VLDTopologyLoss._build_graph_reachability(
        len(b),
        edges,
        ea,
        x,
        x.device,
        threshold=args["input_state_threshold"],
        sources=args["reachability_sources"],
    )
    pred_d, grad = pred.detach(), pred.grad.detach()
    df = pd.DataFrame(
        {
            "bus": b["bus"].to_numpy(),
            "REF": b["REF"].fillna(0).astype(int).to_numpy(),
            "available": avail.numpy().astype(int),
            "y_reach": reach.numpy().astype(int),
            "y_target": target[:, BUS_STATUS_TARGET].numpy().astype(int),
            "V_true": target[:, VM_H].numpy().round(4),
            "P(on)": torch.sigmoid(pred_d[:, 4]).numpy().round(3),
            "V_hat": pred_d[:, 0].numpy(),
            "dL/dz (status logit)": grad[:, 4].numpy().round(4),
            "dL/dV_hat": grad[:, 0].numpy().round(4),
        },
    )
    df["conflict"] = df["y_reach"] != df["y_target"]
    terms = {k: float(v) for k, v in out.items()}
    df.attrs.update(
        terms=terms,
        w_off=terms["VLD Off Class Weight"],
        threshold=terms["VLD Decision Threshold"],
        scenario=int(scenario),
    )
    return df


def plot_reachability(bus_df, branch_df, scenario, ax=None):
    """Draw the feeder for one scenario, coloured by BFS reachability."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    from gridfm_app_dist.datasets.globals import BRANCH_BASE_STATUS_E, BRANCH_CONT_E

    b, x, edges, ea, _, _ = scenario_problem(bus_df, branch_df, scenario)
    reach, avail = scenario_reachability(bus_df, branch_df, scenario)
    base = (ea[:, BRANCH_BASE_STATUS_E] > 0.5).numpy()
    hit = (ea[:, BRANCH_CONT_E] > 0.5).numpy()
    e = edges.T.numpy()

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(range(len(b)))
    layout_graph.add_edges_from(map(tuple, e[base]))
    try:
        pos = nx.kamada_kawai_layout(layout_graph)
    except Exception:  # scipy missing or disconnected base graph
        pos = nx.spring_layout(layout_graph, seed=0)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))
    ref = b["REF"].fillna(0).to_numpy() > 0.5
    colors = [
        "#1f5fa8" if s else ("#d9731a" if r else ("#9a9a9a" if a == 0 else "white"))
        for r, a, s in zip(reach, avail, ref)
    ]
    nx.draw_networkx_edges(
        layout_graph,
        pos,
        edgelist=[tuple(t) for t in e[base & ~hit]],
        ax=ax,
        width=1.4,
    )
    nx.draw_networkx_edges(
        layout_graph,
        pos,
        edgelist=[tuple(t) for t in e[base & hit]],
        ax=ax,
        style="dashed",
        edge_color="red",
        width=2.0,
    )
    nx.draw_networkx_nodes(
        layout_graph,
        pos,
        ax=ax,
        node_color=colors,
        edgecolors="black",
        node_size=300,
    )
    nx.draw_networkx_labels(
        layout_graph,
        pos,
        labels={i: str(v) for i, v in enumerate(b["bus"])},
        ax=ax,
        font_size=7,
    )
    handles = [
        Line2D([], [], marker="o", ls="", mfc=c, mec="black", ms=9, label=lab)
        for c, lab in [
            ("#1f5fa8", "source (REF)"),
            ("#d9731a", "reachable -> energised"),
            ("#9a9a9a", "hit by contingency"),
            ("white", "healthy but islanded"),
        ]
    ] + [Line2D([], [], color="red", ls="--", label="outaged branch")]
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.set_title(f"case33bw, scenario {scenario}: BFS reachability from the source")
    ax.axis("off")
    return ax


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
    args = dict(TUTORIAL_LOSS_ARGS)
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
