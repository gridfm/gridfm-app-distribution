"""
Phase 1 of the VLD improvement plan (items 1.2-1.7): unit tests on toy graphs.

  1.2  PBELoss: slack power from the label, closed switches merged into
       electrical nodes, energised scope, per-graph load normalisation,
       legacy path unchanged.
  1.3  MaskedMSE channels.
  1.4  model.edge_input_transform.
  1.5  AddVLDHeteroMask(ref_vm_input, mask_pv_vm).
  1.6  non-finite guard, gradient clipping kwargs, no non-finite export.
  1.7  prediction file naming and metadata.

Toy power-flow graph (float64, consistent by construction):

    0 (REF, ext_grid) -- 1 == 2 -- 3        4 (isolated, de-energised)

"==" is a closed switch (all admittance entries 0, V1 = V2). The labels are
derived from S_inj = V conj(Y V) computed here with a dense complex Y, so with
the true voltages the balance holds exactly; the load of the switch pair is
split unevenly between buses 1 and 2 (only the sum balances, as in the data).
"""

import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_scatter")
pytest.importorskip("torch_geometric")

from gridfm_app_dist.datasets.globals import (  # noqa: E402
    BRANCH_BASE_STATUS_E,
    BS,
    BUS_STATUS_TARGET,
    B_ON,
    GS,
    PD_H,
    PG_BUS_H,
    PG_H,
    PQ_H,
    PV_H,
    QD_H,
    QG_H,
    QG_OUT,
    REF_H,
    VA_H,
    VA_OUT,
    VM_H,
    VM_OUT,
    YFF_TT_I,
    YFF_TT_R,
    YFT_TF_I,
    YFT_TF_R,
)
from gridfm_app_dist.io.param_handler import NestedNamespace  # noqa: E402
from gridfm_app_dist.training.loss import MaskedMSELoss, PBELoss  # noqa: E402

DT = torch.float64
BB = ("bus", "connects", "bus")
GB = ("gen", "connected_to", "bus")
N = 5
LINES = [(0, 1, complex(2.0, -8.0)), (2, 3, complex(1.5, -6.0))]
SWITCHES = [(1, 2)]
V_TRUE = [
    (1.00, 0.00),
    (0.98, -0.02),
    (0.98, -0.02),  # same node as bus 1
    (0.97, -0.03),
    (0.00, 0.00),  # de-energised
]
SPLIT = complex(0.05, 0.02)  # load moved from bus 2 to bus 1 (sum unchanged)
OFF_LOAD = complex(0.10, 0.03)  # unsupplied in-service load of bus 4


def _vld_args():
    return NestedNamespace(task={"task_name": "VoltageLossDetection"})


class FakeModel:
    def __init__(self, bus_x, bus_batch=None):
        self.latest_x_dict = {"bus": bus_x}
        self.latest_bus_batch = bus_batch


def _edges(scale=1.0):
    src, dst, attrs = [], [], []
    for a, b, y in LINES:
        for f, t in ((a, b), (b, a)):
            e = torch.zeros(13, dtype=DT)
            e[YFF_TT_R], e[YFF_TT_I] = scale * y.real, scale * y.imag
            e[YFT_TF_R], e[YFT_TF_I] = -scale * y.real, -scale * y.imag
            src.append(f)
            dst.append(t)
            attrs.append(e)
    for a, b in SWITCHES:
        for f, t in ((a, b), (b, a)):
            src.append(f)
            dst.append(t)
            attrs.append(torch.zeros(13, dtype=DT))
    ea = torch.stack(attrs)
    ea[:, B_ON] = 1.0
    ea[:, BRANCH_BASE_STATUS_E] = 1.0
    return torch.tensor([src, dst]), ea


def _s_inj_dense(V, ei, ea):
    """Reference S = V conj(Y V) with a dense complex Y (independent of the loss)."""
    Y = torch.zeros(N, N, dtype=torch.complex128)
    for k in range(ei.size(1)):
        i, j = int(ei[0, k]), int(ei[1, k])
        Y[i, i] += complex(float(ea[k, YFF_TT_R]), float(ea[k, YFF_TT_I]))
        Y[i, j] += complex(float(ea[k, YFT_TF_R]), float(ea[k, YFT_TF_I]))
    return V * torch.conj(Y @ V)


def _graph(scale=1.0):
    """One consistent toy scenario; powers and admittances scaled by ``scale``."""
    ei, ea = _edges(scale)
    vm = torch.tensor([v for v, _ in V_TRUE], dtype=DT)
    va = torch.tensor([a for _, a in V_TRUE], dtype=DT)
    V = vm * torch.exp(1j * va)
    S = _s_inj_dense(V, ei, ea)

    y = torch.zeros(N, 7, dtype=DT)
    y[:, VM_H], y[:, VA_H] = vm, va
    y[:, BUS_STATUS_TARGET] = torch.tensor([1, 1, 1, 1, 0], dtype=DT)
    # PQ buses: S_net = -S_load = S_inj
    for b in (1, 2, 3):
        y[b, PD_H], y[b, QD_H] = -S[b].real, -S[b].imag
    y[1, PD_H] += scale * SPLIT.real
    y[1, QD_H] += scale * SPLIT.imag
    y[2, PD_H] -= scale * SPLIT.real
    y[2, QD_H] -= scale * SPLIT.imag
    # slack bus (ext_grid, no generator rows): labelled injection
    y[0, PG_BUS_H], y[0, QG_H] = S[0].real, S[0].imag
    # de-energised bus keeps its in-service load in the labels
    y[4, PD_H], y[4, QD_H] = scale * OFF_LOAD.real, scale * OFF_LOAD.imag

    x = torch.zeros(N, 17, dtype=DT)
    x[:, PQ_H] = 1.0
    x[0, PQ_H], x[0, REF_H] = 0.0, 1.0
    ref = x[:, REF_H] > 0.5
    pq = x[:, PQ_H] > 0.5

    mask_bus = torch.zeros(N, 17, dtype=torch.bool)
    mask_bus[pq, VM_H] = True
    mask_bus[pq, VA_H] = True
    mask_bus[ref, VM_H] = True
    mask_bus[ref, QG_H] = True
    return {
        "x": x,
        "y": y,
        "ei": ei,
        "ea": ea,
        "mask": {
            "bus": mask_bus,
            "gen": torch.zeros(0, 7, dtype=torch.bool),
            "REF": ref,
            "PV": torch.zeros(N, dtype=torch.bool),
        },
    }


def _perfect_pred(g):
    pred = torch.zeros(N, 5, dtype=DT)
    pred[:, VM_OUT] = g["y"][:, VM_H]
    pred[:, VA_OUT] = g["y"][:, VA_H]
    # PG_OUT / QG_OUT left at 0: the decoded slack values are not needed with
    # slack_power: label, and are exactly what made the legacy floor
    return pred.requires_grad_(True)


def _call(loss, g, pred, bus_batch=None):
    return loss(
        {"bus": pred, "gen": torch.zeros(0, 1, dtype=DT)},
        {"bus": g["y"], "gen": torch.zeros(0, 7, dtype=DT)},
        {BB: g["ei"], GB: torch.zeros(2, 0, dtype=torch.long)},
        {BB: g["ea"]},
        g["mask"],
        FakeModel(g["x"], bus_batch),
    )


def _pbe(**opts):
    return PBELoss(NestedNamespace(**opts), None)


# ------------------------------------------------------------------ 1.2 PBE
def test_pbe_defaults_legacy_for_other_tasks_and_new_for_vld():
    assert PBELoss().legacy
    assert PBELoss(
        NestedNamespace(),
        NestedNamespace(task={"task_name": "PowerFlow"}),
    ).legacy
    vld = PBELoss(NestedNamespace(), _vld_args())
    assert not vld.legacy
    assert vld.slack_power == "label" and vld.merge_switches
    assert vld.bus_scope == "energised" and vld.normalisation == "none"
    with pytest.raises(ValueError):
        _pbe(slack_power="decoder")
    with pytest.raises(ValueError):
        _pbe(bus_scope="on")
    with pytest.raises(ValueError):
        _pbe(normalisation="max")


def test_real_injection_matches_complex_reference():
    g = _graph()
    torch.manual_seed(0)
    vm = 0.9 + 0.2 * torch.rand(N, dtype=DT)
    va = 0.1 * torch.randn(N, dtype=DT)
    V = vm * torch.exp(1j * va)
    gs = 0.01 * torch.rand(N, dtype=DT)
    bs = 0.02 * torch.rand(N, dtype=DT)
    p, q = PBELoss._injection(V.real, V.imag, g["ei"], g["ea"], N, gs, bs)
    ref = _s_inj_dense(V, g["ei"], g["ea"]) + V * torch.conj((gs + 1j * bs) * V)
    assert torch.allclose(p, ref.real, atol=1e-12)
    assert torch.allclose(q, ref.imag, atol=1e-12)


def test_electrical_nodes_are_switch_components():
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    is_switch = torch.tensor([False, True, True, False, True, False])
    node, n = PBELoss.electrical_nodes(ei, is_switch, 8)
    # groups: {1,2,3}, {4,5}, and singletons 0, 6, 7
    assert n == 5
    assert node[1] == node[2] == node[3]
    assert node[4] == node[5]
    assert len({int(node[i]) for i in (0, 1, 4, 6, 7)}) == 5
    node, n = PBELoss.electrical_nodes(ei, torch.zeros(6, dtype=torch.bool), 8)
    assert n == 8 and node.tolist() == list(range(8))


def test_legacy_pbe_has_slack_and_switch_floor():
    g = _graph()
    out = _call(PBELoss(), g, _perfect_pred(g))
    assert out["loss"] > 1e-2  # ext_grid P (and decoded Q = 0) missing at REF


def test_slack_label_removes_the_slack_floor():
    g = _graph()
    out = _call(_pbe(slack_power="label"), g, _perfect_pred(g))
    # left: the switch split |SPLIT| on buses 1 and 2 and the off load on bus 4
    expected = (2 * abs(SPLIT) + abs(OFF_LOAD)) / N
    assert math.isclose(float(out["Power loss in p.u."]), expected, rel_tol=1e-6)


def test_switch_merge_balances_the_electrical_node():
    g = _graph()
    out = _call(
        _pbe(slack_power="label", merge_switches=True),
        g,
        _perfect_pred(g),
    )
    # 4 electrical nodes {0}, {1,2}, {3}, {4}; only the off-load floor remains
    assert math.isclose(
        float(out["Power loss in p.u."]),
        abs(OFF_LOAD) / 4,
        rel_tol=1e-5,
    )


def test_energised_scope_is_zero_at_the_truth_and_gradients_are_finite():
    g = _graph()
    pred = _perfect_pred(g)
    out = _call(
        _pbe(slack_power="label", merge_switches=True, bus_scope="energised"),
        g,
        pred,
    )
    assert float(out["Power loss in p.u."]) < 1e-5
    out["loss"].backward()
    assert torch.isfinite(pred.grad).all()


def test_pbe_reacts_to_voltage_error():
    g = _graph()
    loss = _pbe(slack_power="label", merge_switches=True, bus_scope="energised")
    base = float(_call(loss, g, _perfect_pred(g))["loss"])
    pred = _perfect_pred(g).detach().clone()
    pred[3, VM_OUT] += 0.01
    assert float(_call(loss, g, pred)["loss"]) > base + 1e-3


def _two_graph_batch(scale):
    a, b = _graph(1.0), _graph(scale)
    off = N
    g = {
        "x": torch.cat([a["x"], b["x"]]),
        "y": torch.cat([a["y"], b["y"]]),
        "ei": torch.cat([a["ei"], b["ei"] + off], dim=1),
        "ea": torch.cat([a["ea"], b["ea"]]),
        "mask": {
            k: torch.cat([a["mask"][k], b["mask"][k]])
            for k in ("bus", "gen", "REF", "PV")
        },
    }
    return g, torch.cat(
        [torch.zeros(N, dtype=torch.long), torch.ones(N, dtype=torch.long)],
    )


def _call2(loss, g, pred, bus_batch):
    return loss(
        {"bus": pred, "gen": torch.zeros(0, 1, dtype=DT)},
        {"bus": g["y"], "gen": torch.zeros(0, 7, dtype=DT)},
        {BB: g["ei"], GB: torch.zeros(2, 0, dtype=torch.long)},
        {BB: g["ea"]},
        g["mask"],
        FakeModel(g["x"], bus_batch),
    )


def test_load_normalisation_makes_networks_comparable():
    scale = 1000.0
    g, bus_batch = _two_graph_batch(scale)
    pred = torch.zeros(2 * N, 5, dtype=DT)
    pred[:, VM_OUT], pred[:, VA_OUT] = g["y"][:, VM_H], g["y"][:, VA_H]
    pred[3, VM_OUT] += 0.01  # same voltage error in both graphs
    pred[N + 3, VM_OUT] += 0.01
    opts = dict(slack_power="label", merge_switches=True, bus_scope="energised")

    one, _ = _two_graph_batch(1.0)  # the same batch with both graphs unscaled
    raw = _call2(_pbe(**opts), g, pred, bus_batch)
    raw_one = _call2(_pbe(**opts), one, pred, bus_batch)
    norm = _call2(_pbe(normalisation="load", **opts), g, pred, bus_batch)
    norm_one = _call2(_pbe(normalisation="load", **opts), one, pred, bus_batch)

    # un-normalised: the scaled graph dominates (~scale/2 x); normalised: identical
    assert float(raw["loss"]) > 100 * float(raw_one["loss"])
    assert math.isclose(float(norm["loss"]), float(norm_one["loss"]), rel_tol=1e-9)
    assert "Power loss (load-normalised)" in norm
    assert math.isclose(
        float(norm["Power loss in p.u."]),
        float(raw["Power loss in p.u."]),
        rel_tol=1e-12,
    )


def test_load_weight_is_inverse_mean_load_per_graph():
    g, bus_batch = _two_graph_batch(10.0)
    node = torch.arange(2 * N)
    w = PBELoss._load_weight(FakeModel(g["x"], bus_batch), g["y"], node, 2 * N, DT)
    s = torch.sqrt(g["y"][:, PD_H] ** 2 + g["y"][:, QD_H] ** 2)
    for k in (0, 1):
        sel = (bus_batch == k) & (s > 0)
        assert torch.allclose(w[bus_batch == k], 1.0 / s[sel].mean().expand(N))


def test_slack_label_needs_pg_bus_column():
    g = _graph()
    g["y"] = g["y"][:, :PG_BUS_H]
    with pytest.raises(ValueError, match="PG_BUS_H"):
        _call(_pbe(slack_power="label"), g, _perfect_pred(g))


def test_shunts_enter_the_new_path():
    g = _graph()
    g["x"][3, GS] = 0.05
    g["x"][3, BS] = -0.02
    opts = dict(slack_power="label", merge_switches=True, bus_scope="energised")
    with_sh = float(_call(_pbe(**opts), g, _perfect_pred(g))["loss"])
    without = float(
        _call(_pbe(include_shunts=False, **opts), g, _perfect_pred(g))["loss"],
    )
    assert without < 1e-5 < with_sh


# ------------------------------------------------------------------ 1.3 MaskedMSE
def _mse_batch():
    torch.manual_seed(1)
    pred = {"bus": torch.randn(4, 5), "gen": torch.randn(2, 1)}
    target = {"bus": torch.randn(4, 7), "gen": torch.randn(2, 7)}
    mask_bus = torch.zeros(4, 17, dtype=torch.bool)
    mask_bus[:, VM_H] = True
    mask_bus[1:, VA_H] = True
    mask_bus[0, QG_H] = True
    mask_gen = torch.zeros(2, 7, dtype=torch.bool)
    mask_gen[0, PG_H] = True
    return pred, target, {"bus": mask_bus, "gen": mask_gen}


def test_masked_mse_default_channels():
    pred, target, mask = _mse_batch()
    legacy = MaskedMSELoss(NestedNamespace(), None)
    assert legacy.channels == ("Vm", "Va", "Qg", "gen_Pg")
    vld = MaskedMSELoss(NestedNamespace(), _vld_args())
    assert vld.channels == ("Vm", "Va")

    pb, tb, mb = pred["bus"], target["bus"], mask["bus"]
    sel_p = [pb[:, VM_OUT][mb[:, VM_H]], pb[:, VA_OUT][mb[:, VA_H]]]
    sel_t = [tb[:, VM_H][mb[:, VM_H]], tb[:, VA_H][mb[:, VA_H]]]
    expected_vld = torch.mean((torch.cat(sel_p) - torch.cat(sel_t)) ** 2)
    assert torch.isclose(vld(pred, target, None, None, mask)["loss"], expected_vld)

    sel_p += [pb[:, QG_OUT][mb[:, QG_H]], pred["gen"][:, :1][mask["gen"][:, :1]]]
    sel_t += [tb[:, QG_H][mb[:, QG_H]], target["gen"][:, :1][mask["gen"][:, :1]]]
    expected_all = torch.mean((torch.cat(sel_p) - torch.cat(sel_t)) ** 2)
    assert torch.isclose(legacy(pred, target, None, None, mask)["loss"], expected_all)


def test_masked_mse_channel_validation():
    assert MaskedMSELoss(NestedNamespace(channels="Vm"), None).channels == ("Vm",)
    for bad in (["Vm", "Pd"], []):
        with pytest.raises(ValueError):
            MaskedMSELoss(NestedNamespace(channels=bad), None)


# ------------------------------------------------------------------ 1.5 masking
def _hetero():
    from torch_geometric.data import HeteroData

    data = HeteroData()
    x = torch.zeros(3, 17)
    x[0, REF_H], x[1, PV_H], x[2, PQ_H] = 1.0, 1.0, 1.0
    x[:, VM_H] = torch.tensor([1.02, 0.0, 0.97])
    data["bus"].x = x
    data["gen"].x = torch.zeros(1, 7)
    data[BB].edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data[BB].edge_attr = torch.zeros(4, 13)
    data[GB].edge_index = torch.tensor([[0], [1]])
    data[("bus", "connected_to", "gen")].edge_index = torch.tensor([[1], [0]])
    return data


def test_vld_mask_defaults_unchanged_and_flags():
    from gridfm_app_dist.datasets.masking import AddVLDHeteroMask

    default = AddVLDHeteroMask()(_hetero()).mask_dict["bus"]
    assert default[0, VM_H] and not default[1, VM_H] and default[2, VM_H]

    ref_in = AddVLDHeteroMask(ref_vm_input=True)(_hetero()).mask_dict["bus"]
    assert not ref_in[0, VM_H]
    assert torch.equal(ref_in[1:], default[1:])

    pv = AddVLDHeteroMask(mask_pv_vm=True)(_hetero()).mask_dict["bus"]
    assert pv[1, VM_H]
    diff = pv ^ default
    assert diff.sum() == 1


def test_task_transforms_pass_the_flags():
    from gridfm_app_dist.datasets.masking import AddVLDHeteroMask
    from gridfm_app_dist.datasets.task_transforms import VoltageLossDetectionTransforms

    args = NestedNamespace(
        task={
            "task_name": "VoltageLossDetection",
            "ref_vm_input": True,
            "mask_pv_vm": True,
        },
        data={"mask_value": 0.0},
    )
    masks = [
        t
        for t in VoltageLossDetectionTransforms(args).transforms
        if isinstance(t, AddVLDHeteroMask)
    ]
    assert len(masks) == 1 and masks[0].ref_vm_input and masks[0].mask_pv_vm


# ------------------------------------------------------------------ 1.4 edge input
def _model(**model_opts):
    from gridfm_app_dist.models.gnn_heterogeneous_gns import GNS_heterogeneous

    model = {
        "num_layers": 1,
        "hidden_size": 8,
        "input_bus_dim": 17,
        "input_gen_dim": 7,
        "output_bus_dim": 5,
        "output_gen_dim": 1,
        "edge_dim": 13,
        "attention_head": 2,
    }
    model.update(model_opts)
    return GNS_heterogeneous(
        NestedNamespace(model=model, task={"task_name": "VoltageLossDetection"}),
    )


def test_edge_input_transform():
    ea = torch.randn(6, 13) * 1e4
    plain = _model()
    assert plain.edge_input(BB, ea) is ea

    m = _model(edge_input_transform="asinh", edge_input_scale=10.0)
    out = m.edge_input(BB, ea)
    cols = [YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I]
    other = [c for c in range(13) if c not in cols]
    assert torch.allclose(out[:, cols], torch.asinh(ea[:, cols] / 10.0))
    assert torch.equal(out[:, other], ea[:, other])
    assert torch.equal(m.edge_input(GB, ea), ea)  # only bus-bus edges

    with pytest.raises(ValueError):
        _model(edge_input_transform="log")
    with pytest.raises(ValueError):
        _model(edge_input_transform="asinh", edge_input_scale=0.0)


# ------------------------------------------------------------------ 1.6 stability
def _task(max_nonfinite_steps=3):
    pytest.importorskip("lightning")
    from gridfm_app_dist.tasks.base_task import BaseTask

    class Tiny(BaseTask):
        def __init__(self, args):
            super().__init__(args, [])
            self.lin = torch.nn.Linear(2, 1)

        def forward(self, x):
            return self.lin(x)

        def training_step(self, batch):
            return None

        def validation_step(self, batch, batch_idx):
            return None

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            return None

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            return None

    return Tiny(NestedNamespace(training={"max_nonfinite_steps": max_nonfinite_steps}))


def test_non_finite_loss_becomes_zero_gradient_and_step_is_skipped():
    task = _task()
    loss = task.guard_non_finite_loss(torch.tensor(float("nan")))
    assert loss.requires_grad and float(loss) == 0.0
    loss.backward()
    assert all(torch.equal(p.grad, torch.zeros_like(p)) for p in task.parameters())
    task.on_before_optimizer_step(None)
    assert all(p.grad is None for p in task.parameters())  # AdamW skips them


def test_finite_step_keeps_gradients_and_resets_the_streak():
    task = _task()
    out = task(torch.ones(1, 2)).sum()
    loss = task.guard_non_finite_loss(out)
    assert loss is out
    loss.backward()
    task._nonfinite_streak = 2
    task.on_before_optimizer_step(None)
    assert all(p.grad is not None for p in task.parameters())
    assert task._nonfinite_streak == 0


def test_non_finite_gradient_is_skipped_and_streak_stops_training():
    task = _task(max_nonfinite_steps=2)
    for step in range(2):
        task.guard_non_finite_loss(torch.tensor(1.0))
        for p in task.parameters():
            p.grad = torch.full_like(p, float("inf"))
        if step == 0:
            task.on_before_optimizer_step(None)
            assert all(p.grad is None for p in task.parameters())
        else:
            with pytest.raises(RuntimeError, match="consecutive"):
                task.on_before_optimizer_step(None)


def test_non_finite_state_dicts_are_detected():
    from gridfm_app_dist.training.callbacks import non_finite_tensors

    sd = {
        "a": torch.ones(2),
        "b": torch.tensor([1.0, float("nan")]),
        "c": torch.tensor([1]),
    }
    assert non_finite_tensors(sd) == ["b"]


def test_best_export_skips_non_finite_weights(tmp_path):
    pytest.importorskip("lightning")
    from gridfm_app_dist.training.callbacks import SaveBestModelStateDict

    cb = SaveBestModelStateDict(monitor="m", mode="max")
    module = torch.nn.Linear(2, 1)
    with torch.no_grad():
        module.weight[0, 0] = float("nan")
    trainer = SimpleNamespace(
        sanity_checking=False,
        callback_metrics={"m": torch.tensor(0.9)},
        current_epoch=3,
        logger=SimpleNamespace(save_dir=str(tmp_path)),
    )
    cb.on_validation_end(trainer, module)
    assert not cb.has_best and cb.best_model_path is None
    with torch.no_grad():
        module.weight[0, 0] = 1.0
    cb.on_validation_end(trainer, module)
    assert cb.has_best and cb.best_epoch == 3


def test_gradient_clipping_kwargs():
    pytest.importorskip("lightning")
    from gridfm_app_dist.cli import gradient_clipping_kwargs

    vld = NestedNamespace(training={}, task={"task_name": "VoltageLossDetection"})
    assert gradient_clipping_kwargs(vld) == {
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
    }
    off = NestedNamespace(
        training={"gradient_clip_val": None},
        task={"task_name": "VoltageLossDetection"},
    )
    assert gradient_clipping_kwargs(off) == {}
    pf = NestedNamespace(training={}, task={"task_name": "PowerFlow"})
    assert gradient_clipping_kwargs(pf) == {}
    bad = NestedNamespace(
        training={"gradient_clip_algorithm": "x"},
        task={"task_name": "VoltageLossDetection"},
    )
    with pytest.raises(ValueError):
        gradient_clipping_kwargs(bad)


# ------------------------------------------------------------------ 1.7 predictions
def test_prediction_tag_and_metadata(tmp_path):
    pytest.importorskip("lightning")
    pd = pytest.importorskip("pandas")
    from gridfm_app_dist.cli import prediction_tag, write_prediction_metadata

    ckpt = tmp_path / "urban LV 0.8w_500ep.pt"
    ckpt.write_bytes(b"weights")
    args = SimpleNamespace(
        command="predict",
        model_path=str(ckpt),
        config=str(tmp_path / "cfg.yaml"),
        normalizer_stats=None,
        data_path="data",
    )
    tag, path = prediction_tag(args)
    assert tag == "urban_LV_0.8w_500ep" and path == str(ckpt)

    assert prediction_tag(
        SimpleNamespace(command="train"),
        ("/x/best.pt", 7, 0.9, "m"),
    ) == (
        "best",
        "/x/best.pt",
    )
    assert prediction_tag(SimpleNamespace(command="train"), None)[0] == "last"

    df = pd.DataFrame({"scenario": [0, 0, 1], "threshold": [0.4, 0.4, 0.4]})
    out = tmp_path / f"net_{tag}_predictions.parquet"
    write_prediction_metadata(str(out), df, args, {"seed": 1}, "net", tag, path)
    meta = json.loads((tmp_path / f"net_{tag}_predictions.json").read_text())
    assert meta["decision_threshold"] == pytest.approx(0.4)
    assert meta["scenarios"] == 2 and meta["rows"] == 3
    assert meta["checkpoint_tag"] == tag and meta["checkpoint_sha256"]
    assert meta["config"] == {"seed": 1}
