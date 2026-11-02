"""
Unit tests for the reachability-driven VLD loss (reachability lives inside
VLDTopologyLoss; MaskedMSE and PBE are independent of it).

Toy graph (bus ids 0..6, bus 0 is REF):

    0 - 1 - 2 = 3 - 4        "=" : edge 2-3 is hit by the contingency
        |
        5 - 6

Reachable from REF: {0, 1, 2, 5, 6}; unreachable: {3, 4}.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_scatter")
pytest.importorskip("torch_geometric")

from gridfm_app_dist.datasets.globals import (  # noqa: E402
    BRANCH_BASE_STATUS_E,
    BRANCH_CONT_E,
    BUS_BASE_STATUS_H,
    # BUS_CONT_H,
    BUS_STATUS_LOGIT_OUT,
    B_ON,
    PQ_H,
    REF_H,
    VM_H,
    VM_OUT,
    YFF_TT_I,
    YFF_TT_R,
    YFT_TF_I,
    YFT_TF_R,
)
from gridfm_app_dist.io.param_handler import NestedNamespace  # noqa: E402
from gridfm_app_dist.training.loss import VLDTopologyLoss  # noqa: E402

N_BUS = 7
BUS_FEAT = 17
EDGE_FEAT = 13
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6)]
HIT = (2, 3)
REACH = [1, 1, 1, 0, 0, 1, 1]


class FakeModel:
    def __init__(self, bus_x):
        self.latest_x_dict = {"bus": bus_x}


def _graph():
    bus_x = torch.zeros(N_BUS, BUS_FEAT)
    bus_x[:, BUS_BASE_STATUS_H] = 1.0
    bus_x[:, PQ_H] = 1.0
    bus_x[0, REF_H] = 1.0
    bus_x[0, PQ_H] = 0.0
    src = [a for a, b in EDGES] + [b for a, b in EDGES]
    dst = [b for a, b in EDGES] + [a for a, b in EDGES]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.zeros(edge_index.size(1), EDGE_FEAT)
    edge_attr[:, B_ON] = 1.0
    edge_attr[:, BRANCH_BASE_STATUS_E] = 1.0
    edge_attr[:, YFF_TT_R] = 1.0
    edge_attr[:, YFF_TT_I] = -10.0
    edge_attr[:, YFT_TF_R] = -1.0
    edge_attr[:, YFT_TF_I] = 10.0
    for k, (a, b) in enumerate(zip(src, dst)):
        if {a, b} == set(HIT):
            edge_attr[k, BRANCH_CONT_E] = 1.0
    return bus_x, edge_index, edge_attr


def _args(**vld_args):
    vld = {
        "topology_weight": 1.0,
        "target_anchor_weight": 0.5,
        "off_vm_weight": 1.0,
        "on_vm_weight": 0.5,
        "unreachable_l1_weight": 1.0,
        "unreachable_pos_weight": "auto",
        "pos_weight_momentum": 0.0,
        "max_pos_weight": 10.0,
    }
    vld.update(vld_args)
    return NestedNamespace(
        training={
            "losses": ["MaskedMSE", "PBE", "VLDTopologyLoss"],
            "loss_args": [{}, {}, vld],
            "loss_weights": [0.1, 0.4, 0.5],
        },
    )


def _batch(label=None, vm_pred=None, logit=None):
    bus_x, edge_index, edge_attr = _graph()
    label = torch.tensor(REACH if label is None else label, dtype=torch.float)
    vm_true = label * 1.0  # de-energised buses carry Vm = 0 in the data
    target_bus = torch.zeros(N_BUS, 7)
    target_bus[:, VM_H] = vm_true
    target_bus[:, 5] = label  # BUS_STATUS_TARGET
    pred_bus = torch.zeros(N_BUS, 5)
    pred_bus[:, VM_OUT] = torch.full((N_BUS,), 0.9) if vm_pred is None else vm_pred
    pred_bus[:, BUS_STATUS_LOGIT_OUT] = torch.zeros(N_BUS) if logit is None else logit
    pred_bus.requires_grad_(True)
    mask_bus = torch.zeros(N_BUS, BUS_FEAT, dtype=torch.bool)
    mask_bus[1:, VM_H] = True  # every PQ bus has its Vm masked (predicted)
    pred = {"bus": pred_bus, "gen": torch.zeros(0, 1, requires_grad=True)}
    target = {"bus": target_bus, "gen": torch.zeros(0, 7)}
    eid = {
        ("bus", "connects", "bus"): edge_index,
        ("gen", "connected_to", "bus"): torch.zeros(2, 0, dtype=torch.long),
    }
    eattr = {("bus", "connects", "bus"): edge_attr}
    mask = {"bus": mask_bus, "gen": torch.zeros(0, 7, dtype=torch.bool)}
    return pred, target, eid, eattr, mask, FakeModel(bus_x)


def test_reachability_from_input_flags():
    bus_x, edge_index, edge_attr = _graph()
    reach, avail = VLDTopologyLoss._build_graph_reachability(
        num_bus=N_BUS,
        edge_index=edge_index,
        edge_attr=edge_attr,
        bus_x=bus_x,
        device=bus_x.device,
    )
    assert reach.tolist() == REACH
    assert avail.tolist() == [1.0] * N_BUS


def test_status_and_voltage_terms_are_independent():
    args = _args()

    # voltage terms only -> no gradient reaches the status logit
    loss_v = VLDTopologyLoss(
        _args(topology_weight=0.0, target_anchor_weight=0.0).training.loss_args[2],
        args,
    )
    pred, target, eid, eattr, mask, model = _batch()
    loss_v(pred, target, eid, eattr, mask, model)["loss"].backward()
    assert torch.all(pred["bus"].grad[:, BUS_STATUS_LOGIT_OUT] == 0)
    assert torch.any(pred["bus"].grad[:, VM_OUT] != 0)

    # status terms only -> no gradient reaches Vm
    loss_s = VLDTopologyLoss(
        _args(off_vm_weight=0.0, on_vm_weight=0.0).training.loss_args[2],
        args,
    )
    pred, target, eid, eattr, mask, model = _batch()
    loss_s(pred, target, eid, eattr, mask, model)["loss"].backward()
    assert torch.all(pred["bus"].grad[:, VM_OUT] == 0)
    assert torch.any(pred["bus"].grad[:, BUS_STATUS_LOGIT_OUT] != 0)


def test_voltage_terms_do_not_depend_on_status_prediction():
    args = _args()
    loss = VLDTopologyLoss(args.training.loss_args[2], args)
    a = loss(*_batch(logit=torch.full((N_BUS,), -8.0)))  # everything predicted off
    b = loss(*_batch(logit=torch.full((N_BUS,), 8.0)))  # everything predicted on
    assert torch.isclose(a["VLD On Vm Loss"], b["VLD On Vm Loss"])
    assert torch.isclose(a["VLD Off Vm Loss"], b["VLD Off Vm Loss"])


def test_voltage_masks_follow_reachability_not_the_label():
    args = _args()
    loss = VLDTopologyLoss(args.training.loss_args[2], args)
    # label says bus 3 is energised (a conflict); reachability says it is not
    label = list(REACH)
    label[3] = 1
    vm = torch.full((N_BUS,), 0.9)
    base = loss(*_batch(label=label, vm_pred=vm))["VLD Off Vm Loss"]
    vm2 = vm.clone()
    vm2[3] = 0.0  # unreachable bus pushed to 0 -> off-Vm loss falls
    assert loss(*_batch(label=label, vm_pred=vm2))["VLD Off Vm Loss"] < base
    vm3 = vm.clone()
    vm3[1] = 0.0  # reachable bus: not in the off-Vm term
    assert torch.isclose(
        loss(*_batch(label=label, vm_pred=vm3))["VLD Off Vm Loss"],
        base,
    )


def test_class_weight_and_anchor_come_from_reachability():
    args = _args()
    loss = VLDTopologyLoss(args.training.loss_args[2], args)
    # labels all energised: the class weight must still be n_on/n_off of REACHABILITY
    out = loss(*_batch(label=[1] * N_BUS))
    assert torch.isclose(out["VLD Off Class Weight"], torch.tensor(5.0 / 2.0))
    # no conflict -> the anchor is exactly 0
    out = loss(*_batch())
    assert out["VLD Target Anchor BCE"] == 0
    # a conflict activates the anchor, and the topology term still covers the bus
    label = list(REACH)
    label[3] = 1
    logit = torch.zeros(N_BUS)
    a = loss(*_batch(label=label, logit=logit))
    assert a["VLD Target Anchor BCE"] > 0
    logit2 = logit.clone()
    logit2[3] = -5.0  # agree with reachability on the conflict bus
    b = loss(*_batch(label=label, logit=logit2))
    assert b["VLD Topology BCE"] < a["VLD Topology BCE"]
    assert b["VLD Target Anchor BCE"] > a["VLD Target Anchor BCE"]
