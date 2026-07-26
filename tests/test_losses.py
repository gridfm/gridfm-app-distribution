import pytest
import torch
from gridfm_dist_app.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_dist_app.io.param_handler import NestedNamespace
from gridfm_dist_app.datasets.globals import VM_H, VA_H, QG_H, PG_BUS_H
from torch_scatter import scatter_add
from gridfm_dist_app.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)


@pytest.fixture
def small_grid_data_module():
    # Load config
    import yaml

    with open("tests/config/datamodule_test_base_config.yaml") as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")

    # Fake trainer for setup
    class DummyTrainer:
        is_global_zero = True

    dm.trainer = DummyTrainer()
    dm.setup("train")
    return dm


def test_pbe_loss_zero_with_real_data(small_grid_data_module):
    loader = small_grid_data_module.train_dataloader()
    batch = next(iter(loader))

    branch_flow_layer = ComputeBranchFlow()
    node_injection_layer = ComputeNodeInjection()
    node_residuals_layer = ComputeNodeResiduals()

    num_bus = batch.x_dict["bus"].size(0)
    bus_edge_index = batch.edge_index_dict[("bus", "connects", "bus")]
    bus_edge_attr = batch.edge_attr_dict[("bus", "connects", "bus")]

    # Build the PBE target entirely from the bus label so Pg and Qg come from the SAME source and per-unit base
    target = torch.stack(
        [
            batch.y_dict["bus"][:, VM_H],       # VM_OUT = 0
            batch.y_dict["bus"][:, VA_H],       # VA_OUT = 1
            batch.y_dict["bus"][:, PG_BUS_H],   # PG_OUT = 2  (net bus Pg, same base as Qg)
            batch.y_dict["bus"][:, QG_H],       # QG_OUT = 3
        ],
        dim=1,
    )

    Pft, Qft = branch_flow_layer(target, bus_edge_index, bus_edge_attr)
    P_in, Q_in = node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
    residual_P, residual_Q = node_residuals_layer(
        P_in,
        Q_in,
        target,
        batch.x_dict["bus"],
    )
    # Power balance only holds on ENERGIZED buses.
    # In contingency scenarios an islanded sub-network is de-energized (Vm == 0), where residuals are physically meaningless.
    # Mask to include only energized buses before asserting.
    energized = batch.x_dict["bus"][:, VM_H] > 1e-6
    assert torch.max(torch.abs(residual_P[energized])) < 1e-4, (
        f"Active residuals nonzero on energized buses! {torch.max(torch.abs(residual_P[energized]))}"
    )
    assert torch.max(torch.abs(residual_Q[energized])) < 1e-4, (
        f"Reactive residuals nonzero on energized buses! {torch.max(torch.abs(residual_Q[energized]))}"
    )